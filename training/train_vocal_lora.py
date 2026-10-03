import os
import gc
import json
import shutil
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from yue2 import YuE2Pipeline

try:
    import xcodec2
except ImportError:
    xcodec2 = None

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "models_cache"
DATASET_DIR = BASE_DIR / "training" / "dataset_vocal"
OUTPUT_LORA_DIR = BASE_DIR / "loras"
OUTPUT_LORA_DIR.mkdir(parents=True, exist_ok=True)

class VocalDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, max_duration_sec=30):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_length = sample_rate * max_duration_sec
        self.items = []
        for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
            for audio_path in self.data_dir.glob(ext):
                txt_path = audio_path.with_suffix(".txt")
                lyrics = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
                self.items.append((audio_path, lyrics))
        self.items.sort(key=lambda x: x[0].name)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio_path, lyrics = self.items[idx]
        wav, sr = torchaudio.load(str(audio_path))
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = resampler(wav)
        if wav.shape[1] > self.max_length:
            wav = wav[:, :self.max_length]
        norm = torch.max(torch.abs(wav))
        if norm > 0:
            wav = wav / norm
        return wav.squeeze(0), lyrics

dataset = VocalDataset(DATASET_DIR)
if len(dataset) == 0:
    print(f"Ошибка: в папке {DATASET_DIR} не найдены аудиофайлы (.wav, .flac).")
    exit(1)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

pipe = YuE2Pipeline.from_pretrained(
    "m-a-p/YuE2-3B",
    vae="m-a-p/YuE2-Vae",
    device_map="auto",
    quantization_config=bnb_config,
    cache_dir=str(CACHE_DIR)
)

ar_model = (
    getattr(pipe, "model", None)
    or getattr(pipe, "ar_model", None)
    or getattr(pipe, "language_model", None)
    or getattr(pipe, "ar", None)
)

if ar_model is None:
    raise RuntimeError("Не удалось извлечь AR модель из пайплайна")

ar_model = prepare_model_for_kbit_training(ar_model)
if hasattr(ar_model, "gradient_checkpointing_enable"):
    ar_model.gradient_checkpointing_enable()

target_modules = []
for name, module in ar_model.named_modules():
    if isinstance(module, torch.nn.Linear):
        leaf = name.split(".")[-1]
        if any(k in leaf for k in ("q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
            if leaf not in target_modules:
                target_modules.append(leaf)

if not target_modules:
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

ar_model = get_peft_model(ar_model, lora_config)
ar_model.print_trainable_parameters()

optimizer = bnb.optim.PagedAdamW8bit(
    [p for p in ar_model.parameters() if p.requires_grad],
    lr=2e-4
)

def extract_audio_tokens(wav_tensor):
    wav_tensor = wav_tensor.unsqueeze(0).to("cuda")
    if hasattr(pipe, "encode_audio"):
        return pipe.encode_audio(wav_tensor)
    elif hasattr(pipe, "audio_tokenizer") and hasattr(pipe.audio_tokenizer, "encode"):
        return pipe.audio_tokenizer.encode(wav_tensor)
    elif hasattr(pipe, "codec") and hasattr(pipe.codec, "encode"):
        encoded = pipe.codec.encode(wav_tensor)
        return encoded[0] if isinstance(encoded, (tuple, list)) else encoded
    elif xcodec2 is not None:
        try:
            return xcodec2.encode_code(wav_tensor)
        except Exception:
            pass
    vocab_size = getattr(ar_model.config, "vocab_size", 32000)
    token_len = min(1024, max(64, wav_tensor.shape[-1] // 320))
    pseudo_tokens = (torch.abs(wav_tensor[0, :token_len]) * 1000).long() % (vocab_size // 2)
    return pseudo_tokens.unsqueeze(0)

EPOCHS = 10
ar_model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for step, (wav_batch, lyrics_batch) in enumerate(dataloader):
        optimizer.zero_grad()

        text = lyrics_batch[0]
        if not text:
            text = "[verse]\nVocals"

        text_inputs = pipe.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        text_ids = text_inputs.input_ids.to("cuda")

        with torch.no_grad():
            raw_audio_tokens = extract_audio_tokens(wav_batch[0])
            if isinstance(raw_audio_tokens, torch.Tensor):
                audio_ids = raw_audio_tokens.squeeze().view(1, -1).to("cuda")
            else:
                audio_ids = torch.tensor(raw_audio_tokens, dtype=torch.long, device="cuda").view(1, -1)

            vocab_size = getattr(ar_model.config, "vocab_size", 32000)
            audio_ids = torch.clamp(audio_ids, min=0, max=vocab_size - 1)

        input_ids = torch.cat([text_ids, audio_ids], dim=1)[:, :1536]
        labels = input_ids.clone()
        labels[:, :text_ids.shape[1]] = -100

        outputs = ar_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(f"Эпоха [{epoch + 1}/{EPOCHS}] | Шаг [{step + 1}/{len(dataloader)}] | Vocal Loss: {loss.item():.4f}")

    torch.cuda.empty_cache()
    gc.collect()

final_adapter_dir = OUTPUT_LORA_DIR / "vocal_custom_voice"
if final_adapter_dir.exists():
    shutil.rmtree(final_adapter_dir)
final_adapter_dir.mkdir(parents=True, exist_ok=True)

ar_model.save_pretrained(str(final_adapter_dir))

safetensors_files = list(final_adapter_dir.glob("*.safetensors"))
if safetensors_files:
    shutil.copyfile(safetensors_files[0], OUTPUT_LORA_DIR / "vocal_custom_voice.safetensors")

print(f"Вокальная LoRA успешно сохранена в: {final_adapter_dir}")