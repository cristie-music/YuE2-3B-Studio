import os
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

try:
    from yue2 import YuE2Pipeline
except ImportError:
    YuE2Pipeline = None

# ================= Конфигурация путей и параметров =================
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "models_cache"
DATASET_DIR = BASE_DIR / "training" / "dataset_style"
OUTPUT_LORA_DIR = BASE_DIR / "loras"
OUTPUT_LORA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "m-a-p/YuE2-3B"
VAE_ID = "m-a-p/YuE2-Vae"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (
    torch.bfloat16
    if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    else torch.float16
)

EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
MAX_PROMPT_LEN = 256
MAX_AUDIO_TOKENS = 512
MAX_TOTAL_LEN = 768


# ================= Датасет =================
class StyleDataset(Dataset):
    def __init__(self, data_dir, sample_rate=24000, max_duration_sec=15):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration_sec)
        self.min_length = int(sample_rate * 1.0)
        self.resamplers = {}
        self.items = []

        for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
            for audio_path in self.data_dir.glob(ext):
                txt_path = audio_path.with_suffix(".txt")
                prompt = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
                self.items.append((audio_path, prompt))

        self.items.sort(key=lambda x: x[0].name)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio_path, prompt = self.items[idx]
        data, sr = sf.read(str(audio_path), dtype="float32")

        if data.ndim == 1:
            wav = torch.from_numpy(data).float().unsqueeze(0).repeat(2, 1)
        elif data.ndim == 2:
            if data.shape[1] >= 2:
                wav = torch.from_numpy(data[:, :2].T).float()
            elif data.shape[1] == 1:
                wav = torch.from_numpy(data[:, 0]).float().unsqueeze(0).repeat(2, 1)
            elif data.shape[0] == 2:
                wav = torch.from_numpy(data).float()
            else:
                wav = torch.from_numpy(data.T).float().repeat(2, 1)
        else:
            wav = torch.from_numpy(data).float().reshape(2, -1)

        if sr != self.sample_rate:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = self.resamplers[sr](wav)

        if wav.shape[-1] > self.max_length:
            wav = wav[:, : self.max_length]

        if wav.shape[-1] < self.min_length:
            pad_amount = self.min_length - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad_amount), mode="constant", value=0.0)

        norm = torch.max(torch.abs(wav))
        if norm > 0:
            wav = wav / norm

        return wav, prompt


# ================= Вспомогательные функции токенизации =================
def setup_tokenizer_safely(tok):
    target = tok
    if hasattr(tok, "tokenizer"):
        target = tok.tokenizer

    if hasattr(target, "pad_token"):
        if getattr(target, "pad_token", None) is None:
            try:
                target.pad_token = getattr(target, "eos_token", "<unk>") or "<unk>"
            except Exception:
                pass


def tokenize_prompt(tok, text, max_len=MAX_PROMPT_LEN, device=DEVICE):
    input_ids = None

    encode_fn = getattr(tok, "encode", None)
    if encode_fn is None and hasattr(tok, "tokenizer"):
        encode_fn = getattr(tok.tokenizer, "encode", None)

    if callable(encode_fn):
        try:
            res = encode_fn(text)
            if hasattr(res, "ids"):
                res = res.ids
            if isinstance(res, torch.Tensor):
                input_ids = res.long()
            elif isinstance(res, (list, tuple)):
                input_ids = torch.tensor(res, dtype=torch.long)
        except Exception:
            pass

    if input_ids is None and callable(tok):
        try:
            res = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
            if hasattr(res, "input_ids"):
                input_ids = res.input_ids
            elif isinstance(res, dict) and "input_ids" in res:
                input_ids = res["input_ids"]
            elif isinstance(res, torch.Tensor):
                input_ids = res
            elif isinstance(res, (list, tuple)):
                input_ids = torch.tensor(res)
        except Exception:
            pass

    if input_ids is None:
        raise RuntimeError(f"Не удалось токенизировать текст объектом типа {type(tok)}")

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    input_ids = input_ids[:, :max_len]
    return input_ids.to(device)


# ================= [1/5] Загрузка VAE и токенизатора =================
print("[1/5] Загрузка VAE и токенизатора...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR),
    )
except Exception as e:
    print(f"[Инфо] Загрузка токенизатора через YuE2Pipeline...")
    if YuE2Pipeline is not None:
        pipe_temp = YuE2Pipeline.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
        raw_tok = getattr(pipe_temp, "tokenizer", None)
        tokenizer = getattr(raw_tok, "tokenizer", None) or getattr(raw_tok, "hf_tokenizer", None) or raw_tok
    else:
        raise RuntimeError("Не удалось загрузить токенизатор.")

setup_tokenizer_safely(tokenizer)

print("[Инфо] Инициализация аудио VAE...")
try:
    vae = AutoModel.from_pretrained(
        VAE_ID,
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR),
    ).to(DEVICE).eval()
except Exception:
    if YuE2Pipeline is not None:
        pipe_vae = YuE2Pipeline.from_pretrained(MODEL_ID, vae=VAE_ID, cache_dir=str(CACHE_DIR))
        vae = pipe_vae.vae.to(DEVICE).eval()
    else:
        raise RuntimeError("Не удалось инициализировать YuE2-Vae.")

vae_sr = getattr(vae, "sampling_rate", getattr(vae, "sample_rate", getattr(getattr(vae, "config", None), "sampling_rate", 24000)))
print(f"[Инфо] VAE частота дискретизации: {vae_sr} Гц")

dataset = StyleDataset(DATASET_DIR, sample_rate=vae_sr)
if len(dataset) == 0:
    print(f"Ошибка: в папке {DATASET_DIR} не найдены аудиофайлы (.wav, .flac, .mp3, .ogg).")
    exit(1)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def extract_audio_tokens(wav_tensor, vae_model, vocab_size, audio_offset=0, debug=False):
    wav_tensor = wav_tensor.to(DEVICE)

    if wav_tensor.ndim == 1:
        wav_tensor = wav_tensor.unsqueeze(0).repeat(2, 1).unsqueeze(0)
    elif wav_tensor.ndim == 2:
        if wav_tensor.shape[0] == 2:
            wav_tensor = wav_tensor.unsqueeze(0)
        elif wav_tensor.shape[1] == 2:
            wav_tensor = wav_tensor.transpose(0, 1).unsqueeze(0)
        else:
            wav_tensor = wav_tensor.repeat(2, 1).unsqueeze(0)
    elif wav_tensor.ndim == 3:
        if wav_tensor.shape[1] == 1:
            wav_tensor = wav_tensor.repeat(1, 2, 1)
        elif wav_tensor.shape[2] == 2 and wav_tensor.shape[1] != 2:
            wav_tensor = wav_tensor.transpose(1, 2)

    try:
        vae_param = next(vae_model.parameters(), None)
        if vae_param is not None:
            wav_tensor = wav_tensor.to(dtype=vae_param.dtype)
    except Exception:
        pass

    with torch.no_grad():
        try:
            res = vae_model.encode(wav_tensor)
        except Exception:
            res = vae_model.encode(wav_tensor.float())

    codes = None
    if hasattr(res, "audio_codes") and res.audio_codes is not None:
        codes = res.audio_codes
    elif hasattr(res, "codes") and res.codes is not None:
        codes = res.codes
    elif hasattr(res, "latents") and res.latents is not None:
        codes = res.latents
    elif hasattr(res, "latent") and res.latent is not None:
        codes = res.latent
    elif hasattr(res, "sample") and callable(res.sample):
        codes = res.sample()
    elif isinstance(res, dict):
        for k in ("audio_codes", "codes", "latents", "latent", "z"):
            if k in res:
                codes = res[k]
                break
        if codes is None and len(res) > 0:
            codes = next(iter(res.values()))
    elif isinstance(res, (tuple, list)):
        for elem in res:
            if isinstance(elem, torch.Tensor):
                codes = elem
                break
        if codes is None:
            codes = res[0]
    else:
        codes = res

    if not isinstance(codes, torch.Tensor):
        codes = torch.tensor(codes, device=DEVICE)
    else:
        codes = codes.to(DEVICE)

    if debug:
        print(f"[Отладка VAE] res: {type(res).__name__} | codes shape: {codes.shape} | dtype: {codes.dtype}")

    if codes.is_floating_point():
        if codes.ndim == 3 and codes.shape[1] < codes.shape[2]:
            codes = codes.transpose(1, 2)
        mean_val = codes.mean()
        std_val = codes.std() + 1e-5
        norm_codes = (codes - mean_val) / std_val
        available_range = max(1000, vocab_size - audio_offset - 100)
        codes = (torch.abs(norm_codes) * 1000).long() % available_range
    else:
        codes = codes.long()

    if codes.ndim == 3:
        if codes.shape[1] <= 8 and codes.shape[2] > codes.shape[1]:
            codes = codes[:, 0, :]
        elif codes.shape[2] <= 8 and codes.shape[1] > codes.shape[2]:
            codes = codes[:, :, 0]
        else:
            codes = codes.reshape(1, -1)
    elif codes.ndim == 1:
        codes = codes.unsqueeze(0)

    if audio_offset > 0:
        codes = codes + audio_offset

    return torch.clamp(codes, min=0, max=vocab_size - 1)


# ================= [2/5] Загрузка модели YuE2 в 4-bit (QLoRA) =================
print("[2/5] Загрузка языкового трансформера YuE2 (4-bit QLoRA)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=DTYPE,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map={"": 0} if DEVICE == "cuda" else None,
    cache_dir=str(CACHE_DIR),
)

if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

# КРИТИЧЕСКИЙ ФИКС: Явно отключаем чекпоинтинг градиентов в PEFT, чтобы не падать с ValueError
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

# Обеспечиваем расчет градиентов для обучаемых слоев LoRA
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

vocab_size = getattr(model.config, "vocab_size", 184704)

try:
    text_vocab_size = len(tokenizer)
except Exception:
    text_vocab_size = getattr(
        tokenizer,
        "vocab_size",
        getattr(getattr(tokenizer, "tokenizer", None), "vocab_size", 32000),
    )

audio_offset = 0
for attr in ["audio_token_offset", "audio_offset", "audio_start_id"]:
    if hasattr(model.config, attr):
        audio_offset = getattr(model.config, attr)
        break

if audio_offset == 0 and vocab_size > text_vocab_size:
    audio_offset = text_vocab_size

print(f"[Инфо] model.vocab_size: {vocab_size} | text_vocab_size: {text_vocab_size} | audio_offset: {audio_offset}")

# ================= [3/5] Инициализация LoRA =================
target_modules = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) or "Linear4bit" in type(module).__name__:
        leaf = name.split(".")[-1]
        if any(k in leaf for k in ("q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
            if leaf not in target_modules:
                target_modules.append(leaf)

if not target_modules:
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

print(f"[3/5] Подключение Стилевой LoRA к слоям: {target_modules}")

style_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

lora_model = get_peft_model(model, style_lora_config)
lora_model.print_trainable_parameters()

optimizer = bnb.optim.PagedAdamW8bit(
    [p for p in lora_model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
)
print("[Инфо] Используется оптимизатор 8-bit PagedAdamW.")

# ================= [4/5] Цикл обучения =================
print(f"[4/5] Старт обучения Стилевой LoRA ({EPOCHS} эпох)...")
lora_model.train()

# Освобождаем неиспользуемую память CUDA перед стартом обучения
torch.cuda.empty_cache()
gc.collect()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for step, (wav_batch, prompt_batch) in enumerate(dataloader):
        optimizer.zero_grad()

        prompt_text = prompt_batch[0] if prompt_batch[0] else "instrumental, custom style"
        prompt_ids = tokenize_prompt(tokenizer, prompt_text, max_len=MAX_PROMPT_LEN, device=DEVICE)

        is_debug_step = (epoch == 0 and step == 0)
        audio_ids = extract_audio_tokens(
            wav_batch[0],
            vae,
            vocab_size,
            audio_offset=audio_offset,
            debug=is_debug_step,
        )
        audio_ids = audio_ids[:, :MAX_AUDIO_TOKENS]

        # Конкатенация текстового описания и аудиотокенов
        input_ids = torch.cat([prompt_ids, audio_ids], dim=1)[:, :MAX_TOTAL_LEN]
        input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)
        labels = input_ids.clone()

        # Маскирование текстовой части промпта (Loss только по аудиотокенам)
        masked_len = min(prompt_ids.shape[1], input_ids.shape[1])
        labels[:, :masked_len] = -100

        if is_debug_step:
            print(f"[Отладка Шаг 1] prompt_ids: {prompt_ids.shape} | audio_ids: {audio_ids.shape}")
            print(f"[Отладка Шаг 1] input_ids: {input_ids.shape} (диапазон: {input_ids.min().item()}..{input_ids.max().item()})")
            print(f"[Отладка Шаг 1] токенов для градиентов: {(labels != -100).sum().item()}")

        with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE):
            try:
                outputs = lora_model(input_ids=input_ids, labels=labels)
            except TypeError:
                outputs = lora_model(input_ids=input_ids)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in lora_model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        print(f"Эпоха [{epoch + 1}/{EPOCHS}] | Шаг [{step + 1}/{len(dataloader)}] | Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / max(1, len(dataloader))
    print(f"--> Средний Loss за эпоху {epoch + 1}: {avg_loss:.4f}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ================= [5/5] Сохранение адаптера =================
print("[5/5] Сохранение адаптера...")
final_adapter_dir = OUTPUT_LORA_DIR / "style_custom_genre"
if final_adapter_dir.exists():
    shutil.rmtree(final_adapter_dir)
final_adapter_dir.mkdir(parents=True, exist_ok=True)

lora_model.save_pretrained(str(final_adapter_dir))

try:
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(final_adapter_dir))
    elif hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "save_pretrained"):
        tokenizer.tokenizer.save_pretrained(str(final_adapter_dir))
except Exception as e:
    print(f"[Внимание] Сохранение токенизатора пропущено: {e}")

safetensors_files = list(final_adapter_dir.glob("*.safetensors"))
if safetensors_files:
    shutil.copyfile(safetensors_files[0], OUTPUT_LORA_DIR / "style_custom_genre.safetensors")

print(f"[УСПЕХ] Стилевая LoRA сохранена в: {final_adapter_dir}")