import os
import gc
import json
import shutil
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
from yue2 import YuE2Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "models_cache"
DATASET_DIR = BASE_DIR / "training" / "dataset_style"
OUTPUT_LORA_DIR = BASE_DIR / "loras"
OUTPUT_LORA_DIR.mkdir(parents=True, exist_ok=True)

class StyleDataset(Dataset):
    def __init__(self, data_dir, sample_rate=48000, max_duration_sec=15):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.target_len = sample_rate * max_duration_sec
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
        wav, sr = torchaudio.load(str(audio_path))
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = resampler(wav)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]
        if wav.shape[1] > self.target_len:
            wav = wav[:, :self.target_len]
        elif wav.shape[1] < self.target_len:
            wav = torch.nn.functional.pad(wav, (0, self.target_len - wav.shape[1]))
        return wav, prompt

dataset = StyleDataset(DATASET_DIR)
if len(dataset) == 0:
    print(f"Ошибка: в папке {DATASET_DIR} не найдены аудиофайлы (.wav, .flac).")
    exit(1)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

pipe = YuE2Pipeline.from_pretrained(
    "m-a-p/YuE2-3B",
    vae="m-a-p/YuE2-Vae",
    device="cuda",
    backend="torch-eager",
    cache_dir=str(CACHE_DIR)
)

nar_model = getattr(pipe, "nar_model", None) or getattr(pipe, "diffusion_model", None) or getattr(pipe, "nar", None)
if nar_model is None:
    raise RuntimeError("Не удалось обнаружить диффузионный NAR-блок в YuE2Pipeline")

for param in nar_model.parameters():
    param.requires_grad = False

target_modules = []
for name, module in nar_model.named_modules():
    if isinstance(module, torch.nn.Linear):
        leaf = name.split(".")[-1]
        if any(k in leaf for k in ("to_q", "to_k", "to_v", "to_out", "q_proj", "v_proj", "out_proj")):
            if leaf not in target_modules:
                target_modules.append(leaf)

if not target_modules:
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

style_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none"
)

nar_model = get_peft_model(nar_model, style_lora_config)
nar_model.print_trainable_parameters()

optimizer = bnb.optim.PagedAdamW8bit(
    [p for p in nar_model.parameters() if p.requires_grad],
    lr=1e-4
)

EPOCHS = 10
nar_model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for step, (audio_batch, prompts) in enumerate(dataloader):
        optimizer.zero_grad()
        audio_tensor = audio_batch.to("cuda", dtype=torch.float32)

        with torch.no_grad():
            if hasattr(pipe, "vae") and pipe.vae is not None:
                if hasattr(pipe.vae, "encode"):
                    encoded = pipe.vae.encode(audio_tensor)
                    if hasattr(encoded, "latent_dist"):
                        latents = encoded.latent_dist.sample()
                    elif hasattr(encoded, "sample"):
                        latents = encoded.sample()
                    elif hasattr(encoded, "latents"):
                        latents = encoded.latents
                    elif isinstance(encoded, (tuple, list)):
                        latents = encoded[0]
                    else:
                        latents = encoded
                else:
                    latents = pipe.vae(audio_tensor)
            else:
                latents = audio_tensor

            model_dtype = next(nar_model.parameters()).dtype
            latents = latents.to("cuda", dtype=model_dtype)

            batch_size = latents.shape[0]
            timesteps = torch.rand((batch_size,), device="cuda", dtype=model_dtype)
            noise = torch.randn_like(latents)

            t_broadcast = timesteps.view(batch_size, *([1] * (latents.ndim - 1)))
            x_t = (1.0 - t_broadcast) * noise + t_broadcast * latents
            target_velocity = latents - noise

        try:
            pred_velocity = nar_model(x_t, timestep=timesteps * 1000.0)
        except TypeError:
            try:
                pred_velocity = nar_model(x_t, timesteps * 1000.0)
            except TypeError:
                pred_velocity = nar_model(x_t, timestep=timesteps)

        loss = torch.mean((pred_velocity - target_velocity) ** 2)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(f"Эпоха [{epoch + 1}/{EPOCHS}] | Шаг [{step + 1}/{len(dataloader)}] | Flow Loss: {loss.item():.4f}")

    torch.cuda.empty_cache()
    gc.collect()

final_adapter_dir = OUTPUT_LORA_DIR / "style_custom_genre"
if final_adapter_dir.exists():
    shutil.rmtree(final_adapter_dir)
final_adapter_dir.mkdir(parents=True, exist_ok=True)

nar_model.save_pretrained(str(final_adapter_dir))

safetensors_files = list(final_adapter_dir.glob("*.safetensors"))
if safetensors_files:
    shutil.copyfile(safetensors_files[0], OUTPUT_LORA_DIR / "style_custom_genre.safetensors")

print(f"Стилевая LoRA успешно сохранена в: {final_adapter_dir}")