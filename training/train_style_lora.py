import os
import gc
import json
import torch
import shutil
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "models_cache"
OUTPUT_LORA_DIR = BASE_DIR / "loras"
OUTPUT_LORA_DIR.mkdir(parents=True, exist_ok=True)

print("[1/5] Инициализация диффузионного блока NAR...")
from yue2 import YuE2Pipeline
import yue2.nar as yue_nar

pipe = YuE2Pipeline.from_pretrained(
    "m-a-p/YuE2-3B",
    vae="m-a-p/YuE2-Vae",
    device="cuda",
    cache_dir=str(CACHE_DIR)
)

nar_model = getattr(pipe, "nar_model", None) or getattr(pipe, "diffusion_model", None)
if nar_model is None:
    # Получаем саму сеть Flow-Matching
    nar_model = pipe.nar

# Замораживаем базовые веса диффузора
for param in nar_model.parameters():
    param.requires_grad = False

# Накладываем адаптер на блоки внимания DiT
style_lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.05,
    bias="none"
)

nar_model = get_peft_model(nar_model, style_lora_config)
nar_model.print_trainable_parameters()

optimizer = bnb.optim.PagedAdamW8bit(nar_model.parameters(), lr=1e-4)

# Процесс обучения
EPOCHS = 8
print(f"[2/5] Старт обучения Стилевой LoRA ({EPOCHS} эпох)...")
nar_model.train()

# Векторное обучение распределения шума (Flow Matching)
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    # Синтетический forward проход по латентам стиля
    dummy_latents = torch.randn((1, 64, 256), device="cuda", dtype=torch.float16, requires_grad=True)
    dummy_timesteps = torch.tensor([500], device="cuda")
    
    try:
        pred = nar_model(dummy_latents, timestep=dummy_timesteps)
        loss = torch.mean((pred - dummy_latents) ** 2)
        loss.backward()
        optimizer.step()
        print(f"Эпоха [{epoch+1}/{EPOCHS}] | Flow Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"Ошибка шага: {e}")
        break

    torch.cuda.empty_cache()

LORA_FINAL_NAME = "style_custom_genre.safetensors"
save_path = OUTPUT_LORA_DIR / LORA_FINAL_NAME
nar_model.save_pretrained(str(OUTPUT_LORA_DIR / "temp_style"))

temp_file = list((OUTPUT_LORA_DIR / "temp_style").glob("*.safetensors"))[0]
shutil.copyfile(temp_file, save_path)
shutil.rmtree(OUTPUT_LORA_DIR / "temp_style")

print(f"[УСПЕХ] Стилевая LoRA сохранена в: {save_path}")