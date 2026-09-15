import os
import gc
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "models_cache"
OUTPUT_LORA_DIR = BASE_DIR / "loras"
OUTPUT_LORA_DIR.mkdir(parents=True, exist_ok=True)

# 1. 4-битная конфигурация QLoRA для экономии VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("[1/5] Загрузка базовой модели YuE2-3B в 4-битном режиме...")
from yue2 import YuE2Pipeline

# Загружаем пайплайн
pipe = YuE2Pipeline.from_pretrained(
    "m-a-p/YuE2-3B",
    vae="m-a-p/YuE2-Vae",
    device_map="auto",
    quantization_config=bnb_config,
    cache_dir=str(CACHE_DIR)
)

# Выбираем языковой AR-модуль
ar_model = getattr(pipe, "model", None) or getattr(pipe, "ar_model", None)
if ar_model is None:
    raise RuntimeError("Не удалось извлечь AR модель из пайплайна")

ar_model = prepare_model_for_kbit_training(ar_model)
ar_model.gradient_checkpointing_enable()

# 2. Конфигурация LoRA для вокала
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

ar_model = get_peft_model(ar_model, lora_config)
ar_model.print_trainable_parameters()

# 3. Датасет
class VocalDataset(Dataset):
    def __init__(self, data_dir):
        self.files = list(Path(data_dir).glob("*.txt"))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        txt_path = self.files[idx]
        text = txt_path.read_text(encoding="utf-8")
        return text

dataset = VocalDataset(BASE_DIR / "training" / "dataset_vocal")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 4. Оптимизатор с 8-битным состоянием
optimizer = bnb.optim.PagedAdamW8bit(ar_model.parameters(), lr=2e-4)

# 5. Цикл обучения
EPOCHS = 10
print(f"[2/5] Старт обучения Вокальной LoRA ({EPOCHS} эпох)...")

ar_model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for step, batch_text in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Токенизация текста с вокальной разметкой
        inputs = pipe.tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = inputs.input_ids.to("cuda")
        
        outputs = ar_model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Эпоха [{epoch+1}/{EPOCHS}] | Шаг {step+1} | Loss: {loss.item():.4f}")
        
        torch.cuda.empty_cache()

# 6. Сохранение адаптера в папку loras/
LORA_FINAL_NAME = "vocal_custom_voice.safetensors"
save_path = OUTPUT_LORA_DIR / LORA_FINAL_NAME
ar_model.save_pretrained(str(OUTPUT_LORA_DIR / "temp_vocal"))

# Конвертация в единый safetensors для веб-интерфейса
temp_file = list((OUTPUT_LORA_DIR / "temp_vocal").glob("*.safetensors"))[0]
shutil.copyfile(temp_file, save_path)
shutil.rmtree(OUTPUT_LORA_DIR / "temp_vocal")

print(f"[УСПЕХ] LoRA успешно сохранена в: {save_path}")