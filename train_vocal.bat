@echo off
chcp 65001 > nul
set PYTHONNOUSERSITE=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
echo ========================================================
echo   Запуск тренировки Вокальной LoRA 
echo ========================================================
"%~dp0python\python.exe" training\train_vocal_lora.py
pause