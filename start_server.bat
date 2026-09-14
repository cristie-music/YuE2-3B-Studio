@echo off
set PYTHONNOUSERSITE=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HOME=%~dp0\models_cache
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set PATH=%~dp0;%~dp0\python\Scripts;;%~dp0\python;%PATH%
set HF_TOKEN=hf_ВАШ_РЕАЛЬНЫЙ_ТОКЕН_ЗДЕСЬ

echo ========================================================
echo Запуск YuE2-3B Studio Server на RTX 4060 Ti (16GB VRAM)
echo ========================================================

start http://127.0.0.1:7860
%~dp0\python\python.exe web_server.py

pause