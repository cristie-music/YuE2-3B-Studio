#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Активация окружения
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "[ОШИБКА] Окружение venv не найдено. Сначала выполните: ./install_mac.sh"
    exit 1
fi

# Настройки среды под macOS
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HOME="$DIR/models_cache"
export YUE_ENABLE_FLASH_ATTN=0
export YUE_DISABLE_CUDA_GRAPH=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# ============================================================
# ВНИМАНИЕ: Укажите здесь свой персональный токен Hugging Face
# ============================================================
export HF_TOKEN="hf_ВАШ_ТОКЕН_ЗДЕСЬ"

echo "========================================================"
echo " Запуск YuE2-3B Web Studio на Apple Silicon (MPS / CPU)"
echo "========================================================"

# Автоматическое открытие браузера на Mac
sleep 2 && open "http://127.0.0.1:7860" &

python web_server.py