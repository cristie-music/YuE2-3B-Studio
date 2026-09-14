#!/usr/bin/env bash
set -e

echo "========================================================"
echo "    YuE2-3B Studio: Установка окружения для macOS (M1/M2/M3)"
echo "========================================================"

# 1. Проверка наличия Homebrew
if ! command -v brew &> /dev/null; then
    echo "[!] Homebrew не найден. Устанавливаем Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 2. Проверка системных библиотек (ffmpeg, libsndfile)
echo "[1/4] Проверка системных кодеков..."
brew install ffmpeg libsndfile python@3.11 2>/dev/null || true

# 3. Создание виртуального окружения Python
echo "[2/4] Создание виртуального окружения (venv)..."
if [ ! -d "venv" ]; then
    python3.11 -m venv venv || python3 -m venv venv
fi

source venv/bin/activate

# 4. Установка пакетов
echo "[3/4] Обновление pip..."
pip install --upgrade pip setuptools wheel

echo "[4/4] Установка зависимостей из requirements_mac.txt..."
pip install -r requirements_mac.txt

echo ""
echo "========================================================"
echo "[ГОТОВО] Окружение на Mac успешно настроено!"
echo "Запускайте студию командой: ./start_server_mac.sh"
echo "========================================================"