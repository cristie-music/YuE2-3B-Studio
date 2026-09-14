@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ========================================================
echo        YuE2-3B Studio: Установка окружения
echo ========================================================
echo.

if not exist "%~dp0python.zip" (
    echo [ОШИБКА] Файл python.zip не найден в корне папки!
    echo Пожалуйста, поместите архив python.zip в эту папку.
    pause
    exit /b 1
)

if not exist "%~dp0python\python.exe" (
    echo [1/3] Распаковка портативного Python из python.zip...
    tar -xf "%~dp0python.zip" -C "%~dp0"
    if not exist "%~dp0python\python.exe" (
        echo [ОШИБКА] Не удалось распаковать python.exe. Проверьте структуру архива.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Папка с Python уже распакована. Пропускаем...
)

echo [2/3] Обновление pip и базовых модулей...
"%~dp0python\python.exe" -m pip install --upgrade pip setuptools wheel

echo [3/3] Установка зависимостей из requirements.txt...
"%~dp0python\python.exe" -m pip install -r "%~dp0requirements.txt"

echo.
echo ========================================================
echo [ГОТОВО] Все зависимости успешно установлены!
echo Теперь откройте start_server.bat и укажите ваш HF_TOKEN.
echo ========================================================
pause