@echo off
chcp 65001 > nul
set PYTHONNOUSERSITE=1
set PATH=%~dp0python;%~dp0python\Scripts;%PATH%

echo ========================================================
echo   Установка и восстановление совместимых версий Transformers и PEFT
echo ========================================================

:: 1. Удаляем несовместимый TRL и конфликтующие версии
"%~dp0python\python.exe" -m pip uninstall -y trl transformers huggingface-hub tokenizers peft sympy

:: 2. Возвращаем строгие рабочие версии для YuE2
"%~dp0python\python.exe" -m pip install sympy==1.14.0 huggingface-hub==0.36.2 tokenizers==0.21.4 transformers==4.49.0

:: 3. Ставим PEFT версии 0.14.0, которая гарантированно работает с transformers 4.49.0 без их обновления
"%~dp0python\python.exe" -m pip install peft==0.14.0 --no-deps

:: 4. Проверяем наличие bitsandbytes (он уже установлен и работает)
"%~dp0python\python.exe" -c "import bitsandbytes, peft, transformers; print('--- ВСЕ МОДУЛИ УСПЕШНО ЗАГРУЖЕНЫ ---')"



%~dp0\python\python.exe -c "import bitsandbytes; print('BitsAndBytes OK')"
%~dp0\python\python.exe -c "from yue2 import YuE2Pipeline; import peft, bitsandbytes; print('>>> СИСТЕМА ПОЛНОСТЬЮ ГОТОВА К РАБОТЕ <<<')"
pause