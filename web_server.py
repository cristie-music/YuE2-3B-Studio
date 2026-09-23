import os
import gc
import json
import time
import queue
import shutil
import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# -------------------------------------------------------------
# 1. Настройка путей и окружения
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_CACHE_DIR = BASE_DIR / "models_cache"
MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRACKS_DIR = BASE_DIR / "outputs" / "web_tracks"
TRACKS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_BASE_DIR = BASE_DIR / "outputs" / "artifacts"
ARTIFACTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
LORAS_DIR = BASE_DIR / "loras"
LORAS_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = BASE_DIR / "history.json"
PROFILE_FILE = BASE_DIR / "profile.json"
PRESETS_FILE = BASE_DIR / "presets.json"
PERSONAS_FILE = BASE_DIR / "personas.json"

os.environ["HF_HOME"] = str(MODELS_CACHE_DIR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["YUE_ENABLE_FLASH_ATTN"] = "0"
os.environ["YUE_DISABLE_CUDA_GRAPH"] = "1"

import torch
import soundfile as sf

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

from yue2 import YuE2Pipeline
import yue2.nar as yue_nar
import yue2.sampling as yue_sampling

try:
    from music21 import converter as m21_converter
except ImportError:
    m21_converter = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

# -------------------------------------------------------------
# 2. Оптимизация памяти NAR под 16GB VRAM и безопасный FAST-патчинг
# -------------------------------------------------------------
ORIG_CACHED_NAR_INIT = yue_nar.CachedNAR.__init__

def patched_cached_nar_init(self, model, chunk, attention, query_chunk_size=None, *args, **kwargs):
    return ORIG_CACHED_NAR_INIT(self, model, chunk, attention, query_chunk_size=512, *args, **kwargs)

yue_nar.CachedNAR.__init__ = patched_cached_nar_init

CURRENT_TASK_IS_FAST = True

ORIGINAL_NAR_SYNTHESIZE = yue_nar.synthesize
def safe_fast_synthesize(*args, **kwargs):
    if CURRENT_TASK_IS_FAST:
        if "steps" in kwargs:
            kwargs["steps"] = 16
        elif len(args) >= 6:
            args_list = list(args)
            args_list[5] = 16
            args = tuple(args_list)
    return ORIGINAL_NAR_SYNTHESIZE(*args, **kwargs)

yue_nar.synthesize = safe_fast_synthesize

ORIGINAL_SAMPLING_GENERATE = yue_sampling.generate_tokens
def safe_fast_generate_tokens(model, prefix, sampling, seed, phase, *args, **kwargs):
    if CURRENT_TASK_IS_FAST and phase == "song":
        if hasattr(sampling, "max_tokens"):
            sampling.max_tokens = min(sampling.max_tokens, 3200)
        elif hasattr(sampling, "max_length"):
            sampling.max_length = min(sampling.max_length, 3200)
    return ORIGINAL_SAMPLING_GENERATE(model, prefix, sampling, seed, phase, *args, **kwargs)

yue_sampling.generate_tokens = safe_fast_generate_tokens

task_queue = queue.Queue()
current_task = {
    "status": "idle",
    "progress_msg": "",
    "task_id": None,
    "error": None
}

# -------------------------------------------------------------
# 3. Синглтон пайплайна YuE2 и двойной менеджер LoRA (AR + NAR)
# -------------------------------------------------------------
GLOBAL_PIPE = None

def get_pipeline():
    global GLOBAL_PIPE
    if GLOBAL_PIPE is None:
        current_task["progress_msg"] = f"Загрузка весов YuE2-3B в память ({device.upper()})..."
        GLOBAL_PIPE = YuE2Pipeline.from_pretrained(
            "m-a-p/YuE2-3B",
            vae="m-a-p/YuE2-Vae",
            device=device,
            backend="torch-eager",
            cache_dir=str(MODELS_CACHE_DIR)
        )
        orig_synth = GLOBAL_PIPE.synthesize
        def safe_synth(*args, **kwargs):
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
            gc.collect()
            return orig_synth(*args, **kwargs)
        GLOBAL_PIPE.synthesize = safe_synth
    return GLOBAL_PIPE



LORA_STATE = {
    "ar_attr": None,       # имя атрибута в pipe, напр. "model"
    "ar_original": None,   # исходный (базовый) модуль
    "nar_attr": None,
    "nar_original": None,
}

# Возможные имена атрибутов AR/NAR в разных сборках yue2_infer
_AR_CANDIDATES  = ("model", "ar_model", "ar", "language_model", "ar_lm", "lm", "text_model")
_NAR_CANDIDATES = ("nar", "nar_model", "nar_diffusion", "diffusion", "diffusion_model", "vae_model")


def _find_pipe_module(pipe, candidates):
    """
    Возвращает (attr_name, module) для первого существующего и не-None
    атрибута из candidates, у которого есть .parameters() (т.е. это nn.Module).
    """
    for name in candidates:
        if not hasattr(pipe, name):
            continue
        val = getattr(pipe, name)
        if val is None:
            continue
        if hasattr(val, "parameters"):
            return name, val
    return None, None


def _scale_lora_adapter(peft_model, adapter_name, user_scale):
    """
    Применяет пользовательский масштаб к уже активированному адаптеру.
    Умножает существующий scaling[adapter_name] (обычно lora_alpha / r) на user_scale.
    """
    try:
        user_scale = float(user_scale)
    except (TypeError, ValueError):
        return

    touched = 0
    for module in peft_model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling:
            scaling[adapter_name] = float(scaling[adapter_name]) * user_scale
            touched += 1
    if touched == 0:
        print(f"[LoRA] Не найдено LoRA-слоёв для адаптера '{adapter_name}' — scale не применён.")


def _attach_lora(module, lora_path, adapter_name, scale):
    """
    Прикрепляет LoRA-адаптер к module и возвращает новый PeftModel.
    Корректно обрабатывает случай, когда module уже является PeftModel.
    """
    if PeftModel is None:
        raise RuntimeError("peft не установлен. Установите: pip install peft")

    if isinstance(module, PeftModel):
        # Модель уже PEFT — просто добавляем/переключаем адаптер
        existing = getattr(module, "peft_config", {}) or {}
        if adapter_name in existing:
            module.set_adapter(adapter_name)
        else:
            module.load_adapter(str(lora_path), adapter_name=adapter_name)
            module.set_adapter(adapter_name)
        peft_model = module
    else:
        peft_model = PeftModel.from_pretrained(
            module, str(lora_path), adapter_name=adapter_name
        )
        peft_model.set_adapter(adapter_name)

    _scale_lora_adapter(peft_model, adapter_name, scale)
    return peft_model


def apply_dual_loras(pipe, vocal_lora, vocal_scale, style_lora, style_scale):
    """Подключает вокальную LoRA к AR и стилевую LoRA к NAR."""
    loaded = {"vocal": False, "style": False}

    # --- Вокальная LoRA → AR ---
    if vocal_lora:
        v_path = LORAS_DIR / vocal_lora
        if not v_path.exists():
            print(f"[LoRA] Файл не найден: {v_path}")
        else:
            ar_attr, ar_base = _find_pipe_module(pipe, _AR_CANDIDATES)
            if ar_base is None:
                print(f"[LoRA] AR-модуль не найден в pipe (проверены: {_AR_CANDIDATES})")
            else:
                try:
                    peft_ar = _attach_lora(ar_base, v_path, "vocal_adapter", vocal_scale)
                    setattr(pipe, ar_attr, peft_ar)   # ← ГЛАВНЫЙ ФИКС
                    LORA_STATE["ar_attr"] = ar_attr
                    LORA_STATE["ar_original"] = ar_base
                    loaded["vocal"] = True
                    print(f"[LoRA] Вокальная LoRA '{vocal_lora}' → pipe.{ar_attr} (scale={vocal_scale})")
                except Exception as e:
                    print(f"[LoRA Error] Вокальная LoRA: {e}")

    # --- Стилевая LoRA → NAR ---
    if style_lora:
        s_path = LORAS_DIR / style_lora
        if not s_path.exists():
            print(f"[LoRA] Файл не найден: {s_path}")
        else:
            nar_attr, nar_base = _find_pipe_module(pipe, _NAR_CANDIDATES)
            if nar_base is None:
                print(f"[LoRA] NAR-модуль не найден в pipe (проверены: {_NAR_CANDIDATES})")
            else:
                try:
                    peft_nar = _attach_lora(nar_base, s_path, "style_adapter", style_scale)
                    setattr(pipe, nar_attr, peft_nar)  # ← ГЛАВНЫЙ ФИКС
                    LORA_STATE["nar_attr"] = nar_attr
                    LORA_STATE["nar_original"] = nar_base
                    loaded["style"] = True
                    print(f"[LoRA] Стилевая LoRA '{style_lora}' → pipe.{nar_attr} (scale={style_scale})")
                except Exception as e:
                    print(f"[LoRA Error] Стилевая LoRA: {e}")

    return loaded


def remove_dual_loras(pipe):
    """Возвращает базовые модули на место, откатывая LoRA."""
    for target in ("ar", "nar"):
        attr     = LORA_STATE.get(f"{target}_attr")
        original = LORA_STATE.get(f"{target}_original")
        if not attr or original is None:
            continue
        try:
            current = getattr(pipe, attr, None)
            if isinstance(current, PeftModel):
                # Пытаемся корректно отгрузить адаптеры (не критично, если не сработает)
                try:
                    if hasattr(current, "unload"):
                        current.unload()
                except Exception:
                    pass
            # Возвращаем базовый модуль
            setattr(pipe, attr, original)
            print(f"[LoRA] Восстановлен pipe.{attr} (базовый модуль)")
        except Exception as e:
            print(f"[LoRA Error] Откат {target}: {e}")
        finally:
            LORA_STATE[f"{target}_attr"] = None
            LORA_STATE[f"{target}_original"] = None


# -------------------------------------------------------------
# 4. База данных JSON и файлы LoRA
# -------------------------------------------------------------
DEFAULT_PRESETS = [
    {
        "id": "p_industrial_metal",
        "name": "Industrial Alternative Metal",
        "style": "industrial alternative metal, EDM, female vocal, progressive metal, aggressive distorted guitars, driving live acoustic drums, 135 bpm"
    },
    {
        "id": "p_dark_triphop",
        "name": "Dark Trip-Hop Instrumental",
        "style": "instrumental, dark trip-hop, industrial electronic, heavy sub bass, crisp breakbeats, atmospheric pads, cinematic, 90 bpm"
    },
    {
        "id": "p_synthpop",
        "name": "Melancholic Synth-Pop",
        "style": "melancholic electronic synth-pop, deep rolling synth bass, tight drums, warm female vocal, atmospheric reverb, melodic, 120 bpm"
    }
]

def load_presets():
    if PRESETS_FILE.exists():
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    save_presets(DEFAULT_PRESETS)
    return DEFAULT_PRESETS

def save_presets(presets):
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)

def load_personas():
    if PERSONAS_FILE.exists():
        try:
            with open(PERSONAS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_personas(personas):
    with open(PERSONAS_FILE, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)

def load_profile():
    if PROFILE_FILE.exists():
        try:
            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"name": "cristie", "avatar_letter": "C"}

def save_profile(data):
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def list_available_loras():
    loras = []
    for file in LORAS_DIR.glob("*.*"):
        if file.suffix.lower() in [".safetensors", ".bin", ".pt"]:
            size_mb = round(file.stat().st_size / (1024 * 1024), 1)
            loras.append({
                "filename": file.name,
                "name": file.stem.replace("_", " ").title(),
                "size": f"{size_mb} MB"
            })
    return loras

def sanitize_abc_notation(raw_abc: str, title: str = "YuE2 Track") -> str:
    clean_lines = []
    has_header = False
    for line in raw_abc.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("X:"):
            has_header = True
            clean_lines.append("X:1")
        elif line.startswith("T:"):
            clean_lines.append(f"T:{title}")
        elif line.startswith(("M:", "L:", "Q:", "K:", "V:")):
            clean_lines.append(line)
        elif not line.startswith("%") and not line.startswith("w:"):
            clean_lines.append(line)

    if not has_header:
        header = ["X:1", f"T:{title}", "M:4/4", "L:1/8", "K:C", "V:1"]
        return "\n".join(header + clean_lines)
    return "\n".join(clean_lines)

def generate_fallback_abc(title, style, lyrics):
    bpm = 120
    for token in style.split(","):
        if "bpm" in token.lower():
            try:
                bpm = int("".join(filter(str.isdigit, token)))
            except Exception:
                pass

    abc_lines = [
        "X:1",
        f"T:{title or 'YuE2 Composition'}",
        "M:4/4",
        "L:1/8",
        f"Q:1/4={bpm}",
        "K:C",
        "V:1 name=\"Lead\"",
        "|: [CEG]4 [DFA]4 | [EGB]4 [CEG]4 :|"
    ]
    return "\n".join(abc_lines)

# -------------------------------------------------------------
# 5. Фоновый воркер инференса
# -------------------------------------------------------------
def generation_worker():
    global current_task, CURRENT_TASK_IS_FAST
    while True:
        task = task_queue.get()
        if task is None:
            break

        task_id = task["id"]
        current_task["status"] = "running"
        current_task["task_id"] = task_id
        current_task["progress_msg"] = "Подготовка параметров генерации..."
        current_task["error"] = None

        CURRENT_TASK_IS_FAST = task.get("fast_mode", True)
        start_time = time.time()
        pipe = None
        loras_status = {"vocal": False, "style": False}

        try:
            pipe = get_pipeline()

            vocal_lora = task.get("vocal_lora")
            vocal_scale = float(task.get("vocal_scale", 0.8))
            style_lora = task.get("style_lora")
            style_scale = float(task.get("style_scale", 0.8))

            if vocal_lora or style_lora:
                current_task["progress_msg"] = "Применение адаптеров LoRA (Вокал/Стиль)..."
                loras_status = apply_dual_loras(pipe, vocal_lora, vocal_scale, style_lora, style_scale)

            style = task["style"]
            if task.get("is_instrumental", False):
                if "instrumental" not in style.lower():
                    style = f"instrumental, {style}"
                lyrics = "[intro]\n[inst]\n\n[verse]\n[inst]\n\n[chorus]\n[inst]\n\n[outro]\n[inst]"
            else:
                lyrics = task.get("lyrics", "").strip()
                if not lyrics:
                    lyrics = "[verse]\nInstrumental melody\n[chorus]\nAtmospheric sound"

            audio_file = task.get("audio_file")
            custom_abc = task.get("abc_score", "").strip()

            gen_kwargs = {
                "style": style,
                "lyrics": lyrics,
                "cfg_scale": float(task.get("cfg_scale", 1.2)),
                "seed": int(task.get("seed", 42))
            }

            if custom_abc:
                current_task["progress_msg"] = "Синтез по партитуре (MIDI/ABC)..."
                gen_kwargs["abc"] = sanitize_abc_notation(custom_abc, task.get("title") or "Track")
                gen_kwargs["cot"] = "melody"
            else:
                chosen_cot = "melody" if audio_file else task.get("cot", "full")
                current_task["progress_msg"] = f"Символическое планирование (cot='{chosen_cot}')..."
                gen_kwargs["cot"] = chosen_cot

            song = pipe(**gen_kwargs)

            current_task["progress_msg"] = "Сохранение аудио и нотных артефактов..."
            filename = f"track_{task_id}.flac"
            file_path = TRACKS_DIR / filename

            if hasattr(song, "save"):
                song.save(str(file_path))
            else:
                audio_data = song.audio if hasattr(song, "audio") else song
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.detach().cpu().float().numpy()
                if audio_data.ndim == 2 and audio_data.shape[0] < audio_data.shape[1]:
                    audio_data = audio_data.T
                sf.write(str(file_path), audio_data, 48000)

            track_artifacts_dir = ARTIFACTS_BASE_DIR / task_id
            track_artifacts_dir.mkdir(parents=True, exist_ok=True)
            target_score_file = track_artifacts_dir / "score.abc"

            saved_abc_text = None
            if custom_abc:
                saved_abc_text = custom_abc
            elif hasattr(song, "abc") and song.abc:
                saved_abc_text = str(song.abc)
            elif hasattr(song, "score") and song.score:
                saved_abc_text = str(song.score)

            if hasattr(song, "save_artifacts"):
                try:
                    song.save_artifacts(str(track_artifacts_dir))
                except Exception:
                    pass

            if not target_score_file.exists():
                found_abcs = list(track_artifacts_dir.rglob("*.abc"))
                if found_abcs:
                    shutil.copyfile(found_abcs[0], target_score_file)
                elif saved_abc_text:
                    target_score_file.write_text(saved_abc_text, encoding="utf-8")
                else:
                    fallback_text = generate_fallback_abc(task.get("title"), style, lyrics)
                    target_score_file.write_text(fallback_text, encoding="utf-8")

            duration_sec = round(time.time() - start_time)

            history = load_history()
            history.insert(0, {
                "id": task_id,
                "title": task.get("title") or f"YuE2 Track #{task_id[:6]}",
                "style": style,
                "lyrics": lyrics,
                "cot": gen_kwargs["cot"],
                "cfg_scale": task.get("cfg_scale", 1.2),
                "seed": task.get("seed", 42),
                "fast_mode": CURRENT_TASK_IS_FAST,
                "is_instrumental": task.get("is_instrumental", False),
                "reference_audio": audio_file,
                "vocal_lora": vocal_lora if loras_status["vocal"] else None,
                "vocal_scale": vocal_scale if loras_status["vocal"] else None,
                "style_lora": style_lora if loras_status["style"] else None,
                "style_scale": style_scale if loras_status["style"] else None,
                "has_abc": True,
                "is_midi_gen": bool(task.get("midi_source", False)),
                "filename": filename,
                "url": f"/audio/{filename}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_render": f"{duration_sec}s",
                "rating": 0
            })
            save_history(history)

            current_task["status"] = "completed"
            current_task["progress_msg"] = f"Готово за {duration_sec} сек!"

        except Exception as e:
            current_task["status"] = "error"
            current_task["error"] = str(e)
            current_task["progress_msg"] = f"Ошибка: {str(e)}"
        finally:
            if pipe and (loras_status["vocal"] or loras_status["style"]):
                remove_dual_loras(pipe)
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
            gc.collect()
            task_queue.task_done()

threading.Thread(target=generation_worker, daemon=True).start()

# -------------------------------------------------------------
# 6. HTTP API сервер
# -------------------------------------------------------------
class StudioHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path in ["/", "/index.html"]:
            html_path = BASE_DIR / "index.html"
            if html_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            self.send_error(404, "index.html not found")
            return

        if parsed.path.startswith("/audio/"):
            filename = parsed.path.replace("/audio/", "")
            file_path = TRACKS_DIR / filename
            if not file_path.exists():
                self.send_error(404, "Audio not found")
                return

            file_size = file_path.stat().st_size
            range_header = self.headers.get("Range")

            if range_header:
                byte_range = range_header.strip().split("=")[-1]
                start_str, end_str = byte_range.split("-")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                length = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", "audio/flac")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                with open(file_path, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
                return
            else:
                self.send_response(200)
                self.send_header("Content-Type", "audio/flac")
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        if parsed.path == "/api/score":
            track_id = qs.get("id", [""])[0]
            track_artifacts_dir = ARTIFACTS_BASE_DIR / track_id
            score_file = track_artifacts_dir / "score.abc"

            if not score_file.exists():
                found = list(track_artifacts_dir.rglob("*.abc")) if track_artifacts_dir.exists() else []
                if found:
                    score_file = found[0]

            if score_file.exists():
                abc_content = score_file.read_text(encoding="utf-8", errors="ignore")
            else:
                history = load_history()
                item = next((x for x in history if x["id"] == track_id), None)
                abc_content = generate_fallback_abc(
                    item.get("title") if item else "Track",
                    item.get("style") if item else "pop",
                    item.get("lyrics") if item else ""
                )
                track_artifacts_dir.mkdir(parents=True, exist_ok=True)
                (track_artifacts_dir / "score.abc").write_text(abc_content, encoding="utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "abc": abc_content}).encode("utf-8"))
            return

        if parsed.path == "/api/midi":
            track_id = qs.get("id", [""])[0]
            track_artifacts_dir = ARTIFACTS_BASE_DIR / track_id
            track_artifacts_dir.mkdir(parents=True, exist_ok=True)
            score_file = track_artifacts_dir / "score.abc"
            midi_file = track_artifacts_dir / "score.mid"

            if not score_file.exists():
                found = list(track_artifacts_dir.rglob("*.abc"))
                if found:
                    shutil.copyfile(found[0], score_file)
                else:
                    history = load_history()
                    item = next((x for x in history if x["id"] == track_id), None)
                    fallback_text = generate_fallback_abc(
                        item.get("title") if item else "Track",
                        item.get("style") if item else "pop",
                        item.get("lyrics") if item else ""
                    )
                    score_file.write_text(fallback_text, encoding="utf-8")

            if not midi_file.exists() or midi_file.stat().st_size == 0:
                if m21_converter is not None:
                    try:
                        abc_txt = score_file.read_text(encoding="utf-8", errors="ignore")
                        parsed_score = m21_converter.parse(abc_txt, format="abc")
                        parsed_score.write("midi", fp=str(midi_file))
                    except Exception as err:
                        print(f"[Предупреждение] music21 не смог разобрать ABC: {err}")
                        try:
                            clean_fallback = generate_fallback_abc("Track", "120 bpm", "")
                            parsed_score = m21_converter.parse(clean_fallback, format="abc")
                            parsed_score.write("midi", fp=str(midi_file))
                        except Exception as e2:
                            self.send_error(500, f"MIDI Generation error: {e2}")
                            return
                else:
                    self.send_error(500, "Библиотека music21 не установлена.")
                    return

            midi_bytes = midi_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/midi")
            self.send_header("Content-Disposition", f'attachment; filename="track_{track_id}.mid"')
            self.send_header("Content-Length", str(len(midi_bytes)))
            self.end_headers()
            self.wfile.write(midi_bytes)
            return

        if parsed.path == "/api/loras":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(list_available_loras()).encode("utf-8"))
            return

        # [LORA FIX] Диагностический эндпоинт — помогает понять, какие атрибуты есть в pipe
        if parsed.path == "/api/loras/debug":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            info = {"pipe_loaded": GLOBAL_PIPE is not None, "attributes": {}, "lora_state": {}}
            if GLOBAL_PIPE is not None:
                for name in dir(GLOBAL_PIPE):
                    if name.startswith("_"):
                        continue
                    try:
                        val = getattr(GLOBAL_PIPE, name)
                        if hasattr(val, "parameters"):
                            info["attributes"][name] = type(val).__name__
                    except Exception:
                        pass
            info["lora_state"] = {
                k: (type(v).__name__ if v is not None else None)
                for k, v in LORA_STATE.items()
            }
            self.wfile.write(json.dumps(info, ensure_ascii=False).encode("utf-8"))
            return

        if parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({
                "queue_size": task_queue.qsize(),
                "current_task": current_task
            }).encode("utf-8"))
            return

        if parsed.path == "/api/history":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(load_history()).encode("utf-8"))
            return

        if parsed.path == "/api/presets":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(load_presets()).encode("utf-8"))
            return

        if parsed.path == "/api/personas":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(load_personas()).encode("utf-8"))
            return

        if parsed.path == "/api/profile":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(load_profile()).encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))

        if parsed.path == "/api/upload":
            content_type = self.headers.get("Content-Type", "")
            raw_data = self.rfile.read(content_length)

            if "boundary=" in content_type:
                boundary = content_type.split("boundary=")[1].encode("utf-8")
                parts = raw_data.split(boundary)
                saved_filename = None
                is_midi = False
                for part in parts:
                    if b'filename="' in part:
                        header_part, body_part = part.split(b"\r\n\r\n", 1)
                        header_str = header_part.decode("utf-8", errors="ignore")
                        fn_start = header_str.find('filename="') + 10
                        fn_end = header_str.find('"', fn_start)
                        orig_fn = header_str[fn_start:fn_end]
                        clean_fn = Path(orig_fn).name
                        ext = clean_fn.lower()
                        if ext.endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mid", ".midi")):
                            saved_filename = f"{int(time.time())}_{clean_fn}"
                            is_midi = ext.endswith((".mid", ".midi"))
                            body_part = body_part.rstrip(b"\r\n--")
                            saved_path = UPLOADS_DIR / saved_filename
                            with open(saved_path, "wb") as f:
                                f.write(body_part)
                            break
            else:
                saved_filename = f"file_{int(time.time())}.bin"
                saved_path = UPLOADS_DIR / saved_filename
                with open(saved_path, "wb") as f:
                    f.write(raw_data)
                is_midi = False

            abc_content = ""
            if is_midi and m21_converter is not None:
                try:
                    midi_score = m21_converter.parse(str(saved_path), format="midi")
                    tmp_abc_path = UPLOADS_DIR / f"{saved_filename}.abc"
                    midi_score.write("abc", fp=str(tmp_abc_path))
                    if tmp_abc_path.exists():
                        raw_abc = tmp_abc_path.read_text(encoding="utf-8", errors="ignore")
                        abc_content = sanitize_abc_notation(raw_abc, saved_filename)
                        tmp_abc_path.unlink()
                except Exception as e:
                    print(f"[Ошибка конвертации MIDI в ABC]: {e}")
                    abc_content = ""

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": True,
                "filename": saved_filename,
                "is_midi": is_midi,
                "abc": abc_content
            }).encode("utf-8"))
            return

        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode("utf-8"))
        except Exception:
            data = {}

        if parsed.path == "/api/tracks/delete":
            track_id = data.get("id")
            history = load_history()
            track_item = next((t for t in history if t["id"] == track_id), None)
            if track_item:
                filename = track_item.get("filename")
                if filename:
                    flac_path = TRACKS_DIR / filename
                    if flac_path.exists():
                        try:
                            flac_path.unlink()
                        except Exception:
                            pass
                art_dir = ARTIFACTS_BASE_DIR / track_id
                if art_dir.exists():
                    try:
                        shutil.rmtree(art_dir)
                    except Exception:
                        pass
                history = [t for t in history if t["id"] != track_id]
                save_history(history)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
            return

        if parsed.path == "/api/loras/delete":
            lora_fn = data.get("filename")
            target_path = LORAS_DIR / lora_fn
            if target_path.exists():
                try:
                    target_path.unlink()
                except Exception:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "loras": list_available_loras()}).encode("utf-8"))
            return

        if parsed.path == "/api/personas":
            persona_id = data.get("id") or f"pers_{int(time.time() * 1000)}"
            name = data.get("name", "New Voice Persona").strip()
            style = data.get("style", "").strip()
            seed = int(data.get("seed", 42))
            cot = data.get("cot", "full")

            personas = load_personas()
            existing = False
            for p in personas:
                if p["id"] == persona_id:
                    p["name"] = name
                    p["style"] = style
                    p["seed"] = seed
                    p["cot"] = cot
                    existing = True
                    break

            if not existing:
                personas.append({
                    "id": persona_id,
                    "name": name,
                    "style": style,
                    "seed": seed,
                    "cot": cot
                })

            save_personas(personas)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "personas": personas}).encode("utf-8"))
            return

        if parsed.path == "/api/personas/delete":
            persona_id = data.get("id")
            personas = [p for p in load_personas() if p["id"] != persona_id]
            save_personas(personas)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "personas": personas}).encode("utf-8"))
            return

        if parsed.path == "/api/presets":
            preset_id = data.get("id") or f"p_{int(time.time() * 1000)}"
            name = data.get("name", "Новый пресет").strip()
            style = data.get("style", "").strip()

            presets = load_presets()
            existing = False
            for p in presets:
                if p["id"] == preset_id:
                    p["name"] = name
                    p["style"] = style
                    existing = True
                    break

            if not existing:
                presets.append({"id": preset_id, "name": name, "style": style})

            save_presets(presets)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "presets": presets}).encode("utf-8"))
            return

        if parsed.path == "/api/presets/delete":
            preset_id = data.get("id")
            presets = [p for p in load_presets() if p["id"] != preset_id]
            save_presets(presets)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "presets": presets}).encode("utf-8"))
            return

        if parsed.path == "/api/rate":
            track_id = data.get("id")
            rating = int(data.get("rating", 0))
            history = load_history()
            for item in history:
                if item["id"] == track_id:
                    item["rating"] = rating if item.get("rating") != rating else 0
                    break
            save_history(history)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
            return

        if parsed.path == "/api/generate":
            task_id = str(int(time.time() * 1000))
            task_item = {
                "id": task_id,
                "title": data.get("title", "").strip(),
                "style": data.get("style", "").strip(),
                "lyrics": data.get("lyrics", "").strip(),
                "cot": data.get("cot", "melody"),
                "cfg_scale": float(data.get("cfg_scale", 1.2)),
                "seed": int(data.get("seed", 42)),
                "fast_mode": bool(data.get("fast_mode", True)),
                "is_instrumental": bool(data.get("is_instrumental", False)),
                "audio_file": data.get("audio_file", None),
                "abc_score": data.get("abc_score", ""),
                "midi_source": bool(data.get("midi_source", False)),
                "vocal_lora": data.get("vocal_lora", None),
                "vocal_scale": float(data.get("vocal_scale", 0.8)),
                "style_lora": data.get("style_lora", None),
                "style_scale": float(data.get("style_scale", 0.8))
            }
            task_queue.put(task_item)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "task_id": task_id}).encode("utf-8"))
            return

        if parsed.path == "/api/profile":
            new_name = data.get("name", "cristie").strip()
            avatar = new_name[0].upper() if new_name else "C"
            updated = {"name": new_name, "avatar_letter": avatar}
            save_profile(updated)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "profile": updated}).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")

def run_server(port=7860):
    server = HTTPServer(("127.0.0.1", port), StudioHandler)
    print("=" * 65)
    print(f" YuE2-3B Studio Server запущен на бэкенде: {device.upper()}")
    print(f" Каталог адаптеров LoRA: {LORAS_DIR}")
    print(f" Доступ в браузере: http://127.0.0.1:{port}")
    print("=" * 65)
    server.serve_forever()

if __name__ == "__main__":
    run_server()