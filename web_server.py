import os
import gc
import sys
import json
import time
import queue
import shutil
import inspect
import threading
import traceback
import dataclasses
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

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

def safe_resolve(base_dir: Path, subpath: str) -> Path:
    cleaned = Path(subpath).name
    target = (base_dir / cleaned).resolve()
    if not target.is_relative_to(base_dir.resolve()):
        raise PermissionError("Access denied")
    return target

def safe_track_id(tid: str) -> str:
    return "".join(c for c in str(tid) if c.isalnum() or c in ("_", "-"))

CURRENT_TASK_IS_FAST = True

# ---------------------------------------------------------------------
# Прокси-обертка для NAR модели для гарантированной защиты от нехватки атрибутов
# ---------------------------------------------------------------------
class SafeNARModelWrapper:
    """
    Гарантирует, что обращение к wrapper.model вернет объект,
    содержащий и embed_tokens, и rotary_emb, и layers.
    """
    def __init__(self, raw_model):
        self._raw_model = raw_model
        real_backbone = None
        
        # 1. Проверяем raw_model.model
        if hasattr(raw_model, "model"):
            cand = raw_model.model
            if hasattr(cand, "embed_tokens"):
                real_backbone = cand
            elif hasattr(cand, "model") and hasattr(cand.model, "embed_tokens"):
                real_backbone = cand.model

        # 2. Если не найден, проверяем raw_model напрямую
        if real_backbone is None:
            if hasattr(raw_model, "embed_tokens"):
                real_backbone = raw_model
            elif hasattr(raw_model, "base_model") and hasattr(raw_model.base_model, "embed_tokens"):
                real_backbone = raw_model.base_model

        # 3. Дефолтный фоллбэк
        if real_backbone is None:
            real_backbone = getattr(raw_model, "model", raw_model)

        # Проверяем наличие rotary_emb у найденного backbone
        if not hasattr(real_backbone, "rotary_emb"):
            rot_cand = None
            if hasattr(raw_model, "rotary_emb"):
                rot_cand = raw_model.rotary_emb
            elif hasattr(raw_model, "model") and hasattr(raw_model.model, "rotary_emb"):
                rot_cand = raw_model.model.rotary_emb
            elif hasattr(real_backbone, "layers") and len(real_backbone.layers) > 0:
                first_layer = real_backbone.layers[0]
                if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "rotary_emb"):
                    rot_cand = first_layer.self_attn.rotary_emb

            if rot_cand is not None:
                try:
                    setattr(real_backbone, "rotary_emb", rot_cand)
                except Exception:
                    pass

        # Проверяем наличие embed_tokens
        if not hasattr(real_backbone, "embed_tokens"):
            emb_cand = None
            if hasattr(raw_model, "get_input_embeddings"):
                emb_cand = raw_model.get_input_embeddings()
            elif hasattr(raw_model, "embed_tokens"):
                emb_cand = raw_model.embed_tokens

            if emb_cand is not None:
                try:
                    setattr(real_backbone, "embed_tokens", emb_cand)
                except Exception:
                    pass

        self.model = real_backbone

    def __getattr__(self, name):
        return getattr(self._raw_model, name)

    def __call__(self, *args, **kwargs):
        return self._raw_model(*args, **kwargs)

# ---------------------------------------------------------------------
# Патч CachedNAR: чанкирование внимания (512 токенов) + применение SafeNARModelWrapper
# ---------------------------------------------------------------------
ORIG_CACHED_NAR_INIT = yue_nar.CachedNAR.__init__

def patched_cached_nar_init(self, *args, **kwargs):
    call_args = list(args)
    call_kwargs = dict(kwargs)

    if len(call_args) > 0:
        call_args[0] = SafeNARModelWrapper(call_args[0])
    elif "model" in call_kwargs:
        call_kwargs["model"] = SafeNARModelWrapper(call_kwargs["model"])

    try:
        sig = inspect.signature(ORIG_CACHED_NAR_INIT)
        param_names = list(sig.parameters.keys())
        if "query_chunk_size" in param_names:
            pos = param_names.index("query_chunk_size")
            arg_idx = pos - 1 if (param_names and param_names[0] == "self") else pos
            if 0 <= arg_idx < len(call_args):
                call_args[arg_idx] = 512
            else:
                call_kwargs["query_chunk_size"] = 512
        else:
            call_kwargs["query_chunk_size"] = 512
    except Exception:
        call_kwargs["query_chunk_size"] = 512

    return ORIG_CACHED_NAR_INIT(self, *call_args, **call_kwargs)

yue_nar.CachedNAR.__init__ = patched_cached_nar_init

# ---------------------------------------------------------------------
# Патч NAR Synthesize: безопасная очистка VRAM и шаги диффузии
# ---------------------------------------------------------------------
ORIGINAL_NAR_SYNTHESIZE = yue_nar.synthesize

def safe_fast_synthesize(*args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    call_args = list(args)
    call_kwargs = dict(kwargs)

    if len(call_args) > 0:
        call_args[0] = SafeNARModelWrapper(call_args[0])
    elif "model" in call_kwargs:
        call_kwargs["model"] = SafeNARModelWrapper(call_kwargs["model"])

    if CURRENT_TASK_IS_FAST:
        modified = False
        for k in ("steps", "num_steps", "n_steps", "diffusion_steps"):
            if k in call_kwargs:
                call_kwargs[k] = 16
                modified = True
                break

        if not modified:
            try:
                sig = inspect.signature(ORIGINAL_NAR_SYNTHESIZE)
                param_names = list(sig.parameters.keys())
                for k in ("steps", "num_steps", "n_steps", "diffusion_steps"):
                    if k in param_names:
                        idx = param_names.index(k)
                        if idx < len(call_args):
                            call_args[idx] = 16
                            modified = True
                        else:
                            call_kwargs[k] = 16
                            modified = True
                        break
            except Exception:
                pass

        if not modified and len(call_args) >= 6 and isinstance(call_args[5], int):
            call_args[5] = 16

    return ORIGINAL_NAR_SYNTHESIZE(*call_args, **call_kwargs)

yue_nar.synthesize = safe_fast_synthesize

# ---------------------------------------------------------------------
# Патч Sampling Generate: безопасная модификация frozen dataclass
# ---------------------------------------------------------------------
ORIGINAL_SAMPLING_GENERATE = yue_sampling.generate_tokens

def safe_fast_generate_tokens(model, prefix, sampling=None, seed=42, phase=None, *args, **kwargs):
    call_kwargs = dict(kwargs)
    effective_sampling = sampling

    if CURRENT_TASK_IS_FAST and (phase == "song" or phase is None):
        if effective_sampling is not None:
            if dataclasses.is_dataclass(effective_sampling):
                changes = {}
                for field in dataclasses.fields(effective_sampling):
                    if field.name in ("max_tokens", "max_new_tokens", "max_length", "target_tokens"):
                        val = getattr(effective_sampling, field.name)
                        if isinstance(val, (int, float)) and val > 0:
                            changes[field.name] = min(int(val), 3200)
                if changes:
                    try:
                        effective_sampling = dataclasses.replace(effective_sampling, **changes)
                    except Exception:
                        pass
            elif isinstance(effective_sampling, dict):
                effective_sampling = dict(effective_sampling)
                for k in ("max_tokens", "max_new_tokens", "max_length", "target_tokens"):
                    if k in effective_sampling and isinstance(effective_sampling[k], (int, float)) and effective_sampling[k] > 0:
                        effective_sampling[k] = min(int(effective_sampling[k]), 3200)
            else:
                for attr in ("max_tokens", "max_new_tokens", "max_length", "target_tokens"):
                    if hasattr(effective_sampling, attr):
                        try:
                            val = getattr(effective_sampling, attr)
                            if isinstance(val, (int, float)) and val > 0:
                                setattr(effective_sampling, attr, min(int(val), 3200))
                        except Exception:
                            pass

        for k in ("max_tokens", "max_new_tokens", "max_length", "target_tokens"):
            if k in call_kwargs and isinstance(call_kwargs[k], (int, float)) and call_kwargs[k] > 0:
                call_kwargs[k] = min(int(call_kwargs[k]), 3200)

    res = ORIGINAL_SAMPLING_GENERATE(model, prefix, effective_sampling, seed, phase, *args, **call_kwargs)

    if CURRENT_TASK_IS_FAST and phase == "song" and res is not None:
        if isinstance(res, torch.Tensor) and res.shape[-1] > 3200:
            print(f"[FAST MODE] Обрезка токенов песни: {res.shape[-1]} -> 3200")
            res = res[..., :3200]
        elif isinstance(res, (list, tuple)) and len(res) > 3200:
            print(f"[FAST MODE] Обрезка токенов песни: {len(res)} -> 3200")
            res = res[:3200]

    return res

yue_sampling.generate_tokens = safe_fast_generate_tokens

def patch_all_yue_modules():
    for mod_name, mod in list(sys.modules.items()):
        if mod and mod_name.startswith("yue2"):
            if hasattr(mod, "generate_tokens") and getattr(mod, "generate_tokens") != safe_fast_generate_tokens:
                setattr(mod, "generate_tokens", safe_fast_generate_tokens)
            if hasattr(mod, "synthesize") and getattr(mod, "synthesize") != safe_fast_synthesize:
                setattr(mod, "synthesize", safe_fast_synthesize)
            if hasattr(mod, "CachedNAR"):
                c_cls = getattr(mod, "CachedNAR")
                if hasattr(c_cls, "__init__") and getattr(c_cls, "__init__") != patched_cached_nar_init:
                    c_cls.__init__ = patched_cached_nar_init

patch_all_yue_modules()

task_queue = queue.Queue()
task_lock = threading.Lock()
current_task = {
    "status": "idle",
    "progress_msg": "",
    "task_id": None,
    "error": None
}

def set_task_state(**kwargs):
    with task_lock:
        current_task.update(kwargs)

def get_task_state():
    with task_lock:
        return dict(current_task)

GLOBAL_PIPE = None

def get_pipeline():
    global GLOBAL_PIPE
    if GLOBAL_PIPE is None:
        set_task_state(progress_msg=f"Загрузка конфигурации YuE2-3B ({device.upper()})...")
        GLOBAL_PIPE = YuE2Pipeline.from_pretrained(
            "m-a-p/YuE2-3B",
            vae="m-a-p/YuE2-Vae",
            device=device,
            backend="torch-eager",
            cache_dir=str(MODELS_CACHE_DIR)
        )
        patch_all_yue_modules()
    return GLOBAL_PIPE

# =====================================================================
# БЛОК УПРАВЛЕНИЯ LoRA (Строго AR / MOT модель)
# =====================================================================

LORA_STATE = {
    "parent_obj": None,
    "ar_attr": None,
    "ar_original": None,
    "active_adapters": [],
}

_AR_STRICT_CANDIDATES = (
    "mot", "stage1", "stage1_model", "ar", "ar_model",
    "language_model", "ar_lm", "lm", "text_model"
)

def _ensure_pipe_models_loaded(pipe):
    for method_name in ("load_model", "_load_model", "load_models", "_load_models", "load_mot", "_load_mot"):
        if hasattr(pipe, method_name) and callable(getattr(pipe, method_name)):
            try:
                getattr(pipe, method_name)()
                break
            except Exception:
                pass

def _find_ar_module(pipe):
    _ensure_pipe_models_loaded(pipe)

    for name in _AR_STRICT_CANDIDATES:
        if hasattr(pipe, name):
            val = getattr(pipe, name)
            if val is not None and isinstance(val, torch.nn.Module):
                return pipe, name, val

    for container_name in ("models", "modules", "components", "submodules"):
        if hasattr(pipe, container_name):
            container = getattr(pipe, container_name)
            if isinstance(container, dict):
                for name in _AR_STRICT_CANDIDATES:
                    if name in container and isinstance(container[name], torch.nn.Module):
                        return container, name, container[name]

    for attr in dir(pipe):
        if attr.startswith("__") or attr in ("vae", "model"):
            continue
        try:
            val = getattr(pipe, attr)
            if isinstance(val, torch.nn.Module) and hasattr(val, "named_modules"):
                attr_lower = attr.lower()
                if "vae" not in attr_lower and "nar" not in attr_lower:
                    mod_names = [m[0] for m in val.named_modules()]
                    if any("layers" in m and "self_attn" in m for m in mod_names):
                        return pipe, attr, val
        except Exception:
            pass

    return None, None, None

def _prepare_lora_path(lora_path: Path, adapter_name: str) -> Path:
    if lora_path.is_dir() and (lora_path / "adapter_config.json").exists():
        return lora_path

    target_dir = lora_path.parent / f"_peft_{lora_path.stem}"
    target_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = target_dir / "adapter_config.json"

    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    cfg_dict = {
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": target_modules,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    cfg_path.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    weight_target = target_dir / f"adapter_model{lora_path.suffix}"
    if not weight_target.exists() or weight_target.stat().st_size != lora_path.stat().st_size:
        try:
            shutil.copyfile(lora_path, weight_target)
        except Exception:
            pass
    return target_dir

def _scale_single_adapter(peft_model, adapter_name: str, user_scale: float):
    try:
        user_scale = float(user_scale)
    except (TypeError, ValueError):
        return

    for module in peft_model.modules():
        if hasattr(module, "set_scale"):
            try:
                module.set_scale(adapter_name, user_scale)
                continue
            except Exception:
                pass
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling:
            r_val = getattr(module, "r", 16)
            if isinstance(r_val, dict):
                r_val = r_val.get(adapter_name, 16)
            alpha_val = getattr(module, "lora_alpha", 32)
            if isinstance(alpha_val, dict):
                alpha_val = alpha_val.get(adapter_name, 32)
            base_ratio = float(alpha_val) / float(r_val) if r_val else 1.0
            scaling[adapter_name] = base_ratio * user_scale

def _set_module_on_parent(parent, attr, module):
    if isinstance(parent, dict):
        parent[attr] = module
    else:
        setattr(parent, attr, module)

def _get_module_from_parent(parent, attr):
    if isinstance(parent, dict):
        return parent.get(attr)
    return getattr(parent, attr, None)

def set_active_adapters(peft_model, adapter_names: list[str]):
    if not adapter_names:
        return

    if len(adapter_names) == 1:
        if hasattr(peft_model, "set_adapter"):
            try:
                peft_model.set_adapter(adapter_names[0])
            except Exception as e:
                print(f"[LoRA Warning] peft_model.set_adapter({adapter_names[0]}): {e}")
        return

    for module in peft_model.modules():
        if module is peft_model:
            continue
        activated = False
        if hasattr(module, "set_adapter"):
            try:
                module.set_adapter(adapter_names)
                activated = True
            except Exception:
                pass
        if not activated:
            if hasattr(module, "_active_adapter"):
                try:
                    module._active_adapter = list(adapter_names)
                    activated = True
                except Exception:
                    pass
            elif hasattr(module, "active_adapter") and not isinstance(getattr(type(module), "active_adapter", None), property):
                try:
                    module.active_adapter = list(adapter_names)
                    activated = True
                except Exception:
                    pass

    try:
        if hasattr(peft_model, "_active_adapter"):
            peft_model._active_adapter = adapter_names[0]
        elif hasattr(peft_model, "active_adapter") and not isinstance(getattr(type(peft_model), "active_adapter", None), property):
            peft_model.active_adapter = adapter_names[0]
    except Exception:
        pass

def apply_dual_loras(pipe, vocal_lora, vocal_scale, style_lora, style_scale):
    loaded = {"vocal": False, "style": False}
    if PeftModel is None:
        print("[LoRA Error] Библиотека peft не установлена!")
        return loaded

    parent_obj, ar_attr, ar_base = _find_ar_module(pipe)
    if ar_base is None:
        print("[LoRA Error] Не удалось найти AR/MOT модель в пайплайне.")
        return loaded

    print(f"[LoRA Info] Найдена AR модель для адаптации: {ar_attr} ({type(ar_base).__name__})")

    if LORA_STATE["ar_original"] is None:
        LORA_STATE["parent_obj"] = parent_obj
        LORA_STATE["ar_attr"] = ar_attr
        LORA_STATE["ar_original"] = ar_base

    current_module = _get_module_from_parent(parent_obj, ar_attr)
    adapters_to_load = []

    if vocal_lora:
        try:
            v_path = safe_resolve(LORAS_DIR, vocal_lora)
            if v_path.exists():
                adapters_to_load.append(("vocal_adapter", v_path, vocal_scale, "vocal"))
        except Exception as e:
            print(f"[LoRA Error] Ошибка пути вокальной LoRA: {e}")

    if style_lora:
        try:
            s_path = safe_resolve(LORAS_DIR, style_lora)
            if s_path.exists():
                adapters_to_load.append(("style_adapter", s_path, style_scale, "style"))
        except Exception as e:
            print(f"[LoRA Error] Ошибка пути стилевой LoRA: {e}")

    if not adapters_to_load:
        return loaded

    try:
        peft_model = current_module
        first_name, first_path, first_scale, first_type = adapters_to_load[0]
        ready_first_path = _prepare_lora_path(first_path, first_name)

        if not isinstance(peft_model, PeftModel):
            peft_model = PeftModel.from_pretrained(ar_base, str(ready_first_path), adapter_name=first_name)
            _set_module_on_parent(parent_obj, ar_attr, peft_model)
        else:
            peft_model.load_adapter(str(ready_first_path), adapter_name=first_name)

        _scale_single_adapter(peft_model, first_name, first_scale)
        LORA_STATE["active_adapters"].append(first_name)
        loaded[first_type] = True
        print(f"[LoRA Info] Успешно загружен адаптер '{first_name}' (скейл {first_scale})")

        if len(adapters_to_load) > 1:
            second_name, second_path, second_scale, second_type = adapters_to_load[1]
            ready_second_path = _prepare_lora_path(second_path, second_name)
            peft_model.load_adapter(str(ready_second_path), adapter_name=second_name)
            _scale_single_adapter(peft_model, second_name, second_scale)
            LORA_STATE["active_adapters"].append(second_name)
            loaded[second_type] = True
            print(f"[LoRA Info] Успешно загружен адаптер '{second_name}' (скейл {second_scale})")

        active_names = [a[0] for a in adapters_to_load]
        set_active_adapters(peft_model, active_names)
        print(f"[LoRA Info] Активные адаптеры подключены к AR пайплайну: {active_names}")

    except Exception as e:
        print(f"[LoRA Error] Сбой подключения адаптеров: {e}")

    return loaded

def remove_dual_loras(pipe):
    parent = LORA_STATE.get("parent_obj")
    attr = LORA_STATE.get("ar_attr")
    original = LORA_STATE.get("ar_original")
    active = list(LORA_STATE.get("active_adapters", []))
    if not parent or not attr:
        return

    try:
        current = _get_module_from_parent(parent, attr)
        if isinstance(current, PeftModel):
            try:
                if hasattr(current, "disable_adapters"):
                    current.disable_adapters()
            except Exception:
                pass
            for a_name in active:
                if hasattr(current, "delete_adapter"):
                    try:
                        current.delete_adapter(a_name)
                    except Exception:
                        pass
            if hasattr(current, "unload"):
                try:
                    unloaded = current.unload()
                    _set_module_on_parent(parent, attr, unloaded)
                except Exception:
                    if original is not None:
                        _set_module_on_parent(parent, attr, original)
            elif original is not None:
                _set_module_on_parent(parent, attr, original)
        elif original is not None:
            _set_module_on_parent(parent, attr, original)
    except Exception as e:
        print(f"[LoRA Warning] Ошибка при отключении адаптеров: {e}")
        if original is not None:
            try:
                _set_module_on_parent(parent, attr, original)
            except Exception:
                pass
    finally:
        LORA_STATE["parent_obj"] = None
        LORA_STATE["ar_attr"] = None
        LORA_STATE["ar_original"] = None
        LORA_STATE["active_adapters"] = []

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
    for item in sorted(LORAS_DIR.iterdir()):
        if item.name.startswith((".", "_")):
            continue
        if item.is_dir():
            weights = [f for f in item.iterdir() if f.suffix.lower() in (".safetensors", ".bin", ".pt")]
            if weights or (item / "adapter_config.json").exists():
                size_mb = round(sum(f.stat().st_size for f in weights) / (1024 * 1024), 1) if weights else 0.0
                loras.append({
                    "filename": item.name,
                    "name": item.name.replace("_", " ").title(),
                    "size": f"{size_mb} MB"
                })
        elif item.is_file() and item.suffix.lower() in (".safetensors", ".bin", ".pt"):
            size_mb = round(item.stat().st_size / (1024 * 1024), 1)
            loras.append({
                "filename": item.name,
                "name": item.stem.replace("_", " ").title(),
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

def generation_worker():
    global CURRENT_TASK_IS_FAST
    while True:
        task = task_queue.get()
        if task is None:
            break

        task_id = task["id"]
        set_task_state(status="running", task_id=task_id, progress_msg="Подготовка параметров генерации...", error=None)

        CURRENT_TASK_IS_FAST = task.get("fast_mode", True)
        start_time = time.time()
        pipe = None
        loras_status = {"vocal": False, "style": False}

        try:
            pipe = get_pipeline()
            _ensure_pipe_models_loaded(pipe)
            patch_all_yue_modules()

            vocal_lora = task.get("vocal_lora")
            vocal_scale = float(task.get("vocal_scale", 0.8))
            style_lora = task.get("style_lora")
            style_scale = float(task.get("style_scale", 0.8))

            if vocal_lora or style_lora:
                set_task_state(progress_msg="Применение адаптеров LoRA (Вокал/Стиль)...")
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
                set_task_state(progress_msg="Синтез по партитуре (MIDI/ABC)...")
                gen_kwargs["abc"] = sanitize_abc_notation(custom_abc, task.get("title") or "Track")
                gen_kwargs["cot"] = "melody"
            else:
                chosen_cot = "melody" if audio_file else task.get("cot", "full")
                set_task_state(progress_msg=f"Символическое планирование (cot='{chosen_cot}')...")
                gen_kwargs["cot"] = chosen_cot

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            song = pipe(**gen_kwargs)

            set_task_state(progress_msg="Сохранение аудио и нотных артефактов...")
            filename = f"track_{task_id}.flac"
            file_path = safe_resolve(TRACKS_DIR, filename)

            if hasattr(song, "save"):
                song.save(str(file_path))
            else:
                audio_data = song.audio if hasattr(song, "audio") else song
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.detach().cpu().float().numpy()
                if audio_data.ndim == 2 and audio_data.shape[0] < audio_data.shape[1]:
                    audio_data = audio_data.T
                sf.write(str(file_path), audio_data, 48000)

            clean_tid = safe_track_id(task_id)
            track_artifacts_dir = safe_resolve(ARTIFACTS_BASE_DIR, clean_tid)
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

            set_task_state(status="completed", progress_msg=f"Готово за {duration_sec} сек!")

        except Exception as e:
            traceback.print_exc()
            print(f"[Error in generation_worker] {e}")
            set_task_state(status="error", error=str(e), progress_msg=f"Ошибка: {str(e)}")
        finally:
            if pipe and LORA_STATE["parent_obj"] is not None:
                remove_dual_loras(pipe)
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
            gc.collect()
            task_queue.task_done()

threading.Thread(target=generation_worker, daemon=True).start()

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

        # -------------------------------------------------------------
        # Статика Demucs Web (Экстракция стэмов)
        # -------------------------------------------------------------
        if parsed.path.startswith("/demucs-web"):
            subpath = parsed.path.replace("/demucs-web", "").lstrip("/")
            if not subpath or subpath == "":
                subpath = "index.html"
            target_file = (BASE_DIR / "demucs-web" / subpath).resolve()

            if not target_file.is_relative_to((BASE_DIR / "demucs-web").resolve()):
                self.send_error(403, "Forbidden")
                return

            if not target_file.exists() or not target_file.is_file():
                self.send_error(404, "File not found")
                return

            ext = target_file.suffix.lower()
            mime_map = {
                ".html": "text/html; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".mjs": "application/javascript; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".wasm": "application/wasm",
                ".onnx": "application/octet-stream"
            }
            content_type = mime_map.get(ext, "application/octet-stream")

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(target_file.stat().st_size))
            self.end_headers()
            with open(target_file, "rb") as f:
                self.wfile.write(f.read())
            return

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
            raw_filename = parsed.path.replace("/audio/", "")
            try:
                file_path = safe_resolve(TRACKS_DIR, raw_filename)
            except PermissionError:
                self.send_error(403, "Forbidden")
                return

            if not file_path.exists() or not file_path.is_file():
                self.send_error(404, "Audio not found")
                return

            file_size = file_path.stat().st_size
            range_header = self.headers.get("Range")

            if range_header and range_header.startswith("bytes="):
                byte_range = range_header.split("=")[1].strip()
                if "," not in byte_range:
                    if byte_range.startswith("-"):
                        suffix_len = int(byte_range[1:])
                        start = max(0, file_size - suffix_len)
                        end = file_size - 1
                    elif byte_range.endswith("-"):
                        start = int(byte_range[:-1])
                        end = file_size - 1
                    else:
                        parts = byte_range.split("-")
                        start = int(parts[0])
                        end = min(file_size - 1, int(parts[1]))

                    if start <= end and start < file_size:
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

            self.send_response(200)
            self.send_header("Content-Type", "audio/flac")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
            return

        if parsed.path == "/api/score":
            raw_id = qs.get("id", [""])[0]
            clean_id = safe_track_id(raw_id)
            try:
                track_artifacts_dir = safe_resolve(ARTIFACTS_BASE_DIR, clean_id)
            except PermissionError:
                self.send_error(403, "Forbidden")
                return

            score_file = track_artifacts_dir / "score.abc"

            if not score_file.exists():
                found = list(track_artifacts_dir.rglob("*.abc")) if track_artifacts_dir.exists() else []
                if found:
                    score_file = found[0]

            if score_file.exists():
                abc_content = score_file.read_text(encoding="utf-8", errors="ignore")
            else:
                history = load_history()
                item = next((x for x in history if x["id"] == raw_id), None)
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
            raw_id = qs.get("id", [""])[0]
            clean_id = safe_track_id(raw_id)
            try:
                track_artifacts_dir = safe_resolve(ARTIFACTS_BASE_DIR, clean_id)
            except PermissionError:
                self.send_error(403, "Forbidden")
                return

            track_artifacts_dir.mkdir(parents=True, exist_ok=True)
            score_file = track_artifacts_dir / "score.abc"
            midi_file = track_artifacts_dir / "score.mid"

            if not score_file.exists():
                found = list(track_artifacts_dir.rglob("*.abc"))
                if found:
                    shutil.copyfile(found[0], score_file)
                else:
                    history = load_history()
                    item = next((x for x in history if x["id"] == raw_id), None)
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
                    except Exception:
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
            self.send_header("Content-Disposition", f'attachment; filename="track_{clean_id}.mid"')
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

        if parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({
                "queue_size": task_queue.qsize(),
                "current_task": get_task_state()
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
                boundary = content_type.split("boundary=")[1].strip().encode("utf-8")
                parts = raw_data.split(b"--" + boundary)
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
                            if body_part.endswith(b"\r\n"):
                                body_part = body_part[:-2]
                            if body_part.endswith(b"--"):
                                body_part = body_part[:-2]
                            saved_path = safe_resolve(UPLOADS_DIR, saved_filename)
                            with open(saved_path, "wb") as f:
                                f.write(body_part)
                            break
            else:
                saved_filename = f"file_{int(time.time())}.bin"
                saved_path = safe_resolve(UPLOADS_DIR, saved_filename)
                with open(saved_path, "wb") as f:
                    f.write(raw_data)
                is_midi = False

            abc_content = ""
            if is_midi and m21_converter is not None and saved_filename:
                try:
                    target_file = safe_resolve(UPLOADS_DIR, saved_filename)
                    midi_score = m21_converter.parse(str(target_file), format="midi")
                    tmp_abc_path = safe_resolve(UPLOADS_DIR, f"{saved_filename}.abc")
                    midi_score.write("abc", fp=str(tmp_abc_path))
                    if tmp_abc_path.exists():
                        raw_abc = tmp_abc_path.read_text(encoding="utf-8", errors="ignore")
                        abc_content = sanitize_abc_notation(raw_abc, saved_filename)
                        tmp_abc_path.unlink()
                except Exception:
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
            raw_id = data.get("id", "")
            clean_id = safe_track_id(raw_id)
            history = load_history()
            track_item = next((t for t in history if t["id"] == raw_id), None)
            if track_item:
                filename = track_item.get("filename")
                if filename:
                    try:
                        flac_path = safe_resolve(TRACKS_DIR, filename)
                        if flac_path.exists():
                            flac_path.unlink()
                    except Exception:
                        pass
                if clean_id:
                    try:
                        art_dir = safe_resolve(ARTIFACTS_BASE_DIR, clean_id)
                        if art_dir.exists():
                            shutil.rmtree(art_dir)
                    except Exception:
                        pass
                history = [t for t in history if t["id"] != raw_id]
                save_history(history)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
            return

        if parsed.path == "/api/loras/delete":
            lora_fn = data.get("filename", "")
            try:
                target_path = safe_resolve(LORAS_DIR, lora_fn)
                if target_path.exists():
                    if target_path.is_dir():
                        shutil.rmtree(target_path)
                    else:
                        target_path.unlink()
                    alt_dir = LORAS_DIR / f"_peft_{target_path.stem}"
                    if alt_dir.exists():
                        shutil.rmtree(alt_dir)
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
            raw_audio = data.get("audio_file")
            clean_audio = Path(raw_audio).name if raw_audio else None
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
                "audio_file": clean_audio,
                "abc_score": data.get("abc_score", ""),
                "midi_source": bool(data.get("midi_source", False)),
                "vocal_lora": data.get("vocal_lora") or None,
                "vocal_scale": float(data.get("vocal_scale", 0.8)),
                "style_lora": data.get("style_lora") or None,
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
    print(f" Модуль стэмов Demucs Web: http://127.0.0.1:{port}/demucs-web/")
    print(f" Доступ в браузере: http://127.0.0.1:{port}")
    print("=" * 65)
    server.serve_forever()

if __name__ == "__main__":
    run_server()