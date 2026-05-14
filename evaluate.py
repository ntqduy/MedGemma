#!/usr/bin/env python3
"""MedGemma evaluation for M3D captioning and VQA.

The evaluator is intentionally self-contained because this project keeps only
thin entrypoints. It supports robust sample-level error handling, optional NLP
metrics, prediction export, and runtime benchmarking.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import logging
import math
import os
import re
import shlex
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
CHOICE_LETTERS = ("A", "B", "C", "D", "E")


@dataclass
class EvalSample:
    sample_id: str
    image_path: str
    prompt: str
    ground_truth: str
    split: str = ""
    question: str = ""
    choices: Dict[str, str] = field(default_factory=dict)
    answer_choice: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelBundle:
    model: Any
    processor: Any
    tokenizer: Any
    torch: Any
    device: Any
    dtype: Any


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if (value[0:1] == value[-1:] == '"') or (value[0:1] == value[-1:] == "'"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if lowered == "full":
        return "full"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def minimal_yaml_load(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        key_value = line.strip()
        if ":" not in key_value:
            continue
        key, raw_value = key_value.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(raw_value)
    return root


def load_config(config_path: Path) -> Dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Config root must be a mapping.")
        return loaded
    except ImportError:
        return minimal_yaml_load(text)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def save_yaml_or_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore

        path.write_text(
            yaml.safe_dump(to_jsonable(payload), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except ImportError:
        json_path = path.with_suffix(".json")
        json_path.write_text(
            json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(handle: Any, payload: Dict[str, Any]) -> None:
    handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")
    handle.flush()


def package_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def get_package_availability() -> Dict[str, bool]:
    return {
        "tqdm": package_available("tqdm"),
        "PIL": package_available("PIL"),
        "numpy": package_available("numpy"),
        "nibabel": package_available("nibabel"),
        "torch": package_available("torch"),
        "transformers": package_available("transformers"),
        "nltk": package_available("nltk"),
        "bert_score": package_available("bert_score"),
        "pycocoevalcap": package_available("pycocoevalcap"),
    }


def setup_logging(output_dir: Path, verbose: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("medgemma_eval")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(output_dir / "log.txt", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def resolve_config_path(value: Any, config_dir: Optional[Path] = None) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if path.is_absolute():
        return path
    candidates = [
        (PROJECT_ROOT / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    if config_dir is not None:
        candidates.append((config_dir / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_output_root(value: Any) -> Path:
    if value is None or str(value).strip() == "":
        return PROJECT_ROOT / "results"
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def path_variants(path_value: str) -> Iterable[Path]:
    path = Path(path_value)
    yield path
    lowered = path_value.lower()
    if lowered.endswith(".npy"):
        stem = path_value[:-4]
        for suffix in (".nii.gz", ".nii", ".png", ".jpg", ".jpeg"):
            yield Path(stem + suffix)
    elif lowered.endswith(".nii.gz"):
        stem = path_value[:-7]
        for suffix in (".npy", ".nii", ".png", ".jpg", ".jpeg"):
            yield Path(stem + suffix)
    elif lowered.endswith(".nii"):
        stem = path_value[:-4]
        for suffix in (".npy", ".nii.gz", ".png", ".jpg", ".jpeg"):
            yield Path(stem + suffix)


def resolve_relative_existing(path_value: str, roots: Sequence[Path]) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
    if path.is_absolute():
        for variant in path_variants(str(path)):
            if variant.exists():
                return variant
        return path
    expanded_roots = [root for root in roots if root is not None]
    expanded_roots.extend([PROJECT_ROOT, Path.cwd()])
    for root in expanded_roots:
        for variant in path_variants(str(path)):
            candidate = (root / variant).resolve()
            if candidate.exists():
                return candidate
    return (expanded_roots[0] / path).resolve() if expanded_roots else path.resolve()


def normalize_sample_spec(sample_value: Any) -> Tuple[Optional[int], str]:
    if sample_value is None or str(sample_value).strip().lower() == "full":
        return None, "full"
    try:
        count = int(sample_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid sample value: {sample_value!r}. Use an integer or 'full'.") from exc
    if count <= 0:
        raise ValueError("Sample count must be positive.")
    return count, str(count)


def infer_task_name(args_task: Optional[str], config: Dict[str, Any], config_path: Path) -> str:
    raw = args_task or config.get("task") or config.get("task_name") or config_path.stem
    value = str(raw).lower()
    if value in {"cap", "caption", "captioning", "image_captioning", "image-captioning"}:
        return "cap"
    if value in {"vqa", "visual_question_answering", "visual-question-answering"}:
        return "vqa"
    if "cap" in value:
        return "cap"
    if "vqa" in value:
        return "vqa"
    raise ValueError(f"Cannot infer task from {raw!r}. Pass --task cap or --task vqa.")


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def detect_field(keys: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    key_by_norm = {normalize_key(k): k for k in keys}
    for candidate in candidates:
        if candidate in keys:
            return candidate
        normalized = normalize_key(candidate)
        if normalized in key_by_norm:
            return key_by_norm[normalized]
    return None


def read_text_maybe_path(value: Any, roots: Sequence[Path]) -> Tuple[str, Optional[str], Optional[str]]:
    if value is None:
        return "", None, "Missing text value."
    if not isinstance(value, str):
        return str(value), None, None

    text = value.strip()
    looks_like_file = bool(re.search(r"[\\/]", text)) or text.lower().endswith((".txt", ".md"))
    if looks_like_file:
        candidate = resolve_relative_existing(text, roots)
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace").strip(), str(candidate), None
        return "", str(candidate), f"Ground-truth text file not found: {candidate}"
    return text, None, None


def detect_caption_schema(rows: Sequence[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    keys: List[str] = []
    for row in rows[:20]:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    return {
        "id": detect_field(keys, ("id", "sample_id", "uid", "case_id", "image_id")),
        "image": detect_field(keys, ("image", "image_path", "Image Path", "path", "file", "filename")),
        "text": detect_field(keys, ("text", "caption", "report", "ground_truth", "findings", "label")),
    }


def load_caption_samples(
    config: Dict[str, Any],
    dataset_path: Path,
    image_root: Path,
    split: str,
    max_samples: Optional[int],
    logger: logging.Logger,
) -> Tuple[List[EvalSample], Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    available_splits: List[str] = []
    if isinstance(data, dict):
        available_splits = [str(k) for k in data.keys()]
        if split not in data:
            raise ValueError(f"Split {split!r} not found. Available splits: {available_splits}")
        rows = data[split]
    elif isinstance(data, list):
        rows = data
        split = split or "all"
    else:
        raise ValueError("Caption dataset must be a JSON object of splits or a list of records.")

    if not isinstance(rows, list):
        raise ValueError(f"Caption split {split!r} must be a list.")

    schema = detect_caption_schema(rows)
    if not schema["image"] or not schema["text"]:
        raise ValueError(f"Could not detect caption schema from keys: {schema}")

    selected_rows = rows[:max_samples] if max_samples is not None else rows
    prompt_template = str(config.get("prompt_template") or "<start_of_image> findings:")
    roots = [image_root, dataset_path.parent, dataset_path.parent.parent]
    samples: List[EvalSample] = []

    for index, row in enumerate(selected_rows):
        if not isinstance(row, dict):
            continue
        image_value = str(row.get(schema["image"], "")).strip()
        image_path = resolve_relative_existing(image_value, roots)
        gt, gt_path, gt_error = read_text_maybe_path(row.get(schema["text"]), roots)
        raw_id = row.get(schema["id"]) if schema["id"] else None
        sample_id = str(raw_id or f"{split}_{index:06d}")
        samples.append(
            EvalSample(
                sample_id=sample_id,
                image_path=str(image_path),
                prompt=prompt_template,
                ground_truth=gt,
                split=split,
                meta={
                    "raw_image": image_value,
                    "ground_truth_path": gt_path,
                    "ground_truth_error": gt_error,
                    "row_index": index,
                },
            )
        )

    schema_info = {
        "available_splits": available_splits,
        "selected_split": split,
        "schema": schema,
        "total_rows_in_split": len(rows),
        "selected_rows": len(samples),
    }
    logger.info("Caption schema: %s", json.dumps(schema_info, ensure_ascii=False))
    return samples, schema_info


def detect_vqa_schema(fieldnames: Sequence[str], config: Dict[str, Any]) -> Dict[str, Optional[str]]:
    configured = config.get("columns") or {}
    schema = {
        "id": configured.get("id") or detect_field(fieldnames, ("id", "sample_id", "uid", "image_id")),
        "image": configured.get("image") or detect_field(fieldnames, ("Image Path", "image", "image_path", "path", "file")),
        "question": configured.get("question") or detect_field(fieldnames, ("Question", "question", "query", "prompt")),
        "answer": configured.get("answer") or detect_field(fieldnames, ("Answer", "answer", "gt_answer", "ground_truth")),
        "answer_choice": configured.get("answer_choice") or detect_field(fieldnames, ("Answer Choice", "answer_choice", "label")),
    }
    for letter in CHOICE_LETTERS:
        schema[f"choice_{letter.lower()}"] = configured.get(f"choice_{letter.lower()}") or detect_field(
            fieldnames, (f"Choice {letter}", f"choice_{letter.lower()}", f"option_{letter.lower()}", letter)
        )
    return schema


def render_choices(choices: Dict[str, str]) -> str:
    lines = [f"{letter}. {text}" for letter, text in choices.items() if text]
    if not lines:
        return ""
    return "Choices:\n" + "\n".join(lines)


def build_vqa_prompt(prompt_template: str, question: str, choices: Dict[str, str]) -> str:
    choices_text = render_choices(choices)
    return prompt_template.format(question=question, choices=choices_text, answer_options=choices_text).strip()


def load_vqa_samples(
    config: Dict[str, Any],
    dataset_path: Path,
    image_root: Path,
    max_samples: Optional[int],
    logger: logging.Logger,
) -> Tuple[List[EvalSample], Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        schema = detect_vqa_schema(fieldnames, config)
        required = ("image", "question", "answer")
        missing = [key for key in required if not schema.get(key)]
        if missing:
            raise ValueError(f"Could not detect VQA columns {missing}. Fieldnames: {fieldnames}")
        rows = list(reader)

    selected_rows = rows[:max_samples] if max_samples is not None else rows
    prompt_template = str(
        config.get("prompt_template")
        or "<start_of_image> Answer the medical visual question concisely.\nQuestion: {question}\n{choices}\nAnswer:"
    )
    roots = [image_root, dataset_path.parent, dataset_path.parent.parent]
    samples: List[EvalSample] = []

    for index, row in enumerate(selected_rows):
        image_value = str(row.get(schema["image"] or "", "")).strip()
        image_path = resolve_relative_existing(image_value, roots)
        question = str(row.get(schema["question"] or "", "")).strip()
        answer = str(row.get(schema["answer"] or "", "")).strip()
        answer_choice = str(row.get(schema["answer_choice"] or "", "")).strip() if schema.get("answer_choice") else ""
        choices = {}
        for letter in CHOICE_LETTERS:
            column = schema.get(f"choice_{letter.lower()}")
            if column:
                value = str(row.get(column, "")).strip()
                if value:
                    choices[letter] = value
        if not answer and answer_choice in choices:
            answer = choices[answer_choice]
        raw_id = row.get(schema["id"]) if schema.get("id") else None
        sample_id = str(raw_id or f"vqa_{index:06d}")
        samples.append(
            EvalSample(
                sample_id=sample_id,
                image_path=str(image_path),
                prompt=build_vqa_prompt(prompt_template, question, choices),
                ground_truth=answer,
                question=question,
                choices=choices,
                answer_choice=answer_choice,
                meta={"raw_image": image_value, "row_index": index},
            )
        )

    schema_info = {
        "schema": schema,
        "fieldnames": fieldnames,
        "total_rows": len(rows),
        "selected_rows": len(samples),
    }
    logger.info("VQA schema: %s", json.dumps(schema_info, ensure_ascii=False))
    return samples, schema_info


def choose_slice_axis(shape: Sequence[int], requested_axis: Any) -> int:
    if requested_axis not in (None, "", "auto"):
        axis = int(requested_axis)
        return axis if axis >= 0 else len(shape) + axis
    if len(shape) != 3:
        return 0
    min_axis = min(range(3), key=lambda idx: shape[idx])
    max_dim = max(shape)
    if shape[min_axis] < max_dim * 0.75:
        return min_axis
    return 0


def normalize_array_to_uint8(array: Any) -> Any:
    import numpy as np  # type: ignore

    arr = np.asarray(array)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype("float32", copy=False)
    low, high = np.percentile(arr, [1, 99])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low, high = float(arr.min()), float(arr.max())
    if high <= low:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - low) / (high - low)
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def array_to_pil_image(array: Any, image_config: Dict[str, Any]) -> Any:
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    arr = np.asarray(array)
    arr = np.squeeze(arr)
    while arr.ndim > 3:
        arr = np.take(arr, arr.shape[0] // 2, axis=0)

    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., :3]
        elif arr.shape[0] in (3, 4):
            arr = np.moveaxis(arr[:3, ...], 0, -1)
        else:
            axis = choose_slice_axis(arr.shape, image_config.get("slice_axis", "auto"))
            arr = np.take(arr, arr.shape[axis] // 2, axis=axis)

    if arr.ndim != 2 and not (arr.ndim == 3 and arr.shape[-1] in (3, 4)):
        raise ValueError(f"Unsupported image array shape: {arr.shape}")

    arr = normalize_array_to_uint8(arr)
    if arr.ndim == 2:
        image = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        image = Image.fromarray(arr[..., :3]).convert("RGB")
    return image


def load_image_as_rgb(path: Path, image_config: Dict[str, Any]) -> Any:
    lowered = str(path).lower()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if lowered.endswith(".npy"):
        if not package_available("numpy"):
            raise ImportError("numpy is required to read .npy images.")
        import numpy as np  # type: ignore

        return array_to_pil_image(np.load(path), image_config)
    if lowered.endswith(".nii") or lowered.endswith(".nii.gz"):
        if not package_available("nibabel"):
            raise ImportError("nibabel is required to read NIfTI images.")
        import nibabel as nib  # type: ignore

        return array_to_pil_image(nib.load(str(path)).get_fdata(), image_config)

    if not package_available("PIL"):
        raise ImportError("Pillow is required to read image files.")
    from PIL import Image  # type: ignore

    return Image.open(path).convert("RGB")


def get_torch_dtype(torch: Any, dtype_name: Any) -> Any:
    name = str(dtype_name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[name]


def get_requested_device(torch: Any, device_name: Any) -> str:
    name = str(device_name or "auto").lower()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def load_model_bundle(config: Dict[str, Any], model_path: Path, logger: logging.Logger) -> Tuple[ModelBundle, Dict[str, Any]]:
    logger.info("Model path: %s", model_path)
    try:
        import torch  # type: ignore
        from transformers import AutoProcessor  # type: ignore

        try:
            from transformers import AutoModelForImageTextToText  # type: ignore
        except ImportError:
            AutoModelForImageTextToText = None  # type: ignore
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError:
            AutoTokenizer = None  # type: ignore

        device_name = get_requested_device(torch, config.get("device", "auto"))
        dtype = get_torch_dtype(torch, config.get("dtype", "auto"))
        local_files_only = bool(config.get("local_files_only", True))

        processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=local_files_only,
            trust_remote_code=bool(config.get("trust_remote_code", True)),
        )
        logger.info("Processor loaded successfully")

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None and AutoTokenizer is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=local_files_only,
                trust_remote_code=bool(config.get("trust_remote_code", True)),
            )
        logger.info("Tokenizer loaded successfully")

        model_kwargs = {
            "torch_dtype": dtype,
            "local_files_only": local_files_only,
            "trust_remote_code": bool(config.get("trust_remote_code", True)),
        }
        device_map = config.get("device_map", "auto")
        if device_name != "cpu" and device_map:
            model_kwargs["device_map"] = device_map

        if AutoModelForImageTextToText is None:
            from transformers import AutoModelForVision2Seq  # type: ignore

            model = AutoModelForVision2Seq.from_pretrained(str(model_path), **model_kwargs)
        else:
            model = AutoModelForImageTextToText.from_pretrained(str(model_path), **model_kwargs)
        if device_name == "cpu":
            model.to(device_name)
        model.eval()
        logger.info("Model loaded successfully")

        device = getattr(model, "device", None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device(device_name)
        logger.info("Device: %s", device)
        logger.info("Dtype: %s", dtype)

        stats = collect_model_stats(model, torch)
        return ModelBundle(model=model, processor=processor, tokenizer=tokenizer, torch=torch, device=device, dtype=dtype), stats
    except Exception:
        logger.exception("Model loading failed")
        raise


def collect_model_stats(model: Any, torch: Any) -> Dict[str, Any]:
    total_params = 0
    trainable_params = 0
    first_dtype = None
    for parameter in model.parameters():
        count = int(parameter.numel())
        total_params += count
        if getattr(parameter, "requires_grad", False):
            trainable_params += count
        if first_dtype is None:
            first_dtype = str(parameter.dtype)
    stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "parameter_dtype": first_dtype,
    }
    if torch.cuda.is_available():
        stats["gpu_name"] = torch.cuda.get_device_name(0)
        stats["gpu_memory_allocated_bytes"] = int(torch.cuda.memory_allocated())
        stats["gpu_memory_reserved_bytes"] = int(torch.cuda.memory_reserved())
        stats["gpu_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    else:
        stats["gpu_name"] = None
        stats["gpu_memory_allocated_bytes"] = None
        stats["gpu_memory_reserved_bytes"] = None
        stats["gpu_max_memory_allocated_bytes"] = None
    return stats


def prepare_generation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    generation = dict(config.get("generation") or {})
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(generation.get("max_new_tokens", 128)),
        "do_sample": bool(generation.get("do_sample", False)),
    }
    if generation.get("num_beams") is not None:
        kwargs["num_beams"] = int(generation.get("num_beams"))
    if kwargs["do_sample"]:
        if generation.get("temperature") is not None:
            kwargs["temperature"] = float(generation.get("temperature"))
        if generation.get("top_p") is not None:
            kwargs["top_p"] = float(generation.get("top_p"))
        if generation.get("top_k") is not None:
            kwargs["top_k"] = int(generation.get("top_k"))
    return kwargs


def move_inputs_to_device(inputs: Any, bundle: ModelBundle) -> Any:
    torch = bundle.torch
    if hasattr(inputs, "to"):
        try:
            return inputs.to(bundle.device, dtype=bundle.dtype)
        except TypeError:
            return inputs.to(bundle.device)

    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if torch.is_tensor(value) and value.is_floating_point():
                moved[key] = value.to(device=bundle.device, dtype=bundle.dtype)
            else:
                moved[key] = value.to(device=bundle.device)
        else:
            moved[key] = value
    return moved


def generate_prediction(
    bundle: ModelBundle,
    image: Any,
    prompt: str,
    generation_kwargs: Dict[str, Any],
) -> Tuple[str, float, int]:
    torch = bundle.torch
    processor = bundle.processor
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = move_inputs_to_device(inputs, bundle)
    input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        generation = bundle.model.generate(**inputs, **generation_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    sequence = generation[0]
    is_encoder_decoder = bool(getattr(getattr(bundle.model, "config", None), "is_encoder_decoder", False))
    if input_len and not is_encoder_decoder and int(sequence.shape[-1]) >= input_len:
        sequence = sequence[input_len:]
    generated_tokens = int(sequence.shape[-1])

    decoder = getattr(processor, "decode", None) or getattr(bundle.tokenizer, "decode", None)
    if decoder is None:
        raise RuntimeError("Neither processor nor tokenizer exposes decode().")
    text = decoder(sequence, skip_special_tokens=True)
    return str(text).strip(), elapsed, generated_tokens


def metric_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", str(text).lower())


def ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def corpus_bleu_scores(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, Optional[float]]:
    if not predictions:
        return {f"BLEU-{i}": None for i in range(1, 5)}
    pred_tokens = [metric_tokens(text) for text in predictions]
    ref_tokens = [metric_tokens(text) for text in references]
    pred_len = sum(len(tokens) for tokens in pred_tokens)
    ref_len = sum(len(tokens) for tokens in ref_tokens)
    if pred_len == 0:
        return {f"BLEU-{i}": 0.0 for i in range(1, 5)}
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1.0 - (ref_len / max(pred_len, 1)))
    scores: Dict[str, Optional[float]] = {}
    for max_n in range(1, 5):
        precisions = []
        for n in range(1, max_n + 1):
            matched = 0
            total = 0
            for pred, ref in zip(pred_tokens, ref_tokens):
                pred_counts = ngram_counts(pred, n)
                ref_counts = ngram_counts(ref, n)
                matched += sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
                total += sum(pred_counts.values())
            precisions.append(matched / total if total else 0.0)
        if any(precision <= 0.0 for precision in precisions):
            scores[f"BLEU-{max_n}"] = 0.0
        else:
            scores[f"BLEU-{max_n}"] = float(
                brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_n)
            )
    return scores


def rouge_n_pair(pred_tokens: Sequence[str], ref_tokens: Sequence[str], n: int) -> float:
    pred_counts = ngram_counts(pred_tokens, n)
    ref_counts = ngram_counts(ref_tokens, n)
    if not pred_counts or not ref_counts:
        return 0.0
    overlap = sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(ref_counts.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            if token == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_pair(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_scores(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, Optional[float]]:
    if not predictions:
        return {"ROUGE-1": None, "ROUGE-2": None, "ROUGE-L": None}
    rouge_1 = []
    rouge_2 = []
    rouge_l = []
    for pred, ref in zip(predictions, references):
        pred_tokens = metric_tokens(pred)
        ref_tokens = metric_tokens(ref)
        rouge_1.append(rouge_n_pair(pred_tokens, ref_tokens, 1))
        rouge_2.append(rouge_n_pair(pred_tokens, ref_tokens, 2))
        rouge_l.append(rouge_l_pair(pred_tokens, ref_tokens))
    return {
        "ROUGE-1": float(statistics.mean(rouge_1)),
        "ROUGE-2": float(statistics.mean(rouge_2)),
        "ROUGE-L": float(statistics.mean(rouge_l)),
    }


def meteor_score_safe(
    predictions: Sequence[str],
    references: Sequence[str],
    logger: logging.Logger,
) -> Optional[float]:
    if not predictions:
        return None
    if not package_available("nltk"):
        logger.warning("METEOR unavailable: nltk is not installed")
        return None
    try:
        from nltk.translate.meteor_score import meteor_score  # type: ignore

        scores = []
        for pred, ref in zip(predictions, references):
            scores.append(float(meteor_score([metric_tokens(ref)], metric_tokens(pred))))
        return float(statistics.mean(scores)) if scores else None
    except Exception as exc:
        logger.warning("METEOR failed and will be saved as null: %s", exc)
        logger.debug("METEOR traceback:\n%s", traceback.format_exc())
        return None


def bertscore_safe(
    predictions: Sequence[str],
    references: Sequence[str],
    metrics_config: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, float]]:
    if not predictions:
        return None
    if not package_available("bert_score"):
        logger.warning("BERTScore unavailable: bert_score is not installed")
        return None
    try:
        from bert_score import score as bert_score  # type: ignore

        kwargs: Dict[str, Any] = {
            "lang": str(metrics_config.get("bertscore_lang", "en")),
            "verbose": bool(metrics_config.get("bertscore_verbose", False)),
            "rescale_with_baseline": bool(metrics_config.get("bertscore_rescale_with_baseline", False)),
        }
        if metrics_config.get("bertscore_model_type"):
            kwargs["model_type"] = str(metrics_config.get("bertscore_model_type"))
        if metrics_config.get("bertscore_device"):
            kwargs["device"] = str(metrics_config.get("bertscore_device"))
        precision, recall, f1 = bert_score(list(predictions), list(references), **kwargs)
        return {
            "Precision": float(precision.mean().item()),
            "Recall": float(recall.mean().item()),
            "F1": float(f1.mean().item()),
        }
    except Exception as exc:
        logger.warning("BERTScore failed and will be saved as null: %s", exc)
        logger.debug("BERTScore traceback:\n%s", traceback.format_exc())
        return None


def cider_safe(
    predictions: Sequence[str],
    references: Sequence[str],
    logger: logging.Logger,
) -> Optional[float]:
    if not predictions:
        return None
    if not package_available("pycocoevalcap"):
        logger.warning("CIDEr unavailable: pycocoevalcap is not installed")
        return None
    try:
        from pycocoevalcap.cider.cider import Cider  # type: ignore

        gts = {index: [ref] for index, ref in enumerate(references)}
        res = {index: [pred] for index, pred in enumerate(predictions)}
        score, _ = Cider().compute_score(gts, res)
        return float(score)
    except Exception as exc:
        logger.warning("CIDEr failed and will be saved as null: %s", exc)
        logger.debug("CIDEr traceback:\n%s", traceback.format_exc())
        return None


def spice_safe(
    predictions: Sequence[str],
    references: Sequence[str],
    logger: logging.Logger,
) -> Optional[float]:
    if not predictions:
        return None
    if not package_available("pycocoevalcap"):
        logger.warning("SPICE unavailable: pycocoevalcap is not installed")
        return None
    try:
        from pycocoevalcap.spice.spice import Spice  # type: ignore

        gts = {index: [ref] for index, ref in enumerate(references)}
        res = {index: [pred] for index, pred in enumerate(predictions)}
        score, _ = Spice().compute_score(gts, res)
        return float(score)
    except Exception as exc:
        logger.warning("SPICE failed and will be saved as null: %s", exc)
        logger.debug("SPICE traceback:\n%s", traceback.format_exc())
        return None


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def exact_normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def map_prediction_to_choice(prediction: str, choices: Dict[str, str]) -> str:
    text = str(prediction).strip()
    if not choices:
        return text
    short = text.strip().upper().strip(".:()[]{} ")
    if short in choices:
        return choices[short]

    patterns = [
        r"^\s*(?:answer\s*[:\-]?\s*)?\(?([A-Ea-e])\)?(?:[\.\):\s]|$)",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Ea-e])\)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            letter = match.group(1).upper()
            if letter in choices:
                return choices[letter]

    normalized_text = normalize_answer(text)
    for letter, choice_text in choices.items():
        normalized_choice = normalize_answer(choice_text)
        if normalized_text == normalized_choice:
            return choice_text
        if normalized_choice and normalized_choice in normalized_text:
            return choice_text
    return text


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def confusion_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labels = {"yes", "no", "true", "false"}
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total = 0
    for row in rows:
        gt = row.get("normalized_ground_truth", "")
        pred = row.get("normalized_prediction", "")
        if gt in labels:
            total += 1
            pred_label = pred if pred in labels else "other"
            matrix[gt][pred_label] += 1
    return {
        "labels": sorted(labels),
        "total_binary_samples": total,
        "matrix": {gt: dict(preds) for gt, preds in sorted(matrix.items())},
    }


def compute_text_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    metrics_config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics.update(corpus_bleu_scores(predictions, references) if metrics_config.get("bleu", True) else {})
    metrics.update(rouge_scores(predictions, references) if metrics_config.get("rouge", True) else {})
    metrics["METEOR"] = meteor_score_safe(predictions, references, logger) if metrics_config.get("meteor", True) else None
    metrics["BERTScore"] = (
        bertscore_safe(predictions, references, metrics_config, logger) if metrics_config.get("bertscore", True) else None
    )
    metrics["CIDEr"] = cider_safe(predictions, references, logger) if metrics_config.get("cider", True) else None
    metrics["SPICE"] = spice_safe(predictions, references, logger) if metrics_config.get("spice", True) else None
    return metrics


def compute_task_metrics(
    task: str,
    rows: Sequence[Dict[str, Any]],
    metrics_config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    predictions = [str(row.get("prediction", "")) for row in rows]
    references = [str(row.get("ground_truth", "")) for row in rows]
    metrics: Dict[str, Any] = {
        "num_successful_samples": len(rows),
    }
    if task == "vqa":
        exact_matches = [bool(row.get("exact_match", False)) for row in rows]
        normalized_matches = [bool(row.get("correct", False)) for row in rows]
        f1_scores = [float(row.get("token_f1", 0.0)) for row in rows]
        metrics["Exact Match Accuracy"] = float(statistics.mean(exact_matches)) if rows else None
        metrics["Normalized Accuracy"] = float(statistics.mean(normalized_matches)) if rows else None
        metrics["Token-level F1"] = float(statistics.mean(f1_scores)) if rows else None
        metrics["Confusion Summary"] = confusion_summary(rows)
    metrics.update(compute_text_metrics(predictions, references, metrics_config, logger))
    return metrics


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return float(ordered[index])


def estimate_flops(total_params: Optional[int], generated_tokens: int, successful_samples: int) -> Optional[Dict[str, Any]]:
    if not total_params or generated_tokens <= 0:
        return None
    total_flops = int(2 * total_params * generated_tokens)
    return {
        "method": "approx_generation_flops = 2 * total_parameters * generated_tokens",
        "approx_total_flops": total_flops,
        "approx_total_macs": int(total_flops / 2),
        "approx_flops_per_sample": float(total_flops / successful_samples) if successful_samples else None,
        "total_generated_tokens": generated_tokens,
    }


def build_benchmark(
    model_stats: Dict[str, Any],
    inference_times: Sequence[float],
    total_generated_tokens: int,
    successful_samples: int,
    failed_samples: int,
    requested_samples: int,
    overall_runtime: float,
    model_load_time: float,
) -> Dict[str, Any]:
    inference_total = float(sum(inference_times))
    params = {
        "total_params": model_stats.get("total_params"),
        "trainable_params": model_stats.get("trainable_params"),
        "non_trainable_params": model_stats.get("non_trainable_params"),
        "parameter_dtype": model_stats.get("parameter_dtype"),
    }
    flops = estimate_flops(model_stats.get("total_params"), total_generated_tokens, successful_samples)
    return {
        "parameters": params,
        "flops": flops,
        "runtime": {
            "requested_samples": requested_samples,
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "model_load_time_sec": model_load_time,
            "total_inference_time_sec": inference_total,
            "total_runtime_sec": overall_runtime,
            "avg_inference_latency_sec": float(statistics.mean(inference_times)) if inference_times else None,
            "p50_inference_latency_sec": percentile(inference_times, 0.50),
            "p95_inference_latency_sec": percentile(inference_times, 0.95),
            "throughput_samples_per_sec": successful_samples / inference_total if inference_total > 0 else None,
            "end_to_end_samples_per_sec": successful_samples / overall_runtime if overall_runtime > 0 else None,
            "total_generated_tokens": total_generated_tokens,
            "avg_generated_tokens": total_generated_tokens / successful_samples if successful_samples else None,
        },
        "gpu": {
            "gpu_name": model_stats.get("gpu_name"),
            "memory_allocated_bytes": model_stats.get("gpu_memory_allocated_bytes"),
            "memory_reserved_bytes": model_stats.get("gpu_memory_reserved_bytes"),
            "max_memory_allocated_bytes": model_stats.get("gpu_max_memory_allocated_bytes"),
        },
    }


def preview_text(row: Dict[str, Any], task: str) -> str:
    lines = ["=" * 50, f"SAMPLE_ID: {row.get('sample_id')}", "PROMPT:", str(row.get("prompt", ""))]
    lines.extend(["", "PRED:", str(row.get("prediction", "")), "", "GT:", str(row.get("ground_truth", ""))])
    if task == "vqa":
        lines.extend(["", f"CORRECT: {str(bool(row.get('correct', False))).lower()}"])
    lines.append("=" * 50)
    return "\n".join(lines)


def get_progress(iterable: Iterable[Any], total: int, desc: str, logger: logging.Logger) -> Iterable[Any]:
    if package_available("tqdm"):
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc)
    logger.warning("tqdm unavailable: progress bar disabled")
    return iterable


def evaluate_loop(
    task: str,
    samples: Sequence[EvalSample],
    bundle: ModelBundle,
    config: Dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[float], int, List[Dict[str, Any]]]:
    predictions_path = output_dir / "predictions.jsonl"
    errors_path = output_dir / "errors.jsonl"
    image_config = dict(config.get("image") or {})
    generation_kwargs = prepare_generation_config(config)
    preview_limit = int((config.get("logging") or {}).get("preview_samples", 3))
    save_errors = bool((config.get("logging") or {}).get("save_errors", True))

    logger.info("Generation config: %s", json.dumps(generation_kwargs, ensure_ascii=False))
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    previews: List[Dict[str, Any]] = []
    inference_times: List[float] = []
    total_generated_tokens = 0

    with predictions_path.open("w", encoding="utf-8") as prediction_handle, errors_path.open(
        "w", encoding="utf-8"
    ) as error_handle:
        iterator = get_progress(samples, total=len(samples), desc=f"Eval {task.upper()}", logger=logger)
        for sample in iterator:
            try:
                if sample.meta.get("ground_truth_error"):
                    raise FileNotFoundError(sample.meta["ground_truth_error"])
                if not sample.ground_truth:
                    raise ValueError("Ground truth is empty.")
                image = load_image_as_rgb(Path(sample.image_path), image_config)
                prediction, elapsed, generated_tokens = generate_prediction(bundle, image, sample.prompt, generation_kwargs)
                inference_times.append(elapsed)
                total_generated_tokens += generated_tokens

                base_row: Dict[str, Any] = {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "prompt": sample.prompt,
                    "prediction": prediction,
                    "ground_truth": sample.ground_truth,
                    "inference_time": elapsed,
                }
                if task == "cap":
                    base_row["split"] = sample.split
                else:
                    prediction_answer = map_prediction_to_choice(prediction, sample.choices)
                    normalized_prediction = normalize_answer(prediction_answer)
                    normalized_ground_truth = normalize_answer(sample.ground_truth)
                    exact_match = exact_normalize(prediction_answer) == exact_normalize(sample.ground_truth)
                    correct = normalized_prediction == normalized_ground_truth
                    base_row.update(
                        {
                            "question": sample.question,
                            "choices": sample.choices,
                            "answer_choice": sample.answer_choice,
                            "normalized_prediction": normalized_prediction,
                            "normalized_ground_truth": normalized_ground_truth,
                            "correct": correct,
                            "exact_match": exact_match,
                            "token_f1": token_f1(prediction_answer, sample.ground_truth),
                        }
                    )

                rows.append(base_row)
                append_jsonl(prediction_handle, base_row)
                if len(previews) < preview_limit:
                    previews.append(base_row)
                    logger.info("\n%s", preview_text(base_row, task))
            except Exception as exc:
                error_payload = {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "prompt": sample.prompt,
                    "ground_truth": sample.ground_truth,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                if task == "vqa":
                    error_payload["question"] = sample.question
                errors.append(error_payload)
                if save_errors:
                    append_jsonl(error_handle, error_payload)
                logger.error("Sample %s failed: %s", sample.sample_id, exc, exc_info=True)

    return rows, errors, inference_times, total_generated_tokens, previews


def write_previews(output_dir: Path, previews: Sequence[Dict[str, Any]], task: str) -> None:
    write_json(output_dir / "samples_preview.json", list(previews))
    text = "\n\n".join(preview_text(row, task) for row in previews)
    (output_dir / "samples_preview.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")


def log_metric_summary(metrics: Dict[str, Any], logger: logging.Logger) -> None:
    logger.info("Metric summary:")
    logger.info("%-32s %s", "metric", "value")
    logger.info("%-32s %s", "-" * 32, "-" * 16)
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (dict, list)):
                    continue
                logger.info("%-32s %s", f"{key}.{sub_key}", sub_value)
        else:
            logger.info("%-32s %s", key, value)


def build_output_dir(
    task: str,
    output_root: Path,
    split: str,
    sample_label: str,
    override: Optional[str] = None,
) -> Path:
    if override:
        path = Path(os.path.expandvars(os.path.expanduser(override)))
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if task == "cap":
        return output_root / f"EVAL_CAP_{split}_{sample_label}"
    return output_root / f"EVAL_VQA_{sample_label}"


def run(args: argparse.Namespace) -> int:
    overall_start = time.perf_counter()
    config_path = resolve_config_path(args.config, Path.cwd())
    if config_path is None:
        raise ValueError("--config is required")
    config_path = config_path.resolve()
    config = load_config(config_path)
    task = infer_task_name(args.task, config, config_path)
    split = str(args.split or config.get("split") or "test1k")
    sample_value = args.sample if args.sample is not None else config.get("sample", config.get("max_samples", "full"))
    max_samples, sample_label = normalize_sample_spec(sample_value)

    dataset_path = resolve_config_path(config.get("dataset_path"), config_path.parent)
    image_root = resolve_config_path(config.get("image_root"), config_path.parent)
    model_path = resolve_config_path(config.get("model_path"), config_path.parent)
    if dataset_path is None:
        raise ValueError("dataset_path is required in config.")
    if image_root is None:
        image_root = dataset_path.parent
    if model_path is None:
        raise ValueError("model_path is required in config.")
    output_root = resolve_output_root(config.get("output_root", "results"))
    output_dir = build_output_dir(task, output_root, split, sample_label, args.output_dir)
    logging_config = config.get("logging") or {}
    logger = setup_logging(output_dir, bool(logging_config.get("verbose", True)))

    runtime_config = dict(config)
    runtime_config.update(
        {
            "task": task,
            "config_path": str(config_path),
            "dataset_path": str(dataset_path),
            "image_root": str(image_root),
            "model_path": str(model_path),
            "output_dir": str(output_dir),
            "split": split,
            "sample": sample_label,
            "max_samples": max_samples,
            "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        }
    )
    save_yaml_or_json(output_dir / "run_config.yaml", runtime_config)

    package_status = get_package_availability()
    logger.info("Task name: %s", task)
    logger.info("Command: %s", runtime_config["command"])
    logger.info("Config path: %s", config_path)
    logger.info("Dataset path: %s", dataset_path)
    logger.info("Image root: %s", image_root)
    logger.info("Split: %s", split if task == "cap" else "N/A")
    logger.info("Sample count: %s", sample_label)
    logger.info("Result directory: %s", output_dir)
    logger.info("Metric packages: %s", json.dumps(package_status, ensure_ascii=False))

    if task == "cap":
        samples, schema_info = load_caption_samples(config, dataset_path, image_root, split, max_samples, logger)
    else:
        samples, schema_info = load_vqa_samples(config, dataset_path, image_root, max_samples, logger)
    logger.info("Resolved sample count: %d", len(samples))

    if args.dry_run:
        dry_benchmark = build_benchmark({}, [], 0, 0, 0, len(samples), time.perf_counter() - overall_start, 0.0)
        metrics = {
            "num_successful_samples": 0,
            "dry_run": True,
            "dataset_schema": schema_info,
            "Parameters": dry_benchmark["parameters"],
            "FLOPs": dry_benchmark["flops"],
            "Runtime": dry_benchmark["runtime"],
        }
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "benchmark.json", dry_benchmark)
        (output_dir / "predictions.jsonl").write_text("", encoding="utf-8")
        (output_dir / "errors.jsonl").write_text("", encoding="utf-8")
        write_previews(output_dir, [], task)
        logger.info("Dry run finished without loading model.")
        return 0

    model_load_start = time.perf_counter()
    bundle, model_stats = load_model_bundle(config, model_path, logger)
    model_load_time = time.perf_counter() - model_load_start

    rows, errors, inference_times, generated_tokens, previews = evaluate_loop(
        task=task,
        samples=samples,
        bundle=bundle,
        config=config,
        output_dir=output_dir,
        logger=logger,
    )
    write_previews(output_dir, previews, task)
    overall_runtime = time.perf_counter() - overall_start
    benchmark = build_benchmark(
        model_stats=model_stats,
        inference_times=inference_times,
        total_generated_tokens=generated_tokens,
        successful_samples=len(rows),
        failed_samples=len(errors),
        requested_samples=len(samples),
        overall_runtime=overall_runtime,
        model_load_time=model_load_time,
    )

    metrics_config = dict(config.get("metrics") or {})
    metrics = compute_task_metrics(task, rows, metrics_config, logger)
    metrics.update(
        {
            "task": task,
            "dataset_schema": schema_info,
            "num_requested_samples": len(samples),
            "num_failed_samples": len(errors),
            "Parameters": benchmark["parameters"],
            "FLOPs": benchmark["flops"],
            "Runtime": benchmark["runtime"],
        }
    )
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "benchmark.json", benchmark)
    logger.info("Saved predictions: %s", output_dir / "predictions.jsonl")
    logger.info("Saved metrics: %s", output_dir / "metrics.json")
    logger.info("Saved benchmark: %s", output_dir / "benchmark.json")
    logger.info("Saved errors: %s", output_dir / "errors.jsonl")
    logger.info("Total runtime: %.3f sec", overall_runtime)
    log_metric_summary(metrics, logger)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MedGemma on M3D-Cap and M3D-VQA.")
    parser.add_argument("--config", required=True, help="Path to CAP_task.yaml or VQA_task.yaml.")
    parser.add_argument("--task", choices=["cap", "caption", "captioning", "vqa"], default=None)
    parser.add_argument("--sample", default=None, help="Integer sample count or 'full'.")
    parser.add_argument("--split", default=None, help="Caption split, for example test1k.")
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/dataset and write skeleton outputs.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
