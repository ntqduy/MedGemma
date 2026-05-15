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
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CHOICE_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VQA_QUESTION_TYPE_NAMES = {
    "1": "Plane",
    "2": "Phase",
    "3": "Organ",
    "4": "Abnormality",
    "5": "Location",
}
VQA_EVAL_MODES = {"open", "closed"}

try:
    from src.dataset.prompt_templates import CAPTION_PROMPT_TEMPLATE, VQA_CLOSED_PROMPT_TEMPLATE, VQA_OPEN_PROMPT_TEMPLATE
except Exception:
    CAPTION_PROMPT_TEMPLATE = "<start_of_image> findings:"
    VQA_OPEN_PROMPT_TEMPLATE = "<start_of_image> {question}"
    VQA_CLOSED_PROMPT_TEMPLATE = "<start_of_image> {question} {choices_inline}"


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


@dataclass
class SliceInferenceConfig:
    num_slices: Any = 1
    slice_strategy: str = "middle"
    view: str = "axial"
    inference_mode: str = "montage"


@dataclass
class SliceImageBundle:
    images: List[Any]
    selected_slice_indices: List[int]
    effective_num_slices: int
    prompt: str
    montage_path: Optional[str] = None
    per_slice_prompts: List[str] = field(default_factory=list)


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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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
    prompt_template = str(config.get("prompt_template") or CAPTION_PROMPT_TEMPLATE)
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
        "question_type": configured.get("question_type") or detect_field(fieldnames, ("Question Type", "question_type", "type")),
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


def render_choices_inline(choices: Dict[str, str]) -> str:
    parts = [f"{letter}. {text}" for letter, text in choices.items() if text]
    if not parts:
        return ""
    return "Choices: " + " ".join(parts)


def select_vqa_prompt_template(config: Dict[str, Any], vqa_eval_mode: str) -> str:
    prompt_templates = config.get("prompt_templates")
    if isinstance(prompt_templates, dict):
        mode_template = prompt_templates.get(vqa_eval_mode)
        if mode_template is not None and str(mode_template).strip():
            return str(mode_template)

    mode_key = f"{vqa_eval_mode}_prompt_template"
    if config.get(mode_key) is not None and str(config.get(mode_key)).strip():
        return str(config.get(mode_key))

    if config.get("prompt_template") is not None and str(config.get("prompt_template")).strip():
        return str(config.get("prompt_template"))

    if vqa_eval_mode == "open":
        return VQA_OPEN_PROMPT_TEMPLATE
    return VQA_CLOSED_PROMPT_TEMPLATE


def build_vqa_prompt(prompt_template: str, question: str, choices: Dict[str, str]) -> str:
    choices_text = render_choices(choices)
    choices_inline = render_choices_inline(choices)
    return prompt_template.format(
        question=question,
        choices=choices_text,
        choices_inline=choices_inline,
        answer_options=choices_text,
        answer_options_inline=choices_inline,
    ).strip()


def normalize_question_type(value: str, choices: Dict[str, str], answer: str) -> str:
    raw = normalize_answer(value)
    answer_norm = normalize_answer(answer)
    if "yes" in raw or "no" in raw or "binary" in raw or answer_norm in {"yes", "no"}:
        return "yes_no"
    if "open" in raw or "free" in raw:
        return "open"
    if "choice" in raw or "closed" in raw or "option" in raw or choices:
        return "multiple_choice"
    return "open"


def normalize_vqa_eval_mode(value: Any) -> str:
    if isinstance(value, bool):
        return "closed" if value else "open"
    raw = str(value or "open").strip().lower().replace("-", "_")
    if raw in {"close", "closed", "close_ended", "closeended", "multiple_choice", "mcq", "choice"}:
        return "closed"
    if raw in {"open", "open_ended", "openended", "free", "free_form", "freeform"}:
        return "open"
    raise ValueError(f"Unsupported VQA eval mode {value!r}. Use 'open' or 'closed'.")


def question_type_id_and_name(value: Any) -> Tuple[str, str]:
    raw = str(value or "").strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    normalized = normalize_answer(raw)
    if normalized.isdigit():
        type_id = normalized
    else:
        type_id = raw
    return type_id, VQA_QUESTION_TYPE_NAMES.get(type_id, raw or "Unknown")


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
    vqa_eval_mode = normalize_vqa_eval_mode(config.get("vqa_eval_mode", config.get("close_ended", "open")))
    prompt_template = select_vqa_prompt_template(config, vqa_eval_mode)
    roots = [image_root, dataset_path.parent, dataset_path.parent.parent]
    samples: List[EvalSample] = []

    for index, row in enumerate(selected_rows):
        image_value = str(row.get(schema["image"] or "", "")).strip()
        image_path = resolve_relative_existing(image_value, roots)
        question = str(row.get(schema["question"] or "", "")).strip()
        question_type_raw = str(row.get(schema["question_type"] or "", "")).strip() if schema.get("question_type") else ""
        question_type_id, question_type_name = question_type_id_and_name(question_type_raw)
        answer = str(row.get(schema["answer"] or "", "")).strip()
        answer_choice = str(row.get(schema["answer_choice"] or "", "")).strip() if schema.get("answer_choice") else ""
        all_choices = {}
        for letter in CHOICE_LETTERS:
            column = schema.get(f"choice_{letter.lower()}")
            if column:
                value = str(row.get(column, "")).strip()
                if value:
                    all_choices[letter] = value
        for column_name, raw_value in row.items():
            match = re.match(r"^\s*(?:choice|option)\s+(.+?)\s*$", str(column_name), flags=re.IGNORECASE)
            if not match:
                continue
            label = match.group(1).strip().upper()
            value = str(raw_value or "").strip()
            if label and value and label not in all_choices:
                all_choices[label] = value
        choices = dict(all_choices) if vqa_eval_mode == "closed" else {}
        answer_type = "closed" if choices else normalize_question_type("", choices, answer)
        if not answer and answer_choice in all_choices:
            answer = all_choices[answer_choice]
        answer_with_choice = ""
        if choices and answer_choice:
            answer_text = str(answer or "").strip()
            answer_with_choice = (
                answer_text
                if re.match(rf"^{re.escape(answer_choice)}\s*\.", answer_text, flags=re.IGNORECASE)
                else f"{answer_choice}. {answer_text}".strip()
            )
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
                meta={
                    "raw_image": image_value,
                    "row_index": index,
                    "question_type": question_type_id,
                    "question_type_id": question_type_id,
                    "question_type_name": question_type_name,
                    "answer_type": answer_type,
                    "vqa_eval_mode": vqa_eval_mode,
                    "all_choices": all_choices,
                    "ground_truth_with_choice": answer_with_choice,
                },
            )
        )

    schema_info = {
        "schema": schema,
        "fieldnames": fieldnames,
        "total_rows": len(rows),
        "selected_rows": len(samples),
        "vqa_eval_mode": vqa_eval_mode,
        "prompt_template": prompt_template,
    }
    logger.info("VQA schema: %s", json.dumps(schema_info, ensure_ascii=False))
    return samples, schema_info


def log_sample_image_path_health(samples: Sequence[EvalSample], logger: logging.Logger, max_examples: int = 5) -> None:
    if not samples:
        logger.warning("Image path preflight: no samples were loaded.")
        return

    missing = [sample.image_path for sample in samples if not Path(sample.image_path).exists()]
    existing_count = len(samples) - len(missing)
    if not missing:
        logger.info("Image path preflight: %d/%d image files found.", existing_count, len(samples))
        return

    logger.warning(
        "Image path preflight: %d/%d image files found, %d missing.",
        existing_count,
        len(samples),
        len(missing),
    )
    for image_path in missing[:max_examples]:
        logger.warning("Missing image example: %s", image_path)


def log_vqa_mode_constraints(samples: Sequence[EvalSample], logger: logging.Logger) -> None:
    if not samples:
        return

    mode = str(samples[0].meta.get("vqa_eval_mode") or "").strip().lower()
    samples_with_choices = sum(1 for sample in samples if bool(sample.choices))
    prompts_with_choices = sum(1 for sample in samples if "Choices:" in sample.prompt)
    prompts_with_answer_cue = sum(1 for sample in samples if "Answer:" in sample.prompt)
    logger.info(
        "VQA mode constraints: mode=%s samples=%d with_choices=%d prompts_with_choices=%d prompts_with_answer_cue=%d",
        mode or "unknown",
        len(samples),
        samples_with_choices,
        prompts_with_choices,
        prompts_with_answer_cue,
    )
    if mode == "open":
        if samples_with_choices or prompts_with_choices:
            logger.warning("Open VQA constraint violation: choices must be hidden for all samples.")
        if prompts_with_answer_cue:
            logger.warning("Open VQA differs from Med3DVLM prompt: found 'Answer:' cue in %d prompt(s).", prompts_with_answer_cue)
    elif mode == "closed":
        missing_choices = len(samples) - samples_with_choices
        if missing_choices:
            logger.warning("Closed VQA constraint warning: %d/%d samples do not have answer choices.", missing_choices, len(samples))


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


def load_volume_array(path: Path) -> Any:
    lowered = str(path).lower()
    if lowered.endswith(".npy"):
        if not package_available("numpy"):
            raise ImportError("numpy is required to read .npy images.")
        import numpy as np  # type: ignore

        return np.load(path)
    if lowered.endswith(".nii") or lowered.endswith(".nii.gz"):
        if not package_available("nibabel"):
            raise ImportError("nibabel is required to read NIfTI images.")
        import nibabel as nib  # type: ignore

        return nib.load(str(path)).get_fdata()
    return None


def squeeze_volume_array(array: Any) -> Any:
    import numpy as np  # type: ignore

    arr = np.asarray(array)
    arr = np.squeeze(arr)
    while arr.ndim > 3:
        arr = np.take(arr, arr.shape[0] // 2, axis=0)
        arr = np.squeeze(arr)
    return arr


def view_axis_for_volume(view: str) -> int:
    return {"axial": 0, "coronal": 1, "sagittal": 2}[view]


def is_auto_num_slices(value: Any) -> bool:
    return str(value).strip().lower() in {"auto", "max", "all"}


def slice_window(depth: int) -> Tuple[int, int]:
    margin = int(math.floor(depth * 0.10))
    start = margin
    end = depth - margin - 1
    if end < start:
        return 0, depth - 1
    return start, end


def max_rule_based_slices(depth: int) -> int:
    if depth <= 0:
        return 0
    start, end = slice_window(depth)
    return max(end - start + 1, 1)


def resolve_effective_num_slices(depth: int, requested_num_slices: Any, slice_strategy: str) -> int:
    if is_auto_num_slices(requested_num_slices):
        return max_rule_based_slices(depth) if slice_strategy == "uniform" else 1
    return int(requested_num_slices)


def select_slice_indices(depth: int, num_slices: int, slice_strategy: str) -> List[int]:
    if depth <= 0:
        raise ValueError("Cannot select slices from an empty volume axis.")
    if num_slices <= 1 or slice_strategy == "middle":
        return [depth // 2]

    if slice_strategy == "center_uniform":
        target_count = max(1, min(int(num_slices), depth))
        center = depth // 2
        start = max(0, center - (target_count // 2))
        end = min(depth, start + target_count)
        start = max(0, end - target_count)
        return list(range(start, end))

    start, end = slice_window(depth)
    valid_count = max(end - start + 1, 1)
    target_count = max(1, min(int(num_slices), valid_count))
    if target_count == 1:
        return [(start + end) // 2]
    step = (end - start) / max(target_count - 1, 1)
    return [max(0, min(depth - 1, int(round(start + step * idx)))) for idx in range(target_count)]


def extract_volume_slices(path: Path, slice_config: SliceInferenceConfig) -> Tuple[List[Any], List[int], int]:
    volume = load_volume_array(path)
    if volume is None:
        requested = 1 if is_auto_num_slices(slice_config.num_slices) else int(slice_config.num_slices)
        if requested > 1:
            raise ValueError(f"Cannot select multiple slices from a non-volume image: {path}")
        image = load_image_as_rgb(path, {})
        return [image], [0], 1

    arr = squeeze_volume_array(volume)
    if arr.ndim == 2:
        return [arr], [0], 1
    if arr.ndim != 3:
        raise ValueError(f"Unsupported volume shape after squeeze: {arr.shape}")

    axis = view_axis_for_volume(slice_config.view)
    effective_num_slices = resolve_effective_num_slices(
        arr.shape[axis],
        slice_config.num_slices,
        slice_config.slice_strategy,
    )
    indices = select_slice_indices(arr.shape[axis], effective_num_slices, slice_config.slice_strategy)
    slices = [arr.take(index, axis=axis) for index in indices]
    return slices, indices, len(indices)


def slice_to_rgb_image(slice_array: Any) -> Any:
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    arr = np.asarray(slice_array)
    arr = np.squeeze(arr)
    if arr.ndim > 2:
        while arr.ndim > 2:
            arr = np.take(arr, arr.shape[0] // 2, axis=0)
            arr = np.squeeze(arr)
    arr = normalize_array_to_uint8(arr)
    return Image.fromarray(arr, mode="L").convert("RGB")


def montage_grid_shape(num_slices: int) -> Tuple[int, int]:
    cols = int(math.ceil(math.sqrt(num_slices)))
    rows = int(math.ceil(num_slices / max(cols, 1)))
    return rows, cols


def parse_target_size(value: Any) -> Optional[Tuple[int, int]]:
    if value in (None, "", "auto"):
        return None
    if isinstance(value, int):
        return value, value
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[x,]", value) if part.strip()]
        if len(parts) == 1:
            size = int(parts[0])
            return size, size
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    if isinstance(value, (list, tuple)) and value:
        if len(value) == 1:
            size = int(value[0])
            return size, size
        return int(value[0]), int(value[1])
    if isinstance(value, dict):
        width = value.get("width") or value.get("w")
        height = value.get("height") or value.get("h")
        if width and height:
            return int(width), int(height)
    return None


def processor_target_size(processor: Any, image_config: Dict[str, Any]) -> Tuple[int, int]:
    configured = parse_target_size(image_config.get("montage_size") or image_config.get("target_size"))
    if configured:
        return configured

    image_processor = getattr(processor, "image_processor", processor)
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        width = size.get("width")
        height = size.get("height")
        if width and height:
            return int(width), int(height)
        shortest = size.get("shortest_edge")
        if shortest:
            edge = int(shortest)
            return edge, edge
        longest = size.get("longest_edge")
        if longest:
            edge = int(longest)
            return edge, edge
    if isinstance(size, int):
        return size, size
    return 896, 896


def resize_image(image: Any, target_size: Tuple[int, int]) -> Any:
    from PIL import Image  # type: ignore

    resampling = getattr(Image, "Resampling", Image).BICUBIC
    return image.resize(target_size, resampling)


def build_montage_image(slice_images: Sequence[Any], target_size: Tuple[int, int]) -> Any:
    from PIL import Image  # type: ignore

    if not slice_images:
        raise ValueError("Cannot build montage without slices.")
    rows, cols = montage_grid_shape(len(slice_images))
    target_width, target_height = target_size
    tile_width = max(target_width // max(cols, 1), 1)
    tile_height = max(target_height // max(rows, 1), 1)
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    canvas = Image.new("RGB", target_size, color=(0, 0, 0))
    for idx, image in enumerate(slice_images):
        row = idx // cols
        col = idx % cols
        tile = image.convert("RGB").resize((tile_width, tile_height), resampling)
        canvas.paste(tile, (col * tile_width, row * tile_height))
    return canvas


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "sample"


def with_image_token(prompt: str) -> str:
    prompt = str(prompt).strip()
    if "<start_of_image>" in prompt:
        return prompt
    return f"<start_of_image> {prompt}"


def without_image_token(prompt: str) -> str:
    return str(prompt).replace("<start_of_image>", "", 1).strip()


def vqa_question_block(sample: EvalSample) -> str:
    if sample.prompt:
        return without_image_token(sample.prompt)
    lines = [f"Question: {sample.question}"]
    choices = render_choices(sample.choices)
    question_type = str(sample.meta.get("question_type", "")).strip().lower()
    if choices and question_type not in {"yes_no", "open"}:
        choice_parts = [f"{letter}. {text}" for letter, text in sample.choices.items() if text]
        if choice_parts:
            lines.append("Choices: " + " ".join(choice_parts))
        else:
            lines.append(choices)
    elif question_type == "yes_no" or normalize_answer(sample.ground_truth) in {"yes", "no"}:
        lines.append("Return only yes or no. Do not explain.")
    else:
        lines.append("Answer concisely using only visible findings.")
    return "\n".join(lines)


def build_montage_prompt(
    task: str,
    sample: EvalSample,
    slice_config: SliceInferenceConfig,
    num_slices: int,
) -> str:
    if task == "vqa":
        prompt_body = vqa_question_block(sample)
        return with_image_token(
            f"This image is a montage of {num_slices} {slice_config.view} slices sampled from a 3D "
            "medical volume, ordered from first to last slice.\n"
            f"{prompt_body}"
        )
    return with_image_token(
        f"This image is a montage of {num_slices} {slice_config.view} slices sampled from a 3D "
        "medical volume, ordered from first to last slice. Generate a concise radiology-style caption "
        "describing only visible imaging findings. Do not infer patient history or findings not visible "
        "in the image."
    )


def build_independent_prompt(
    task: str,
    sample: EvalSample,
    slice_config: SliceInferenceConfig,
    ordinal: int,
    slice_index: int,
    num_slices: int,
) -> str:
    prefix = (
        f"This image shows {num_slices} {slice_config.view} slice(s) sampled from a 3D medical volume. "
        f"This is slice {ordinal + 1} of {num_slices}, selected by a fixed rule at slice index {slice_index}."
    )
    if task == "vqa":
        prompt_body = vqa_question_block(sample)
        return with_image_token(
            f"{prefix}\n{prompt_body}"
        )
    return with_image_token(
        f"{prefix} Generate a concise radiology-style caption describing only visible imaging findings. "
        "Do not infer patient history or findings not visible in the image."
    )


def prepare_slice_images(
    sample: EvalSample,
    task: str,
    slice_config: SliceInferenceConfig,
    processor: Any,
    image_config: Dict[str, Any],
    output_dir: Path,
) -> SliceImageBundle:
    raw_slices, selected_indices, effective_num_slices = extract_volume_slices(Path(sample.image_path), slice_config)
    slice_images = [slice_to_rgb_image(slice_array) if not hasattr(slice_array, "convert") else slice_array.convert("RGB") for slice_array in raw_slices]
    target_size = processor_target_size(processor, image_config)
    prompt = build_montage_prompt(task, sample, slice_config, effective_num_slices)

    if slice_config.inference_mode == "independent":
        resized = [resize_image(image, target_size) for image in slice_images]
        per_slice_prompts = [
            build_independent_prompt(task, sample, slice_config, idx, selected_indices[idx], effective_num_slices)
            for idx in range(len(selected_indices))
        ]
        return SliceImageBundle(
            images=resized,
            selected_slice_indices=selected_indices,
            effective_num_slices=effective_num_slices,
            prompt=prompt,
            per_slice_prompts=per_slice_prompts,
        )

    montage = build_montage_image(slice_images, target_size)
    montage_dir = output_dir / "montages"
    montage_dir.mkdir(parents=True, exist_ok=True)
    montage_path = montage_dir / f"{safe_filename(sample.sample_id)}_{slice_config.view}_{effective_num_slices}.png"
    montage.save(montage_path)
    return SliceImageBundle(
        images=[montage],
        selected_slice_indices=selected_indices,
        effective_num_slices=effective_num_slices,
        prompt=prompt,
        montage_path=str(montage_path),
    )


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


def apply_cuda_visible_devices_config(config: Dict[str, Any], logger: logging.Logger) -> None:
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    configured = config.get("cuda_visible_devices", config.get("gpu_ids"))
    if env_value:
        logger.info("CUDA_VISIBLE_DEVICES already set by environment: %s", env_value)
        return
    if configured in (None, "", "auto"):
        logger.info("CUDA_VISIBLE_DEVICES controlled by system/default environment")
        return
    if isinstance(configured, (list, tuple)):
        value = ",".join(str(item).strip() for item in configured if str(item).strip())
    else:
        value = str(configured).strip()
    if not value:
        logger.info("CUDA_VISIBLE_DEVICES controlled by system/default environment")
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    logger.info("Set CUDA_VISIBLE_DEVICES from config: %s", value)


def parse_cuda_device_list(config: Dict[str, Any]) -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES") or config.get("cuda_visible_devices", config.get("gpu_ids"))
    if raw in (None, "", "auto"):
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def should_run_parallel(config: Dict[str, Any], args: argparse.Namespace, samples: Sequence[Any]) -> bool:
    if getattr(args, "no_parallel", False):
        return False
    if getattr(args, "num_shards", 1) and int(getattr(args, "num_shards", 1)) > 1:
        return False
    if len(samples) <= 1:
        return False
    return bool_from_config(config.get("parallel_eval"), False) and len(parse_cuda_device_list(config)) > 1


def apply_sample_shard(samples: Sequence[EvalSample], shard_index: int, num_shards: int) -> List[EvalSample]:
    if num_shards <= 1:
        return list(samples)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"Invalid shard index {shard_index}; expected 0 <= index < {num_shards}")
    return [sample for index, sample in enumerate(samples) if index % num_shards == shard_index]


def bool_from_config(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def get_processor_use_fast(config: Dict[str, Any]) -> bool:
    processor_config = dict(config.get("processor") or {})
    return bool_from_config(processor_config.get("use_fast", config.get("processor_use_fast")), False)


def build_slice_inference_config(config: Dict[str, Any], args: argparse.Namespace) -> SliceInferenceConfig:
    configured = dict(config.get("slice_inference") or {})
    raw_num_slices = (
        args.num_slices
        if getattr(args, "num_slices", None) is not None
        else configured.get("num_slices", configured.get("num_slice", 1))
    )
    if is_auto_num_slices(raw_num_slices):
        num_slices: Any = "auto"
    else:
        try:
            num_slices = int(raw_num_slices)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid num_slices: {raw_num_slices!r}") from exc
        if num_slices < 1:
            raise ValueError("--num_slices must be >= 1 or one of: auto, max, all")

    requested_strategy = str(
        args.slice_strategy if getattr(args, "slice_strategy", None) is not None else configured.get("slice_strategy", "middle")
    ).strip().lower()
    if requested_strategy not in {"middle", "uniform", "center_uniform"}:
        raise ValueError("--slice_strategy must be one of: middle, uniform, center_uniform")
    if num_slices == 1:
        slice_strategy = "middle"
    elif requested_strategy == "middle":
        raise ValueError("--slice_strategy middle requires --num_slices 1. Use center_uniform for central multi-slice runs.")
    else:
        slice_strategy = requested_strategy

    view = str(args.view if getattr(args, "view", None) is not None else configured.get("view", "axial")).strip().lower()
    if view not in {"axial", "sagittal", "coronal"}:
        raise ValueError("--view must be one of: axial, sagittal, coronal")

    inference_mode = str(
        args.inference_mode
        if getattr(args, "inference_mode", None) is not None
        else configured.get("inference_mode", "montage")
    ).strip().lower()
    if inference_mode not in {"montage", "independent"}:
        raise ValueError("--inference_mode must be one of: montage, independent")

    return SliceInferenceConfig(
        num_slices=num_slices,
        slice_strategy=slice_strategy,
        view=view,
        inference_mode=inference_mode,
    )


def slice_inference_config_to_dict(slice_config: SliceInferenceConfig) -> Dict[str, Any]:
    return {
        "num_slices": slice_config.num_slices,
        "slice_strategy": slice_config.slice_strategy,
        "view": slice_config.view,
        "inference_mode": slice_config.inference_mode,
    }


def validate_local_model_path(model_path: Path, local_files_only: bool = True) -> None:
    if not local_files_only:
        return
    if not model_path.exists():
        raise FileNotFoundError(
            "Local model_path does not exist: "
            f"{model_path}\n"
            "Check config model_path, the current project directory, or download/copy the model weights there."
        )
    if not model_path.is_dir():
        raise NotADirectoryError(f"model_path must be a directory, got: {model_path}")
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"Model directory exists but is missing config.json: {model_path}\n"
            "This folder does not look like a complete Hugging Face model directory."
        )

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            weight_files = sorted(set((index_data.get("weight_map") or {}).values()))
        except Exception:
            weight_files = []
        missing = [name for name in weight_files if not (model_path / name).exists()]
        if missing:
            missing_preview = "\n".join(f"- {name}" for name in missing[:20])
            raise FileNotFoundError(
                f"Model directory is missing safetensors shard files listed in {index_path}:\n"
                f"{missing_preview}"
            )
        return

    has_weight_file = any(model_path.glob("*.safetensors")) or any(model_path.glob("pytorch_model*.bin"))
    if not has_weight_file:
        raise FileNotFoundError(
            f"Model directory has no .safetensors or pytorch_model*.bin weight files: {model_path}"
        )


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
        validate_local_model_path(model_path, local_files_only)
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logger.info("CUDA_VISIBLE_DEVICES: %s", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
        logger.info("CUDA device count visible to process: %s", cuda_count)

        processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=local_files_only,
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            use_fast=get_processor_use_fast(config),
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
            "dtype": dtype,
            "local_files_only": local_files_only,
            "trust_remote_code": bool(config.get("trust_remote_code", True)),
        }
        device_map = config.get("device_map", "auto")
        if device_name != "cpu" and device_map:
            model_kwargs["device_map"] = device_map
        logger.info("Requested device: %s", device_name)
        logger.info("Requested device_map: %s", device_map)

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


def first_token_id(*values: Any) -> Optional[Any]:
    for value in values:
        if value is not None:
            return value
    return None


def add_generation_token_ids(generation_kwargs: Dict[str, Any], bundle: ModelBundle) -> Dict[str, Any]:
    kwargs = dict(generation_kwargs)
    tokenizer = bundle.tokenizer
    generation_config = getattr(bundle.model, "generation_config", None)
    model_config = getattr(bundle.model, "config", None)

    eos_token_id = first_token_id(
        getattr(tokenizer, "eos_token_id", None),
        getattr(generation_config, "eos_token_id", None),
        getattr(model_config, "eos_token_id", None),
    )
    pad_token_id = first_token_id(
        getattr(tokenizer, "pad_token_id", None),
        getattr(generation_config, "pad_token_id", None),
        getattr(model_config, "pad_token_id", None),
        eos_token_id,
    )

    if "eos_token_id" not in kwargs and eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if "pad_token_id" not in kwargs and pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def clean_generated_text(text: str) -> str:
    text = str(text or "").replace("\r", "\n").strip()
    if "<end_of_turn>" in text:
        text = text.split("<end_of_turn>", 1)[0]
    text = re.sub(r"</?(?:s|pad|bos|eos)>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<(?:start|end)_of_turn>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```(?:\w+)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text.strip()


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
    generation_kwargs = add_generation_token_ids(generation_kwargs, bundle)
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
    return clean_generated_text(text), elapsed, generated_tokens


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


_MED3DVLM_EVAL_BUNDLE: Optional[Dict[str, Any]] = None


def med3dvlm_eval_bundle(logger: logging.Logger) -> Optional[Dict[str, Any]]:
    global _MED3DVLM_EVAL_BUNDLE
    if _MED3DVLM_EVAL_BUNDLE is not None:
        return _MED3DVLM_EVAL_BUNDLE
    if not package_available("evaluate"):
        logger.warning("Med3DVLM-style metrics unavailable: evaluate is not installed")
        _MED3DVLM_EVAL_BUNDLE = None
        return None
    try:
        import evaluate  # type: ignore

        module_path = getattr(evaluate, "__file__", "")
        if module_path:
            resolved = Path(module_path).resolve()
            if PROJECT_ROOT in resolved.parents:
                logger.warning(
                    "Med3DVLM-style metrics unavailable: 'evaluate' resolved to local module %s. "
                    "Rename local evaluate_cli.py or adjust PYTHONPATH.",
                    resolved,
                )
                _MED3DVLM_EVAL_BUNDLE = None
                return None
        if not hasattr(evaluate, "load"):
            logger.warning(
                "Med3DVLM-style metrics unavailable: 'evaluate' has no load(). "
                "Install/upgrade evaluate>=0.4.2."
            )
            _MED3DVLM_EVAL_BUNDLE = None
            return None

        _MED3DVLM_EVAL_BUNDLE = {
            "bleu": evaluate.load("bleu"),
            "rouge": evaluate.load("rouge"),
            "meteor": evaluate.load("meteor"),
            "bertscore": evaluate.load("bertscore"),
        }
        return _MED3DVLM_EVAL_BUNDLE
    except Exception as exc:
        logger.warning("Med3DVLM-style metrics failed to load: %s", exc)
        logger.debug("Med3DVLM-style metrics traceback:\n%s", traceback.format_exc())
        _MED3DVLM_EVAL_BUNDLE = None
        return None


def resolve_vqa_open_eval_style(metrics_config: Dict[str, Any]) -> str:
    raw = str(metrics_config.get("vqa_open_eval_style", "legacy")).strip().lower()
    if raw in {"med3dvlm", "med3dvlm_style", "med3d"}:
        return "med3dvlm"
    if raw in {"legacy", "default", "medgemma"}:
        return "legacy"
    return "legacy"


def build_med3dvlm_bertscore_kwargs(
    metrics_config: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    if not bool_setting(metrics_config.get("bertscore"), True):
        return None
    kwargs: Dict[str, Any] = {
        "lang": str(metrics_config.get("bertscore_lang", "en")),
        "rescale_with_baseline": bool(metrics_config.get("bertscore_rescale_with_baseline", False)),
    }
    if metrics_config.get("bertscore_model_type"):
        resolved_model_type = resolve_bertscore_model_type(
            metrics_config.get("bertscore_model_type"),
            logger,
            bool_setting(metrics_config.get("bertscore_local_only"), False),
        )
        if resolved_model_type == "__missing_local_bertscore_model__":
            return None
        if resolved_model_type:
            kwargs["model_type"] = resolved_model_type
    if metrics_config.get("bertscore_num_layers") is not None:
        kwargs["num_layers"] = int(metrics_config.get("bertscore_num_layers"))
    if metrics_config.get("bertscore_device"):
        kwargs["device"] = str(metrics_config.get("bertscore_device"))
    return kwargs


def med3dvlm_row_metrics(
    prediction: str,
    reference: str,
    bundle: Dict[str, Any],
    metrics_config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Optional[float]]:
    decoded_preds = [str(prediction).strip()]
    decoded_labels = [[str(reference).strip()]]

    bleu_score = bundle["bleu"].compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
    rouge_score = bundle["rouge"].compute(
        predictions=decoded_preds,
        references=decoded_labels,
        rouge_types=["rouge1"],
    )
    meteor_score = bundle["meteor"].compute(predictions=decoded_preds, references=decoded_labels)
    f1_mean = None
    bertscore_kwargs = build_med3dvlm_bertscore_kwargs(metrics_config, logger)
    if bertscore_kwargs is not None:
        bert_score = bundle["bertscore"].compute(
            predictions=decoded_preds,
            references=decoded_labels,
            **bertscore_kwargs,
        )
        f1_values = bert_score.get("f1") or []
        f1_mean = float(sum(f1_values) / len(f1_values)) if f1_values else None

    return {
        "bleu": float(bleu_score.get("bleu", 0.0)),
        "rouge1": float(rouge_score.get("rouge1", 0.0)),
        "meteor": float(meteor_score.get("meteor", 0.0)),
        "bert_f1": f1_mean,
    }


def compute_text_metrics_med3dvlm(
    predictions: Sequence[str],
    references: Sequence[str],
    metrics_config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    bundle = med3dvlm_eval_bundle(logger)
    if bundle is None:
        return {}
    decoded_preds = [str(text).strip() for text in predictions]
    decoded_labels = [[str(text).strip()] for text in references]

    bleu_score = bundle["bleu"].compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
    rouge_score = bundle["rouge"].compute(
        predictions=decoded_preds,
        references=decoded_labels,
        rouge_types=["rouge1"],
    )
    meteor_score = bundle["meteor"].compute(predictions=decoded_preds, references=decoded_labels)
    f1_values: List[float] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    bertscore_kwargs = build_med3dvlm_bertscore_kwargs(metrics_config, logger)
    if bertscore_kwargs is not None:
        bert_score = bundle["bertscore"].compute(
            predictions=decoded_preds,
            references=decoded_labels,
            **bertscore_kwargs,
        )
        f1_values = bert_score.get("f1") or []
        precision_values = bert_score.get("precision") or []
        recall_values = bert_score.get("recall") or []

    metrics: Dict[str, Any] = {
        "BLEU-1": float(bleu_score.get("bleu", 0.0)),
        "ROUGE-1": float(rouge_score.get("rouge1", 0.0)),
        "METEOR": float(meteor_score.get("meteor", 0.0)),
    }
    if f1_values:
        metrics["BERTScore"] = {
            "Precision": float(sum(precision_values) / len(precision_values)) if precision_values else None,
            "Recall": float(sum(recall_values) / len(recall_values)) if recall_values else None,
            "F1": float(sum(f1_values) / len(f1_values)),
        }
    return metrics


def compute_vqa_med3dvlm_metrics(
    rows: Sequence[Dict[str, Any]],
    metrics_config: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    open_rows = [row for row in rows if is_open_vqa_row(row)]
    closed_rows = [row for row in rows if is_closed_vqa_row(row)]
    metrics: Dict[str, Any] = {
        "samples": {
            "total": len(rows),
            "open": len(open_rows),
            "closed": len(closed_rows),
        },
        "closed_accuracy": accuracy_value(closed_rows, "correct"),
        "open_text_metrics": None,
    }
    if open_rows:
        predictions = [str(row.get("prediction", "")) for row in open_rows]
        references = [str(row.get("ground_truth", "")) for row in open_rows]
        metrics["open_text_metrics"] = compute_text_metrics_med3dvlm(predictions, references, metrics_config, logger)
    return metrics


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

        def resolve_model_type(value: Any) -> Optional[str]:
            raw = str(value or "").strip()
            if not raw or raw.lower() in {"none", "null"}:
                return None
            expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
            candidates = [expanded] if expanded.is_absolute() else [(PROJECT_ROOT / expanded), (Path.cwd() / expanded)]
            for candidate in candidates:
                if candidate.exists():
                    return str(candidate.resolve())
            looks_local = any(sep in raw for sep in ("/", "\\")) or raw.startswith((".", "~")) or ":" in raw
            if looks_local:
                logger.warning(
                    "BERTScore local model path does not exist: %s. "
                    "Download/copy roberta-large there or set metrics.bertscore_model_type to the correct local path.",
                    str(candidates[0].resolve()),
                )
                return "__missing_local_bertscore_model__"
            return raw

        kwargs: Dict[str, Any] = {
            "lang": str(metrics_config.get("bertscore_lang", "en")),
            "verbose": bool(metrics_config.get("bertscore_verbose", False)),
            "rescale_with_baseline": bool(metrics_config.get("bertscore_rescale_with_baseline", False)),
        }
        if metrics_config.get("bertscore_model_type"):
            resolved_model_type = resolve_model_type(metrics_config.get("bertscore_model_type"))
            if resolved_model_type == "__missing_local_bertscore_model__":
                return None
            if resolved_model_type:
                kwargs["model_type"] = resolved_model_type
        if metrics_config.get("bertscore_num_layers") is not None:
            kwargs["num_layers"] = int(metrics_config.get("bertscore_num_layers"))
        elif kwargs.get("model_type") and Path(str(kwargs["model_type"])).exists():
            model_path_name = Path(str(kwargs["model_type"])).name.lower()
            if model_path_name == "roberta-large":
                kwargs["num_layers"] = 17
                logger.info("BERTScore local roberta-large detected; using num_layers=17")
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
        logger.warning("CIDEr unavailable: official pycocoevalcap is not installed")
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


def choice_match_tokens(text: str) -> List[str]:
    tokens = normalize_answer(text).split()
    normalized = []
    for token in tokens:
        if len(token) > 3 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and re.search(r"(ses|xes|zes|ches|shes)$", token):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        normalized.append(token)
    return normalized


def exact_normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def choice_label_regex(choices: Dict[str, str]) -> str:
    labels = sorted((re.escape(str(label)) for label in choices), key=len, reverse=True)
    return "|".join(labels) or r"[A-Z]"


def choice_token_score(prediction: str, choice_text: str) -> float:
    text_tokens = choice_match_tokens(prediction)
    choice_tokens = choice_match_tokens(choice_text)
    if not text_tokens or not choice_tokens:
        return 0.0
    overlap = sum((Counter(text_tokens) & Counter(choice_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(text_tokens)
    recall = overlap / len(choice_tokens)
    return 2 * precision * recall / (precision + recall)


def best_choice_label(prediction: str, choices: Dict[str, str]) -> Tuple[Optional[str], float, bool]:
    best_label = None
    best_score = 0.0
    second_score = 0.0
    for label, choice_text in choices.items():
        score = choice_token_score(prediction, choice_text)
        if score > best_score:
            second_score = best_score
            best_score = score
            best_label = label
        elif score > second_score:
            second_score = score
    ambiguous = best_label is not None and best_score > 0.0 and best_score < second_score + 0.05
    return best_label, best_score, ambiguous


def extract_choice_label(prediction: str, choices: Dict[str, str]) -> Optional[str]:
    if not choices:
        return None
    text = clean_generated_text(prediction)
    short = text.strip().upper().strip(".:()[]{}$ ")
    if short in choices:
        return short

    label_pattern = choice_label_regex(choices)
    patterns = [
        rf"\\boxed\s*\{{?\s*({label_pattern})\s*\}}?",
        rf"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?[^A-Za-z0-9]{{0,40}}\b({label_pattern})\b",
        rf"^\s*(?:answer\s*[:\-]?\s*)?\(?({label_pattern})\)?(?:[\.\):\s]|$)",
        rf"\b({label_pattern})\s*[\.\)]\s*[A-Za-z0-9]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in choices:
                return letter

    normalized_text = normalize_answer(text)
    for letter, choice_text in choices.items():
        normalized_choice = normalize_answer(choice_text)
        if normalized_text == normalized_choice:
            return letter
        if normalized_choice and len(normalized_choice.split()) == 1 and re.search(rf"\b{re.escape(normalized_choice)}\b", normalized_text):
            return letter

    best_letter, best_score, ambiguous = best_choice_label(text, choices)
    if best_letter and best_score >= 0.30 and not ambiguous:
        return best_letter
    return None


def force_prediction_to_choice(prediction: str, choices: Dict[str, str]) -> Tuple[str, Optional[str], bool, float]:
    label = extract_choice_label(prediction, choices)
    if label:
        return choices[label], label, False, 1.0

    best_label, best_score, ambiguous = best_choice_label(prediction, choices)
    if best_label:
        return choices[best_label], best_label, True, best_score

    first_label = next(iter(choices), None)
    if first_label is None:
        return "UNMAPPED", None, True, 0.0
    return choices[first_label], first_label, True, 0.0


def map_prediction_to_choice(prediction: str, choices: Dict[str, str]) -> str:
    text = clean_generated_text(prediction)
    if not choices:
        return text
    mapped, _, _, _ = force_prediction_to_choice(text, choices)
    return mapped


def map_prediction_to_yes_no(prediction: str) -> str:
    text = clean_generated_text(prediction)
    normalized = normalize_answer(text)
    if re.search(r"\bno\b", normalized):
        return "no"
    if re.search(r"\byes\b", normalized):
        return "yes"
    return "UNMAPPED"


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


def is_yes_no_vqa_row(row: Dict[str, Any]) -> bool:
    return (
        not row.get("choices")
        and str(row.get("normalized_ground_truth", "")).strip().lower() in {"yes", "no"}
    )


def is_closed_vqa_row(row: Dict[str, Any]) -> bool:
    return bool(row.get("choices"))


def is_open_vqa_row(row: Dict[str, Any]) -> bool:
    return not is_closed_vqa_row(row) and not is_yes_no_vqa_row(row)


def accuracy_value(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    if not rows:
        return None
    return float(statistics.mean(bool(row.get(key, False)) for row in rows))


def mean_float_value(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    if not rows:
        return None
    return float(statistics.mean(float(row.get(key, 0.0)) for row in rows))


def compute_vqa_group_metrics(
    group_name: str,
    group_rows: Sequence[Dict[str, Any]],
    metrics_config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    metric_group: Dict[str, Any] = {
        "samples": len(group_rows),
        "correct": sum(bool(row.get("correct", False)) for row in group_rows),
        "exact_match_correct": sum(bool(row.get("exact_match", False)) for row in group_rows),
        "normalized_accuracy": accuracy_value(group_rows, "correct"),
        "exact_match_accuracy": accuracy_value(group_rows, "exact_match"),
        "token_f1": mean_float_value(group_rows, "token_f1"),
    }
    if group_name == "closed":
        metric_group["forced_choice_count"] = sum(
            bool(row.get("prediction_was_forced_to_choice", False)) for row in group_rows
        )
        metric_group["mapped_choice_count"] = sum(bool(row.get("predicted_choice_label")) for row in group_rows)
    if group_name == "yes_no":
        metric_group["confusion_summary"] = confusion_summary(group_rows)

    if group_rows and bool_setting(metrics_config.get("group_text_metrics"), True):
        group_metric_config = dict(metrics_config)
        if not bool_setting(metrics_config.get("group_bertscore"), False):
            group_metric_config["bertscore"] = False
        if not bool_setting(metrics_config.get("group_cider_spice"), False):
            group_metric_config["cider"] = False
            group_metric_config["spice"] = False
        predictions = [str(row.get("prediction", "")) for row in group_rows]
        references = [str(row.get("ground_truth", "")) for row in group_rows]
        metric_group["text_metrics"] = compute_text_metrics(predictions, references, group_metric_config, logger)

    return metric_group


def compute_vqa_metric_groups(
    rows: Sequence[Dict[str, Any]],
    metrics_config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    closed_rows = [row for row in rows if is_closed_vqa_row(row)]
    yes_no_rows = [row for row in rows if is_yes_no_vqa_row(row)]
    open_rows = [row for row in rows if is_open_vqa_row(row)]
    grouped_rows = {
        "total": list(rows),
        "closed": closed_rows,
        "yes_no": yes_no_rows,
        "open": open_rows,
    }
    return {
        name: compute_vqa_group_metrics(name, group_rows, metrics_config, logger)
        for name, group_rows in grouped_rows.items()
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
    if metrics_config.get("meteor", True):
        metrics["METEOR"] = meteor_score_safe(predictions, references, logger)
    if metrics_config.get("bertscore", True):
        metrics["BERTScore"] = bertscore_safe(predictions, references, metrics_config, logger)
    if metrics_config.get("cider", True):
        metrics["CIDEr"] = cider_safe(predictions, references, logger)
    if metrics_config.get("spice", True):
        metrics["SPICE"] = spice_safe(predictions, references, logger)
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
        multiple_choice_rows = [row for row in rows if is_closed_vqa_row(row)]
        yes_no_rows = [row for row in rows if is_yes_no_vqa_row(row)]
        open_rows = [row for row in rows if is_open_vqa_row(row)]
        metrics["Total Samples"] = len(rows)
        metrics["Total Correct"] = sum(bool(row.get("correct", False)) for row in rows)
        metrics["Total Accuracy"] = float(statistics.mean(normalized_matches)) if rows else None
        metrics["Multiple Choice Samples"] = len(multiple_choice_rows)
        metrics["Closed Samples"] = len(multiple_choice_rows)
        metrics["Yes/No Samples"] = len(yes_no_rows)
        metrics["Open Samples"] = len(open_rows)
        metrics["Multiple Choice Correct"] = sum(bool(row.get("correct", False)) for row in multiple_choice_rows)
        metrics["Closed Correct"] = metrics["Multiple Choice Correct"]
        metrics["Yes/No Correct"] = sum(bool(row.get("correct", False)) for row in yes_no_rows)
        metrics["Open Exact Match Correct"] = sum(bool(row.get("exact_match", False)) for row in open_rows)
        metrics["Exact Match Accuracy"] = float(statistics.mean(exact_matches)) if rows else None
        metrics["Normalized Accuracy"] = float(statistics.mean(normalized_matches)) if rows else None
        metrics["Token-level F1"] = float(statistics.mean(f1_scores)) if rows else None
        metrics["Multiple Choice Accuracy"] = (
            float(statistics.mean(bool(row.get("correct", False)) for row in multiple_choice_rows))
            if multiple_choice_rows
            else None
        )
        metrics["Closed Accuracy"] = metrics["Multiple Choice Accuracy"]
        metrics["Yes/No Accuracy"] = (
            float(statistics.mean(bool(row.get("correct", False)) for row in yes_no_rows)) if yes_no_rows else None
        )
        metrics["Open Exact Match Accuracy"] = (
            float(statistics.mean(bool(row.get("exact_match", False)) for row in open_rows)) if open_rows else None
        )
        metrics["Open Token-level F1"] = (
            float(statistics.mean(float(row.get("token_f1", 0.0)) for row in open_rows)) if open_rows else None
        )
        metrics["Confusion Summary"] = confusion_summary(rows)
        if bool_setting(metrics_config.get("group_metrics"), True):
            metrics["Metric Groups"] = compute_vqa_metric_groups(rows, metrics_config, logger)
    metrics.update(compute_text_metrics(predictions, references, metrics_config, logger))
    return metrics


def bool_setting(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def model_output_name(config: Dict[str, Any]) -> str:
    raw = config.get("model_name_or_path") or config.get("model_path") or "medgemma"
    path = Path(str(raw).strip().rstrip("/\\"))
    name = path.name or "medgemma"
    return safe_filename(name)


def csv_metric_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8f}"
    return value


def per_sample_meteor_scores(
    predictions: Sequence[str],
    references: Sequence[str],
    logger: logging.Logger,
) -> List[Optional[float]]:
    if not predictions:
        return []
    if not package_available("nltk"):
        return [None] * len(predictions)
    try:
        from nltk.translate.meteor_score import meteor_score  # type: ignore

        return [
            float(meteor_score([metric_tokens(ref)], metric_tokens(pred)))
            for pred, ref in zip(predictions, references)
        ]
    except Exception as exc:
        logger.warning("Per-row METEOR export failed and will be left blank: %s", exc)
        logger.debug("Per-row METEOR traceback:\n%s", traceback.format_exc())
        return [None] * len(predictions)


def resolve_bertscore_model_type(value: Any, logger: logging.Logger, local_only: bool = False) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"none", "null"}:
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    candidates = [expanded] if expanded.is_absolute() else [(PROJECT_ROOT / expanded), (Path.cwd() / expanded)]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    looks_local = any(sep in raw for sep in ("/", "\\")) or raw.startswith((".", "~")) or ":" in raw
    if looks_local or local_only:
        logger.warning(
            "BERTScore local model path does not exist: %s. "
            "Download/copy roberta-large there or set metrics.bertscore_model_type to the correct local path.",
            str(candidates[0].resolve()),
        )
        return "__missing_local_bertscore_model__"
    return raw


def build_bertscore_kwargs(metrics_config: Dict[str, Any], logger: logging.Logger) -> Optional[Dict[str, Any]]:
    kwargs: Dict[str, Any] = {
        "lang": str(metrics_config.get("bertscore_lang", "en")),
        "verbose": bool(metrics_config.get("bertscore_verbose", False)),
        "rescale_with_baseline": bool(metrics_config.get("bertscore_rescale_with_baseline", False)),
    }
    if metrics_config.get("bertscore_model_type"):
        resolved_model_type = resolve_bertscore_model_type(
            metrics_config.get("bertscore_model_type"),
            logger,
            bool_setting(metrics_config.get("bertscore_local_only"), False),
        )
        if resolved_model_type == "__missing_local_bertscore_model__":
            return None
        if resolved_model_type:
            kwargs["model_type"] = resolved_model_type
    if metrics_config.get("bertscore_num_layers") is not None:
        kwargs["num_layers"] = int(metrics_config.get("bertscore_num_layers"))
    elif kwargs.get("model_type") and Path(str(kwargs["model_type"])).exists():
        model_path_name = Path(str(kwargs["model_type"])).name.lower()
        if model_path_name == "roberta-large":
            kwargs["num_layers"] = 17
            logger.info("BERTScore local roberta-large detected; using num_layers=17")
    if metrics_config.get("bertscore_device"):
        kwargs["device"] = str(metrics_config.get("bertscore_device"))
    return kwargs


def per_sample_bertscore_f1(
    predictions: Sequence[str],
    references: Sequence[str],
    metrics_config: Dict[str, Any],
    logger: logging.Logger,
) -> List[Optional[float]]:
    if not predictions:
        return []
    if not bool_setting(metrics_config.get("legacy_row_bertscore"), False):
        return [None] * len(predictions)
    if not package_available("bert_score"):
        logger.warning("Per-row BERTScore export skipped: bert_score is not installed")
        return [None] * len(predictions)
    try:
        from bert_score import score as bert_score  # type: ignore

        kwargs = build_bertscore_kwargs(metrics_config, logger)
        if kwargs is None:
            return [None] * len(predictions)
        _, _, f1 = bert_score(list(predictions), list(references), **kwargs)
        return [float(value) for value in f1.tolist()]
    except Exception as exc:
        logger.warning("Per-row BERTScore export failed and will be left blank: %s", exc)
        logger.debug("Per-row BERTScore traceback:\n%s", traceback.format_exc())
        return [None] * len(predictions)


def build_row_metric_cache(
    rows: Sequence[Dict[str, Any]],
    task: str,
    metrics_config: Dict[str, Any],
    logger: logging.Logger,
) -> List[Dict[str, Optional[float]]]:
    if task == "vqa" and vqa_export_kind(rows) == "open" and resolve_vqa_open_eval_style(metrics_config) == "med3dvlm":
        bundle = med3dvlm_eval_bundle(logger)
        if bundle is None:
            logger.warning("Falling back to legacy row metrics for VQA open.")
        else:
            return [
                med3dvlm_row_metrics(
                    row.get("prediction", ""),
                    row.get("ground_truth", ""),
                    bundle,
                    metrics_config,
                    logger,
                )
                for row in rows
            ]
    predictions = [str(row.get("prediction", "")) for row in rows]
    references = [str(row.get("ground_truth", "")) for row in rows]
    meteor_values = (
        per_sample_meteor_scores(predictions, references, logger)
        if bool_setting(metrics_config.get("meteor"), True)
        else [None] * len(rows)
    )
    bert_f1_values = per_sample_bertscore_f1(predictions, references, metrics_config, logger)

    cache: List[Dict[str, Optional[float]]] = []
    for index, (prediction, reference) in enumerate(zip(predictions, references)):
        bleu_score = corpus_bleu_scores([prediction], [reference]) if bool_setting(metrics_config.get("bleu"), True) else {}
        rouge_score = rouge_scores([prediction], [reference]) if bool_setting(metrics_config.get("rouge"), True) else {}
        cache.append(
            {
                "bleu": bleu_score.get("BLEU-1"),
                "rouge1": rouge_score.get("ROUGE-1"),
                "meteor": meteor_values[index] if index < len(meteor_values) else None,
                "bert_f1": bert_f1_values[index] if index < len(bert_f1_values) else None,
            }
        )
    return cache


def vqa_export_kind(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "mixed"
    close_count = 0
    open_count = 0
    for row in rows:
        is_close = bool(row.get("choices")) or str(row.get("normalized_ground_truth", "")).strip().lower() in {"yes", "no"}
        if is_close:
            close_count += 1
        else:
            open_count += 1
    if close_count and not open_count:
        return "close"
    if open_count and not close_count:
        return "open"
    return "mixed"


def csv_cell_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def debug_text_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def write_predict_debug_text(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    output_path = output_dir / "predict_debug.txt"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            mode = str(row.get("vqa_eval_mode") or "").strip().lower()
            raw_prediction = debug_text_value(row.get("raw_prediction", row.get("prediction")))
            prediction = debug_text_value(row.get("prediction"))
            handle.write("-------------------------\n")
            handle.write("[id_sample]\n")
            handle.write(f"{debug_text_value(row.get('sample_id'))}\n")
            if row.get("question"):
                handle.write("[QUESTION]\n")
                handle.write(f"{debug_text_value(row.get('question'))}\n")
            if row.get("answer_choice"):
                option_label = "Ground-truth option"
                if mode == "open":
                    option_label += " (debug only; hidden from prompt)"
                handle.write(f"{option_label}: {debug_text_value(row.get('answer_choice'))}\n")
            if mode == "open" and row.get("all_choices"):
                handle.write(f"Hidden choices (not shown to model): {debug_text_value(row.get('all_choices'))}\n")
            elif row.get("choices"):
                handle.write(f"Choices: {debug_text_value(row.get('choices'))}\n")
            handle.write("[PR]\n")
            handle.write(f"{prediction}\n")
            if raw_prediction and raw_prediction != prediction:
                handle.write("[RAW_PR]\n")
                handle.write(f"{raw_prediction}\n")
            handle.write("[GT]\n")
            handle.write(f"{debug_text_value(row.get('ground_truth'))}\n")
            if mode == "open":
                handle.write("[TEXT_EXACT_MATCH_DEBUG]\n")
                handle.write(f"{debug_text_value(row.get('exact_match'))}\n")
                handle.write("[TOKEN_F1_DEBUG]\n")
                handle.write(f"{debug_text_value(row.get('token_f1'))}\n")
                handle.write("[SCORE_MODE]\n")
                handle.write("open_text_metrics: use BLEU/ROUGE/METEOR/BERTScore CSV, not this exact-match debug flag.\n")
            else:
                handle.write("[True/False]\n")
                handle.write(f"{debug_text_value(row.get('correct'))}\n")
                if row.get("predicted_choice_label"):
                    handle.write(f"Predicted option: {debug_text_value(row.get('predicted_choice_label'))}\n")
            if row.get("question_type") or row.get("question_type_name"):
                handle.write("[QUESTION_TYPE]\n")
                handle.write(
                    f"{debug_text_value(row.get('question_type'))} "
                    f"{debug_text_value(row.get('question_type_name'))}".strip()
                    + "\n"
                )
            if row.get("vqa_eval_mode"):
                handle.write("[VQA_MODE]\n")
                handle.write(f"{debug_text_value(row.get('vqa_eval_mode'))}\n")
            if row.get("image_path"):
                handle.write("[IMAGE_PATH]\n")
                handle.write(f"{debug_text_value(row.get('image_path'))}\n")
            handle.write("----------------------\n")
    return output_path


def write_predict_exports(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> List[Path]:
    predict_jsonl = output_dir / "predict.jsonl"
    slim_keys = [
        "id_sample",
        "PR",
        "GT",
        "raw_PR",
        "sample_id",
        "image_path",
        "question",
        "question_type",
        "question_type_name",
        "vqa_eval_mode",
        "prompt",
        "prediction",
        "raw_prediction",
        "ground_truth",
        "ground_truth_with_choice",
        "answer_choice",
        "predicted_choice_label",
        "prediction_was_forced_to_choice",
        "choice_match_score",
        "letter_correct",
        "score_mode",
        "correct",
        "exact_match",
        "token_f1",
        "choices",
        "all_choices",
        "view",
        "num_slices",
        "selected_slice_indices",
        "inference_mode",
        "montage_path",
    ]
    with predict_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "id_sample": row.get("sample_id"),
                "PR": row.get("prediction"),
                "GT": row.get("ground_truth"),
                "raw_PR": row.get("raw_prediction", row.get("prediction")),
            }
            payload.update({key: row.get(key) for key in slim_keys if key in row})
            append_jsonl(handle, payload)

    predict_csv = output_dir / "predict.csv"
    fieldnames = [
        "id_sample",
        "PR",
        "GT",
        "raw_PR",
        "sample_id",
        "image_path",
        "question",
        "question_type",
        "question_type_name",
        "vqa_eval_mode",
        "prediction",
        "raw_prediction",
        "ground_truth",
        "ground_truth_with_choice",
        "answer_choice",
        "predicted_choice_label",
        "prediction_was_forced_to_choice",
        "choice_match_score",
        "letter_correct",
        "score_mode",
        "correct",
        "exact_match",
        "token_f1",
        "choices",
        "all_choices",
        "view",
        "num_slices",
        "selected_slice_indices",
        "inference_mode",
        "montage_path",
    ]
    with predict_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            export_row = {
                "id_sample": row.get("sample_id"),
                "PR": row.get("prediction"),
                "GT": row.get("ground_truth"),
                "raw_PR": row.get("raw_prediction", row.get("prediction")),
            }
            export_row.update(row)
            writer.writerow({key: csv_cell_value(export_row.get(key)) for key in fieldnames})

    predict_debug = write_predict_debug_text(output_dir, rows)
    return [predict_jsonl, predict_csv, predict_debug]


def write_med3dvlm_caption_csv(
    output_dir: Path,
    model_name: str,
    rows: Sequence[Dict[str, Any]],
    row_metrics: Sequence[Dict[str, Optional[float]]],
) -> Path:
    output_path = output_dir / f"{model_name}_eval_caption.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Question", "Ground Truth", "pred", "bleu", "rouge1", "meteor", "bert_f1"])
        for row, metric in zip(rows, row_metrics):
            writer.writerow(
                [
                    row.get("prompt", ""),
                    row.get("ground_truth", ""),
                    row.get("prediction", ""),
                    csv_metric_value(metric.get("bleu")),
                    csv_metric_value(metric.get("rouge1")),
                    csv_metric_value(metric.get("meteor")),
                    csv_metric_value(metric.get("bert_f1")),
                ]
            )
    return output_path


def integer_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return str(int(round(value)))
    if isinstance(value, int):
        return str(value)
    return value


def write_caption_report_table_csv(
    output_dir: Path,
    model_name: str,
    metrics: Dict[str, Any],
) -> Path:
    output_path = output_dir / f"{model_name}_eval_caption_report_table.csv"
    bertscore = metrics.get("BERTScore") if isinstance(metrics.get("BERTScore"), dict) else {}
    params = metrics.get("Parameters") if isinstance(metrics.get("Parameters"), dict) else {}
    flops = metrics.get("FLOPs") if isinstance(metrics.get("FLOPs"), dict) else {}
    row = {
        "Method": model_name,
        "BLEU": percent_csv_value(metrics.get("BLEU-1")),
        "ROUGE": percent_csv_value(metrics.get("ROUGE-1")),
        "METEOR": percent_csv_value(metrics.get("METEOR")),
        "BERTScore": percent_csv_value(bertscore.get("F1") if isinstance(bertscore, dict) else None),
        "Parameters": integer_csv_value(params.get("total_params") if isinstance(params, dict) else None),
        "Flops": integer_csv_value(flops.get("approx_total_flops") if isinstance(flops, dict) else None),
    }
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Method", "BLEU", "ROUGE", "METEOR", "BERTScore", "Parameters", "Flops"],
        )
        writer.writeheader()
        writer.writerow(row)
    return output_path


def write_med3dvlm_vqa_csv(
    output_dir: Path,
    model_name: str,
    rows: Sequence[Dict[str, Any]],
    row_metrics: Sequence[Dict[str, Optional[float]]],
) -> Path:
    kind = vqa_export_kind(rows)
    if kind == "close":
        output_path = output_dir / f"{model_name}_eval_close_vqa.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Question Type", "Question", "Answer", "Answer Choice", "Pred", "Correct"])
            for row in rows:
                writer.writerow(
                    [
                        row.get("question_type", ""),
                        row.get("question", ""),
                        row.get("ground_truth", ""),
                        row.get("answer_choice", ""),
                        row.get("prediction", ""),
                        int(bool(row.get("correct", False))),
                    ]
                )
        return output_path

    output_path = output_dir / f"{model_name}_eval_open_vqa.csv" if kind == "open" else output_dir / f"{model_name}_eval_vqa.csv"
    header = ["Question Type", "Question", "Answer", "Pred", "bleu", "rouge1", "meteor", "bert_f1"]
    if kind == "mixed":
        header = ["Question Type", "Question", "Answer", "Answer Choice", "Pred", "Correct", "bleu", "rouge1", "meteor", "bert_f1"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row, metric in zip(rows, row_metrics):
            values = [
                row.get("question_type", ""),
                row.get("question", ""),
                row.get("ground_truth", ""),
            ]
            if kind == "mixed":
                values.extend([row.get("answer_choice", ""), row.get("prediction", ""), int(bool(row.get("correct", False)))])
            else:
                values.append(row.get("prediction", ""))
            values.extend(
                [
                    csv_metric_value(metric.get("bleu")),
                    csv_metric_value(metric.get("rouge1")),
                    csv_metric_value(metric.get("meteor")),
                    csv_metric_value(metric.get("bert_f1")),
                ]
            )
            writer.writerow(values)
    return output_path


def question_type_sort_key(type_id: str) -> Tuple[int, str]:
    text = str(type_id or "")
    if text.isdigit():
        return int(text), text
    return 999, text


def group_rows_by_question_type(
    rows: Sequence[Dict[str, Any]],
    row_metrics: Sequence[Dict[str, Optional[float]]],
) -> Dict[str, List[Tuple[Dict[str, Any], Dict[str, Optional[float]]]]]:
    grouped: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Optional[float]]]]] = defaultdict(list)
    for row, metric in zip(rows, row_metrics):
        type_id = str(row.get("question_type") or row.get("question_type_id") or "").strip()
        if type_id.endswith(".0"):
            type_id = type_id[:-2]
        if not type_id:
            type_id = "unknown"
        grouped[type_id].append((row, metric))
    return dict(grouped)


def mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None]
    return float(statistics.mean(valid)) if valid else None


def percent_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value * 100.0:.2f}"
    return value


def write_vqa_open_question_type_summary_csv(
    output_dir: Path,
    model_name: str,
    rows: Sequence[Dict[str, Any]],
    row_metrics: Sequence[Dict[str, Optional[float]]],
) -> Path:
    output_path = output_dir / f"{model_name}_eval_open_vqa_by_question_type.csv"
    metric_names = ["bleu", "rouge1", "meteor", "bert_f1"]
    grouped = group_rows_by_question_type(rows, row_metrics)
    type_summaries: List[Dict[str, Any]] = []
    for type_id in sorted(grouped.keys(), key=question_type_sort_key):
        pairs = grouped[type_id]
        summary = {
            "Question Type": type_id,
            "Question Type Name": VQA_QUESTION_TYPE_NAMES.get(type_id, type_id),
            "Count": len(pairs),
        }
        for metric_name in metric_names:
            summary[metric_name] = mean_optional([metric.get(metric_name) for _, metric in pairs])
        type_summaries.append(summary)

    macro_summary = {
        "Question Type": "macro_mean",
        "Question Type Name": "Mean over question types",
        "Count": len(type_summaries),
    }
    for metric_name in metric_names:
        macro_summary[metric_name] = mean_optional([summary.get(metric_name) for summary in type_summaries])

    micro_summary = {
        "Question Type": "micro_total",
        "Question Type Name": "All samples",
        "Count": len(rows),
    }
    for metric_name in metric_names:
        micro_summary[metric_name] = mean_optional([metric.get(metric_name) for metric in row_metrics])

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Question Type", "Question Type Name", "Count", *metric_names],
        )
        writer.writeheader()
        for summary in [*type_summaries, macro_summary, micro_summary]:
            writer.writerow({key: csv_metric_value(value) for key, value in summary.items()})
    return output_path


def write_vqa_closed_text_question_type_summary_csv(
    output_dir: Path,
    model_name: str,
    rows: Sequence[Dict[str, Any]],
    row_metrics: Sequence[Dict[str, Optional[float]]],
) -> Path:
    output_path = output_dir / f"{model_name}_eval_close_vqa_text_by_question_type.csv"
    metric_names = ["bleu", "rouge1", "meteor", "bert_f1"]
    grouped = group_rows_by_question_type(rows, row_metrics)
    type_summaries: List[Dict[str, Any]] = []
    for type_id in sorted(grouped.keys(), key=question_type_sort_key):
        pairs = grouped[type_id]
        summary = {
            "Question Type": type_id,
            "Question Type Name": VQA_QUESTION_TYPE_NAMES.get(type_id, type_id),
            "Count": len(pairs),
        }
        for metric_name in metric_names:
            summary[metric_name] = mean_optional([metric.get(metric_name) for _, metric in pairs])
        type_summaries.append(summary)

    macro_summary = {
        "Question Type": "macro_mean",
        "Question Type Name": "Mean over question types",
        "Count": len(type_summaries),
    }
    for metric_name in metric_names:
        macro_summary[metric_name] = mean_optional([summary.get(metric_name) for summary in type_summaries])

    micro_summary = {
        "Question Type": "micro_total",
        "Question Type Name": "All samples",
        "Count": len(rows),
    }
    for metric_name in metric_names:
        micro_summary[metric_name] = mean_optional([metric.get(metric_name) for metric in row_metrics])

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Question Type", "Question Type Name", "Count", *metric_names],
        )
        writer.writeheader()
        for summary in [*type_summaries, macro_summary, micro_summary]:
            writer.writerow({key: csv_metric_value(value) for key, value in summary.items()})
    return output_path


def vqa_question_type_metric_means(
    rows: Sequence[Dict[str, Any]],
    row_metrics: Sequence[Dict[str, Optional[float]]],
) -> Dict[str, Dict[str, Optional[float]]]:
    grouped = group_rows_by_question_type(rows, row_metrics)
    means: Dict[str, Dict[str, Optional[float]]] = {}
    for type_id in sorted(grouped.keys(), key=question_type_sort_key):
        pairs = grouped[type_id]
        means[type_id] = {
            "bleu": mean_optional([metric.get("bleu") for _, metric in pairs]),
            "rouge1": mean_optional([metric.get("rouge1") for _, metric in pairs]),
            "meteor": mean_optional([metric.get("meteor") for _, metric in pairs]),
            "bert_f1": mean_optional([metric.get("bert_f1") for _, metric in pairs]),
            "accuracy": mean_optional([1.0 if bool(row.get("correct", False)) else 0.0 for row, _ in pairs]),
            "exact_match": mean_optional([1.0 if bool(row.get("exact_match", False)) else 0.0 for row, _ in pairs]),
            "token_f1": mean_optional([float(row.get("token_f1", 0.0)) for row, _ in pairs]),
        }
    return means


def write_vqa_open_question_type_table_csv(
    output_dir: Path,
    model_name: str,
    rows: Sequence[Dict[str, Any]],
    row_metrics: Sequence[Dict[str, Optional[float]]],
) -> Path:
    output_path = output_dir / f"{model_name}_eval_open_vqa_question_type_table.csv"
    means = vqa_question_type_metric_means(rows, row_metrics)
    metric_rows = [
        ("BLEU", "bleu"),
        ("ROUGE", "rouge1"),
        ("METEOR", "meteor"),
        ("BERTScore", "bert_f1"),
    ]
    fieldnames = ["Method", "Metric", "Plane", "Phase", "Organ", "Abnormality", "Location", "Mean"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for display_name, metric_key in metric_rows:
            values = []
            row = {"Method": model_name, "Metric": display_name}
            for type_id, type_name in VQA_QUESTION_TYPE_NAMES.items():
                value = means.get(type_id, {}).get(metric_key)
                values.append(value)
                row[type_name] = percent_csv_value(value)
            row["Mean"] = percent_csv_value(mean_optional(values))
            writer.writerow(row)
    return output_path


def write_vqa_closed_question_type_table_csv(
    output_dir: Path,
    model_name: str,
    rows: Sequence[Dict[str, Any]],
    row_metrics: Sequence[Dict[str, Optional[float]]],
) -> Path:
    output_path = output_dir / f"{model_name}_eval_close_vqa_question_type_table.csv"
    means = vqa_question_type_metric_means(rows, row_metrics)
    metric_rows = [
        ("Accuracy", "accuracy"),
        ("Exact Match", "exact_match"),
        ("Token F1", "token_f1"),
    ]
    fieldnames = ["Method", "Metric", "Plane", "Phase", "Organ", "Abnormality", "Location", "Mean"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for display_name, metric_key in metric_rows:
            values = []
            row = {"Method": model_name, "Metric": display_name}
            for type_id, type_name in VQA_QUESTION_TYPE_NAMES.items():
                value = means.get(type_id, {}).get(metric_key)
                values.append(value)
                row[type_name] = percent_csv_value(value)
            row["Mean"] = percent_csv_value(mean_optional(values))
            writer.writerow(row)
    return output_path


def write_vqa_closed_question_type_summary_csv(
    output_dir: Path,
    model_name: str,
    rows: Sequence[Dict[str, Any]],
    row_metrics: Sequence[Dict[str, Optional[float]]],
) -> Path:
    output_path = output_dir / f"{model_name}_eval_close_vqa_by_question_type.csv"
    grouped = group_rows_by_question_type(rows, row_metrics)
    type_summaries: List[Dict[str, Any]] = []
    for type_id in sorted(grouped.keys(), key=question_type_sort_key):
        pairs = grouped[type_id]
        total = len(pairs)
        correct = sum(bool(row.get("correct", False)) for row, _ in pairs)
        type_summaries.append(
            {
                "Question Type": type_id,
                "Question Type Name": VQA_QUESTION_TYPE_NAMES.get(type_id, type_id),
                "Total": total,
                "Correct": correct,
                "Accuracy": correct / total if total else None,
            }
        )

    macro_summary = {
        "Question Type": "macro_mean",
        "Question Type Name": "Mean over question types",
        "Total": len(type_summaries),
        "Correct": "",
        "Accuracy": mean_optional([summary.get("Accuracy") for summary in type_summaries]),
    }
    total = len(rows)
    correct = sum(bool(row.get("correct", False)) for row in rows)
    micro_summary = {
        "Question Type": "micro_total",
        "Question Type Name": "All samples",
        "Total": total,
        "Correct": correct,
        "Accuracy": correct / total if total else None,
    }

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Question Type", "Question Type Name", "Total", "Correct", "Accuracy"],
        )
        writer.writeheader()
        for summary in [*type_summaries, macro_summary, micro_summary]:
            writer.writerow({key: csv_metric_value(value) for key, value in summary.items()})
    return output_path


def write_vqa_question_type_summary_csv(
    output_dir: Path,
    model_name: str,
    rows: Sequence[Dict[str, Any]],
    row_metrics: Sequence[Dict[str, Optional[float]]],
) -> List[Path]:
    if not rows:
        return []
    mode = str(rows[0].get("vqa_eval_mode") or "").strip().lower()
    if not mode:
        mode = "closed" if any(row.get("choices") for row in rows) else "open"
    if mode == "closed":
        return [
            write_vqa_closed_question_type_summary_csv(output_dir, model_name, rows, row_metrics),
            write_vqa_closed_question_type_table_csv(output_dir, model_name, rows, row_metrics),
            write_vqa_closed_text_question_type_summary_csv(output_dir, model_name, rows, row_metrics),
        ]
    return [
        write_vqa_open_question_type_summary_csv(output_dir, model_name, rows, row_metrics),
        write_vqa_open_question_type_table_csv(output_dir, model_name, rows, row_metrics),
    ]


def flatten_metric_rows(
    payload: Any,
    prefix: str = "",
) -> List[Tuple[str, Any]]:
    rows: List[Tuple[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_metric_rows(value, path))
    elif isinstance(payload, list):
        if all(not isinstance(item, (dict, list)) for item in payload):
            rows.append((prefix, json.dumps(payload, ensure_ascii=False)))
        else:
            for index, value in enumerate(payload):
                rows.extend(flatten_metric_rows(value, f"{prefix}.{index}"))
    else:
        rows.append((prefix, payload))
    return rows


def write_metrics_full_csv(output_dir: Path, metrics: Dict[str, Any]) -> Path:
    output_path = output_dir / "metrics_full.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric_path, value in flatten_metric_rows(metrics):
            if isinstance(value, float):
                value = csv_metric_value(value)
            writer.writerow([metric_path, value if value is not None else ""])
    return output_path


def write_vqa_overall_metrics_csv(output_dir: Path, metrics: Dict[str, Any]) -> Optional[Path]:
    if str(metrics.get("task") or "").lower() != "vqa":
        return None
    output_path = output_dir / "vqa_overall_metrics.csv"
    bertscore = metrics.get("BERTScore") if isinstance(metrics.get("BERTScore"), dict) else {}
    selected_rows = [
        ("samples.total", metrics.get("Total Samples", metrics.get("num_successful_samples"))),
        ("samples.requested", metrics.get("num_requested_samples")),
        ("samples.failed", metrics.get("num_failed_samples")),
        ("overall.normalized_accuracy", metrics.get("Normalized Accuracy")),
        ("overall.exact_match_accuracy", metrics.get("Exact Match Accuracy")),
        ("overall.token_f1", metrics.get("Token-level F1")),
        ("closed.qa_accuracy", metrics.get("Closed Accuracy", metrics.get("Multiple Choice Accuracy"))),
        ("closed.correct", metrics.get("Closed Correct", metrics.get("Multiple Choice Correct"))),
        ("closed.samples", metrics.get("Closed Samples", metrics.get("Multiple Choice Samples"))),
        ("yes_no.accuracy", metrics.get("Yes/No Accuracy")),
        ("yes_no.correct", metrics.get("Yes/No Correct")),
        ("yes_no.samples", metrics.get("Yes/No Samples")),
        ("open.exact_match_accuracy", metrics.get("Open Exact Match Accuracy")),
        ("open.token_f1", metrics.get("Open Token-level F1")),
        ("open.samples", metrics.get("Open Samples")),
        ("text.BLEU-1", metrics.get("BLEU-1")),
        ("text.BLEU-2", metrics.get("BLEU-2")),
        ("text.BLEU-3", metrics.get("BLEU-3")),
        ("text.BLEU-4", metrics.get("BLEU-4")),
        ("text.ROUGE-1", metrics.get("ROUGE-1")),
        ("text.ROUGE-2", metrics.get("ROUGE-2")),
        ("text.ROUGE-L", metrics.get("ROUGE-L")),
        ("text.METEOR", metrics.get("METEOR")),
        ("text.BERTScore.Precision", bertscore.get("Precision") if isinstance(bertscore, dict) else None),
        ("text.BERTScore.Recall", bertscore.get("Recall") if isinstance(bertscore, dict) else None),
        ("text.BERTScore.F1", bertscore.get("F1") if isinstance(bertscore, dict) else None),
        ("text.CIDEr", metrics.get("CIDEr")),
        ("text.SPICE", metrics.get("SPICE")),
    ]

    for section_name in ["slice_inference", "Parameters", "FLOPs", "Runtime"]:
        section = metrics.get(section_name)
        if isinstance(section, dict):
            for key, value in flatten_metric_rows(section, section_name):
                selected_rows.append((key, value))

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric_name, value in selected_rows:
            writer.writerow([metric_name, csv_metric_value(value)])
    return output_path


def write_eval_exports(
    output_dir: Path,
    task: str,
    rows: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    metrics_config = dict(config.get("metrics") or {})
    logging_config = dict(config.get("logging") or {})
    model_name = model_output_name(config)

    if bool_setting(logging_config.get("save_predictions"), True):
        predict_paths = write_predict_exports(output_dir, rows)
        logger.info("Saved predict exports: %s", ", ".join(str(path) for path in predict_paths))

    if bool(metrics.get("skip_metrics", False)):
        return

    metrics_full_path = write_metrics_full_csv(output_dir, metrics)
    logger.info("Saved full metric CSV: %s", metrics_full_path)
    vqa_overall_path = write_vqa_overall_metrics_csv(output_dir, metrics)
    if vqa_overall_path is not None:
        logger.info("Saved VQA overall metric CSV: %s", vqa_overall_path)

    if bool_setting(metrics_config.get("export_med3dvlm_csv"), True):
        row_metrics = build_row_metric_cache(rows, task, metrics_config, logger)
        if task == "cap":
            csv_path = write_med3dvlm_caption_csv(output_dir, model_name, rows, row_metrics)
            extra_csv_paths: List[Path] = [write_caption_report_table_csv(output_dir, model_name, metrics)]
        else:
            csv_path = write_med3dvlm_vqa_csv(output_dir, model_name, rows, row_metrics)
            extra_csv_paths = write_vqa_question_type_summary_csv(output_dir, model_name, rows, row_metrics)
        logger.info("Saved Med3DVLM-style CSV: %s", csv_path)
        for extra_csv_path in extra_csv_paths:
            logger.info("Saved metric report CSV: %s", extra_csv_path)

    if bool_setting(metrics_config.get("export_extra_metrics_file"), True):
        write_json(output_dir / "metrics_extra.json", metrics)
        logger.info("Saved extra metrics: %s", output_dir / "metrics_extra.json")

    if task == "vqa" and bool_setting(metrics_config.get("export_med3dvlm_metrics"), True):
        med3dvlm_metrics = compute_vqa_med3dvlm_metrics(rows, metrics_config, logger)
        if med3dvlm_metrics is not None:
            output_path = output_dir / "metrics_med3dvlm.json"
            write_json(output_path, med3dvlm_metrics)
            logger.info("Saved Med3DVLM-style VQA metrics: %s", output_path)

    metric_groups = metrics.get("Metric Groups")
    if isinstance(metric_groups, dict):
        write_json(output_dir / "metrics_by_group.json", metric_groups)
        logger.info("Saved grouped metrics: %s", output_dir / "metrics_by_group.json")


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
    lines = ["[SAMPLE]", f"ID: {row.get('sample_id')}"]
    if row.get("requested_num_slices") is not None:
        lines.append(f"Requested slices: {row.get('requested_num_slices')}")
    if row.get("num_slices") is not None:
        lines.append(f"Actual slices: {row.get('num_slices')}")
    slice_indices = row.get("selected_slice_indices") or row.get("slice_indices")
    if slice_indices is not None:
        lines.append(f"Slice indices: {slice_indices}")
    if task == "vqa" and row.get("question"):
        lines.extend(["[QUESTION]", str(row.get("question", ""))])
        if row.get("answer_choice"):
            lines.append(f"Ground-truth option: {row.get('answer_choice')}")
        if row.get("predicted_choice_label"):
            lines.append(f"Predicted option: {row.get('predicted_choice_label')}")
        if row.get("prediction_was_forced_to_choice"):
            lines.append(f"Forced to input option: True (score={row.get('choice_match_score')})")
    lines.extend(["[PR]", str(row.get("prediction", "")), "[GT]", str(row.get("ground_truth", ""))])
    if task == "vqa":
        correct = row.get("correct")
        value = "N/A" if correct is None else str(bool(correct))
        lines.extend(["[True/False]", value])
    if task == "cap":
        lines.append("[METRIC]")
        lines.append("See final Metric summary for BLEU-1/2/3/4, ROUGE-1/2/L, and BERTScore.")
    return "\n".join(lines)


def get_progress(iterable: Iterable[Any], total: int, desc: str, logger: logging.Logger) -> Iterable[Any]:
    if package_available("tqdm"):
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc)
    logger.warning("tqdm unavailable: progress bar disabled")
    return iterable


def aggregate_vqa_predictions(per_slice_predictions: Sequence[Dict[str, Any]], choices: Dict[str, str]) -> str:
    mapped_answers = [str(row.get("mapped_answer") or row.get("prediction") or "") for row in per_slice_predictions]
    counts = Counter(exact_normalize(answer) for answer in mapped_answers if exact_normalize(answer))
    if not counts:
        return ""
    best_count = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == best_count}
    for answer in mapped_answers:
        normalized = exact_normalize(answer)
        if normalized in tied:
            if choices:
                for choice_text in choices.values():
                    if exact_normalize(choice_text) == normalized:
                        return choice_text
            return answer
    return mapped_answers[0] if mapped_answers else ""


def join_caption_predictions(per_slice_predictions: Sequence[Dict[str, Any]]) -> str:
    captions = [str(row.get("prediction", "")).strip() for row in per_slice_predictions]
    return " ".join(caption for caption in captions if caption)


def evaluate_loop(
    task: str,
    samples: Sequence[EvalSample],
    bundle: ModelBundle,
    config: Dict[str, Any],
    slice_config: SliceInferenceConfig,
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
                slice_bundle = prepare_slice_images(
                    sample=sample,
                    task=task,
                    slice_config=slice_config,
                    processor=bundle.processor,
                    image_config=image_config,
                    output_dir=output_dir,
                )
                logger.info(
                    "Slice selection | sample_id=%s | view=%s | requested_num_slices=%s | actual_num_slices=%d | strategy=%s | mode=%s | indices=%s",
                    sample.sample_id,
                    slice_config.view,
                    slice_config.num_slices,
                    slice_bundle.effective_num_slices,
                    slice_config.slice_strategy,
                    slice_config.inference_mode,
                    slice_bundle.selected_slice_indices,
                )

                per_slice_predictions: List[Dict[str, Any]] = []
                if slice_config.inference_mode == "independent":
                    elapsed = 0.0
                    generated_tokens = 0
                    for idx, image in enumerate(slice_bundle.images):
                        slice_prompt = slice_bundle.per_slice_prompts[idx]
                        slice_prediction, slice_elapsed, slice_generated_tokens = generate_prediction(
                            bundle,
                            image,
                            slice_prompt,
                            generation_kwargs,
                        )
                        elapsed += slice_elapsed
                        generated_tokens += slice_generated_tokens
                        slice_row: Dict[str, Any] = {
                            "slice_order": idx,
                            "slice_index": slice_bundle.selected_slice_indices[idx],
                            "prompt": slice_prompt,
                            "prediction": slice_prediction,
                            "inference_time": slice_elapsed,
                            "generated_tokens": slice_generated_tokens,
                        }
                        if task == "vqa":
                            slice_row["mapped_answer"] = map_prediction_to_choice(slice_prediction, sample.choices)
                        per_slice_predictions.append(slice_row)
                    if task == "vqa":
                        prediction = aggregate_vqa_predictions(per_slice_predictions, sample.choices)
                    else:
                        prediction = join_caption_predictions(per_slice_predictions)
                else:
                    prediction, elapsed, generated_tokens = generate_prediction(
                        bundle,
                        slice_bundle.images[0],
                        slice_bundle.prompt,
                        generation_kwargs,
                    )

                inference_times.append(elapsed)
                total_generated_tokens += generated_tokens
                raw_prediction = prediction

                base_row: Dict[str, Any] = {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "view": slice_config.view,
                    "num_slices": slice_bundle.effective_num_slices,
                    "requested_num_slices": slice_config.num_slices,
                    "slice_strategy": slice_config.slice_strategy,
                    "selected_slice_indices": slice_bundle.selected_slice_indices,
                    "inference_mode": slice_config.inference_mode,
                    "prompt": slice_bundle.prompt,
                    "prediction": prediction,
                    "ground_truth": sample.ground_truth,
                    "inference_time": elapsed,
                    "generated_tokens": generated_tokens,
                }
                if slice_config.inference_mode == "independent":
                    base_row["per_slice_predictions"] = per_slice_predictions
                if slice_bundle.montage_path:
                    base_row["montage_path"] = slice_bundle.montage_path
                if task == "cap":
                    base_row["split"] = sample.split
                else:
                    if sample.choices:
                        score_mode = "closed_choice_accuracy"
                        prediction_answer, predicted_choice_label, forced_choice, choice_match_score = force_prediction_to_choice(
                            raw_prediction, sample.choices
                        )
                        answer_choice = str(sample.answer_choice or "").strip()
                        letter_correct = bool(
                            answer_choice
                            and predicted_choice_label
                            and predicted_choice_label.upper() == answer_choice.upper()
                        )
                        text_correct = normalize_answer(prediction_answer) == normalize_answer(sample.ground_truth)
                        exact_match = exact_normalize(prediction_answer) == exact_normalize(sample.ground_truth)
                        correct = bool(letter_correct or text_correct)
                    elif normalize_answer(sample.ground_truth) in {"yes", "no"}:
                        score_mode = "yes_no_accuracy"
                        prediction_answer = map_prediction_to_yes_no(raw_prediction)
                        predicted_choice_label = None
                        forced_choice = False
                        choice_match_score = None
                        letter_correct = None
                        exact_match = exact_normalize(prediction_answer) == exact_normalize(sample.ground_truth)
                        correct = normalize_answer(prediction_answer) == normalize_answer(sample.ground_truth)
                    else:
                        score_mode = "open_text_metrics"
                        prediction_answer = clean_generated_text(raw_prediction)
                        predicted_choice_label = None
                        forced_choice = False
                        choice_match_score = None
                        letter_correct = None
                        exact_match = exact_normalize(prediction_answer) == exact_normalize(sample.ground_truth)
                        correct = normalize_answer(prediction_answer) == normalize_answer(sample.ground_truth)
                    base_row["raw_prediction"] = raw_prediction
                    base_row["prediction"] = prediction_answer
                    if sample.choices:
                        base_row["predicted_choice_label"] = predicted_choice_label
                        base_row["prediction_was_forced_to_choice"] = forced_choice
                        base_row["choice_match_score"] = choice_match_score
                        base_row["letter_correct"] = letter_correct
                        base_row["ground_truth_with_choice"] = sample.meta.get("ground_truth_with_choice")
                    normalized_prediction = normalize_answer(prediction_answer)
                    normalized_ground_truth = normalize_answer(sample.ground_truth)
                    base_row.update(
                        {
                            "question": sample.question,
                            "question_type": sample.meta.get("question_type_id", sample.meta.get("question_type")),
                            "question_type_name": sample.meta.get("question_type_name"),
                            "answer_type": sample.meta.get("answer_type"),
                            "vqa_eval_mode": sample.meta.get("vqa_eval_mode"),
                            "choices": sample.choices,
                            "all_choices": sample.meta.get("all_choices"),
                            "answer_choice": sample.answer_choice,
                            "score_mode": score_mode,
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
                    "view": slice_config.view,
                    "num_slices": slice_config.num_slices,
                    "slice_strategy": slice_config.slice_strategy,
                    "inference_mode": slice_config.inference_mode,
                    "prompt": build_montage_prompt(task, sample, slice_config, slice_config.num_slices),
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


def metric_display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def log_summary_row(logger: logging.Logger, label: str, value: Any, note: str = "") -> None:
    suffix = f"  {note}" if note else ""
    logger.info("  %-30s %s%s", label, metric_display(value), suffix)


def log_accuracy_row(
    logger: logging.Logger,
    label: str,
    accuracy: Any,
    correct: Any,
    total: Any,
) -> None:
    if total is None or int(total) == 0:
        log_summary_row(logger, label, None, "(0 samples)")
        return
    log_summary_row(logger, label, accuracy, f"({correct}/{total})")


def log_metric_summary(metrics: Dict[str, Any], logger: logging.Logger) -> None:
    task = str(metrics.get("task") or "").lower()
    logger.info("[METRIC] %s", task.upper() if task else "UNKNOWN")

    logger.info("Samples:")
    log_summary_row(logger, "successful", metrics.get("num_successful_samples"))
    log_summary_row(logger, "requested", metrics.get("num_requested_samples"))
    log_summary_row(logger, "failed", metrics.get("num_failed_samples"))
    if task == "vqa":
        log_summary_row(logger, "closed", metrics.get("Closed Samples", metrics.get("Multiple Choice Samples", 0)))
        log_summary_row(logger, "yes_no", metrics.get("Yes/No Samples", 0))
        log_summary_row(logger, "open_ended", metrics.get("Open Samples", 0))

    if task == "vqa":
        logger.info("VQA core metrics:")
        total = metrics.get("num_successful_samples", 0)
        normalized_correct = None
        if metrics.get("Normalized Accuracy") is not None and total:
            normalized_correct = round(float(metrics["Normalized Accuracy"]) * int(total))
        log_accuracy_row(logger, "overall_normalized_acc", metrics.get("Normalized Accuracy"), normalized_correct, total)
        log_summary_row(logger, "overall_exact_match", metrics.get("Exact Match Accuracy"))
        log_summary_row(logger, "overall_token_f1", metrics.get("Token-level F1"))
        log_accuracy_row(
            logger,
            "multiple_choice_acc",
            metrics.get("Multiple Choice Accuracy"),
            metrics.get("Multiple Choice Correct", 0),
            metrics.get("Multiple Choice Samples", 0),
        )
        log_accuracy_row(
            logger,
            "yes_no_acc",
            metrics.get("Yes/No Accuracy"),
            metrics.get("Yes/No Correct", 0),
            metrics.get("Yes/No Samples", 0),
        )
        open_sample_count = metrics.get("Open Samples", 0)
        log_accuracy_row(
            logger,
            "open_exact_match_acc",
            metrics.get("Open Exact Match Accuracy"),
            metrics.get("Open Exact Match Correct", 0),
            open_sample_count,
        )
        log_summary_row(
            logger,
            "open_token_f1",
            metrics.get("Open Token-level F1"),
            "(0 samples)" if int(open_sample_count or 0) == 0 else "",
        )
        metric_groups = metrics.get("Metric Groups")
        if isinstance(metric_groups, dict):
            logger.info("VQA metric groups:")
            for group_name in ["total", "closed", "yes_no", "open"]:
                group = metric_groups.get(group_name)
                if not isinstance(group, dict):
                    continue
                logger.info("  [%s]", group_name)
                log_accuracy_row(
                    logger,
                    "normalized_acc",
                    group.get("normalized_accuracy"),
                    group.get("correct", 0),
                    group.get("samples", 0),
                )
                log_accuracy_row(
                    logger,
                    "exact_match_acc",
                    group.get("exact_match_accuracy"),
                    group.get("exact_match_correct", 0),
                    group.get("samples", 0),
                )
                log_summary_row(logger, "token_f1", group.get("token_f1"))

    logger.info("Text metrics%s:", " (secondary for VQA; best for open-ended answers)" if task == "vqa" else "")
    for key in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"]:
        if key in metrics:
            log_summary_row(logger, key, metrics.get(key))
    bertscore = metrics.get("BERTScore")
    if isinstance(bertscore, dict):
        log_summary_row(logger, "BERTScore_F1", bertscore.get("F1"))
        log_summary_row(logger, "BERTScore_Precision", bertscore.get("Precision"))
        log_summary_row(logger, "BERTScore_Recall", bertscore.get("Recall"))
    elif "BERTScore" in metrics:
        log_summary_row(logger, "BERTScore", bertscore, "(not computed)")
    for key in ["CIDEr", "SPICE"]:
        if key in metrics:
            note = "(requires official pycocoevalcap)" if metrics.get(key) is None else ""
            log_summary_row(logger, key, metrics.get(key), note)

    slice_info = metrics.get("slice_inference")
    if isinstance(slice_info, dict):
        logger.info("Slice inference:")
        for key in ["num_slices", "slice_strategy", "view", "inference_mode"]:
            log_summary_row(logger, key, slice_info.get(key))

    runtime = metrics.get("Runtime")
    if isinstance(runtime, dict):
        logger.info("Runtime:")
        for key in [
            "model_load_time_sec",
            "total_inference_time_sec",
            "avg_inference_latency_sec",
            "throughput_samples_per_sec",
            "total_generated_tokens",
            "avg_generated_tokens",
        ]:
            log_summary_row(logger, key, runtime.get(key))

    params = metrics.get("Parameters")
    if isinstance(params, dict):
        logger.info("Model:")
        for key in ["total_params", "trainable_params", "parameter_dtype"]:
            log_summary_row(logger, key, params.get(key))

    flops = metrics.get("FLOPs")
    if isinstance(flops, dict):
        logger.info("Compute estimate:")
        for key in ["approx_total_flops", "approx_flops_per_sample"]:
            log_summary_row(logger, key, flops.get(key))


def first_dict_value(items: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    for item in items:
        value = item.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def aggregate_parallel_flops(
    shard_metrics: Sequence[Dict[str, Any]],
    parameters: Dict[str, Any],
    total_generated_tokens: int,
    successful_samples: int,
) -> Optional[Dict[str, Any]]:
    flops_items = [item.get("FLOPs") for item in shard_metrics if isinstance(item.get("FLOPs"), dict)]
    total_flops_values = [
        float(item.get("approx_total_flops"))
        for item in flops_items
        if item.get("approx_total_flops") is not None
    ]
    total_macs_values = [
        float(item.get("approx_total_macs"))
        for item in flops_items
        if item.get("approx_total_macs") is not None
    ]
    token_values = [
        int(item.get("total_generated_tokens"))
        for item in flops_items
        if item.get("total_generated_tokens") is not None
    ]

    if total_flops_values:
        approx_total_flops = int(round(sum(total_flops_values)))
    else:
        total_params = parameters.get("total_params") if isinstance(parameters, dict) else None
        if total_params is None or total_generated_tokens <= 0:
            return None
        approx_total_flops = int(2 * int(total_params) * int(total_generated_tokens))

    approx_total_macs = int(round(sum(total_macs_values))) if total_macs_values else int(approx_total_flops // 2)
    total_tokens = sum(token_values) if token_values else int(total_generated_tokens)
    method = next(
        (
            str(item.get("method"))
            for item in flops_items
            if item.get("method")
        ),
        "approx_generation_flops = 2 * total_parameters * generated_tokens",
    )
    return {
        "method": method,
        "approx_total_flops": approx_total_flops,
        "approx_total_macs": approx_total_macs,
        "approx_flops_per_sample": approx_total_flops / successful_samples if successful_samples else None,
        "total_generated_tokens": total_tokens,
    }


def run_parallel_eval(
    args: argparse.Namespace,
    config_path: Path,
    config: Dict[str, Any],
    task: str,
    split: str,
    sample_value: Any,
    sample_label: str,
    slice_config: SliceInferenceConfig,
    output_dir: Path,
    schema_info: Dict[str, Any],
    samples: Sequence[EvalSample],
    logger: logging.Logger,
    overall_start: float,
) -> int:
    devices = parse_cuda_device_list(config)
    num_workers = min(len(devices), len(samples))
    devices = devices[:num_workers]
    shard_root = output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    logger.info("Parallel eval enabled: %d workers over devices %s", num_workers, ",".join(devices))
    processes: List[Tuple[int, str, Path, subprocess.Popen]] = []
    for shard_index, device in enumerate(devices):
        safe_device = safe_filename(device)
        shard_dir = shard_root / f"shard_{shard_index:02d}_gpu_{safe_device}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--task",
            task,
            "--sample",
            str(sample_value),
            "--output-dir",
            str(shard_dir),
            "--num-shards",
            str(num_workers),
            "--shard-index",
            str(shard_index),
            "--no-parallel",
            "--skip-metrics",
            "--num_slices",
            str(slice_config.num_slices),
            "--slice_strategy",
            str(slice_config.slice_strategy),
            "--view",
            str(slice_config.view),
            "--inference_mode",
            str(slice_config.inference_mode),
        ]
        if task == "cap":
            command.extend(["--split", split])
        if task == "vqa":
            command.extend(["--vqa-mode", normalize_vqa_eval_mode(config.get("vqa_eval_mode", "open"))])
        if args.dry_run:
            command.append("--dry-run")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        env["PYTHONUNBUFFERED"] = "1"
        logger.info("Starting shard %d/%d on physical GPU(s) %s -> %s", shard_index, num_workers, device, shard_dir)
        logger.info("Shard command: %s", " ".join(shlex.quote(part) for part in command))
        processes.append(
            (
                shard_index,
                device,
                shard_dir,
                subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=env),
            )
        )

    failed = []
    for shard_index, device, shard_dir, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((shard_index, device, return_code, shard_dir))
    if failed:
        for shard_index, device, return_code, shard_dir in failed:
            logger.error("Shard %d on GPU %s failed with code %s. See %s", shard_index, device, return_code, shard_dir)
        return 1

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    shard_metrics: List[Dict[str, Any]] = []
    for shard_index, device, shard_dir, _ in processes:
        rows.extend(read_jsonl(shard_dir / "predictions.jsonl"))
        errors.extend(read_jsonl(shard_dir / "errors.jsonl"))
        metrics_path = shard_dir / "metrics.json"
        if metrics_path.exists():
            shard_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            append_jsonl(handle, row)
    with (output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for row in errors:
            append_jsonl(handle, row)
    write_previews(output_dir, rows[: int((config.get("logging") or {}).get("preview_samples", 3))], task)

    total_generated_tokens = sum(int(row.get("generated_tokens") or 0) for row in rows)
    inference_times = [float(row.get("inference_time") or 0.0) for row in rows]
    parallel_parameters = first_dict_value(shard_metrics, "Parameters")
    parallel_flops = aggregate_parallel_flops(
        shard_metrics=shard_metrics,
        parameters=parallel_parameters,
        total_generated_tokens=total_generated_tokens,
        successful_samples=len(rows),
    )
    metrics_config = dict(config.get("metrics") or {})
    if getattr(args, "skip_metrics", False):
        skip_runtime = {
            "requested_samples": len(samples),
            "successful_samples": len(rows),
            "failed_samples": len(errors),
            "parallel_workers": num_workers,
            "parallel_devices": devices,
            "total_runtime_sec": time.perf_counter() - overall_start,
        }
        skip_benchmark = {
            "runtime": skip_runtime,
            "parallel_eval": True,
            "shards": [str(p[2]) for p in processes],
        }
        metrics = {
            "task": task,
            "skip_metrics": True,
            "num_successful_samples": len(rows),
            "num_requested_samples": len(samples),
            "num_failed_samples": len(errors),
            "slice_inference": slice_inference_config_to_dict(slice_config),
            "dataset_schema": schema_info,
            "Parameters": parallel_parameters,
            "FLOPs": parallel_flops,
            "Runtime": skip_runtime,
        }
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "benchmark.json", skip_benchmark)
        write_eval_exports(output_dir, task, rows, metrics, config, logger)
        logger.info("Saved predictions: %s", output_dir / "predictions.jsonl")
        logger.info("Saved metrics: %s", output_dir / "metrics.json")
        logger.info("Saved benchmark: %s", output_dir / "benchmark.json")
        logger.info("Saved errors: %s", output_dir / "errors.jsonl")
        logger.info("Skipped metric computation for worker shard.")
        return 0

    metrics = compute_task_metrics(task, rows, metrics_config, logger)
    runtime = {
        "requested_samples": len(samples),
        "successful_samples": len(rows),
        "failed_samples": len(errors),
        "parallel_workers": num_workers,
        "parallel_devices": devices,
        "total_inference_time_sec": float(sum(inference_times)),
        "total_runtime_sec": time.perf_counter() - overall_start,
        "avg_inference_latency_sec": float(statistics.mean(inference_times)) if inference_times else None,
        "p50_inference_latency_sec": percentile(inference_times, 0.50),
        "p95_inference_latency_sec": percentile(inference_times, 0.95),
        "throughput_samples_per_sec": len(rows) / max(time.perf_counter() - overall_start, 1e-9),
        "total_generated_tokens": total_generated_tokens,
        "avg_generated_tokens": total_generated_tokens / len(rows) if rows else None,
    }
    metrics.update(
        {
            "task": task,
            "parallel_eval": True,
            "parallel_workers": num_workers,
            "parallel_devices": devices,
            "shard_metrics": shard_metrics,
            "slice_inference": slice_inference_config_to_dict(slice_config),
            "dataset_schema": schema_info,
            "num_requested_samples": len(samples),
            "num_failed_samples": len(errors),
            "Parameters": parallel_parameters,
            "FLOPs": parallel_flops,
            "Runtime": runtime,
        }
    )
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "benchmark.json", {"runtime": runtime, "parallel_eval": True, "shards": [str(p[2]) for p in processes]})
    write_eval_exports(output_dir, task, rows, metrics, config, logger)
    logger.info("Parallel eval finished. Merged predictions: %s", output_dir / "predictions.jsonl")
    logger.info("Merged errors: %s", output_dir / "errors.jsonl")
    logger.info("Merged metrics: %s", output_dir / "metrics.json")
    log_metric_summary(metrics, logger)
    return 0


def build_output_dir(
    task: str,
    output_root: Path,
    split: str,
    sample_label: str,
    slice_config: Optional[SliceInferenceConfig] = None,
    override: Optional[str] = None,
    vqa_eval_mode: Optional[str] = None,
) -> Path:
    if override:
        path = Path(os.path.expandvars(os.path.expanduser(override)))
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    slice_label = ""
    if slice_config is not None:
        slice_label = f"_S{slice_config.num_slices}_{slice_config.view}_{slice_config.inference_mode}"
    if task == "cap":
        return output_root / f"EVAL_CAP_{split}_{sample_label}{slice_label}"
    mode_label = f"{normalize_vqa_eval_mode(vqa_eval_mode or 'open')}_"
    return output_root / f"EVAL_VQA_{mode_label}{sample_label}{slice_label}"


def run(args: argparse.Namespace) -> int:
    overall_start = time.perf_counter()
    config_path = resolve_config_path(args.config, Path.cwd())
    if config_path is None:
        raise ValueError("--config is required")
    config_path = config_path.resolve()
    config = load_config(config_path)
    task = infer_task_name(args.task, config, config_path)
    if task == "vqa":
        if getattr(args, "close_ended", False) and getattr(args, "open_ended", False):
            raise ValueError("Use only one of --close_ended or --open_ended.")
        if getattr(args, "close_ended", False):
            config["vqa_eval_mode"] = "closed"
        elif getattr(args, "open_ended", False):
            config["vqa_eval_mode"] = "open"
        elif getattr(args, "vqa_eval_mode", None) is not None:
            config["vqa_eval_mode"] = normalize_vqa_eval_mode(args.vqa_eval_mode)
        else:
            config["vqa_eval_mode"] = normalize_vqa_eval_mode(config.get("vqa_eval_mode", config.get("close_ended", "open")))
    slice_config = build_slice_inference_config(config, args)
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
    output_dir = build_output_dir(
        task,
        output_root,
        split,
        sample_label,
        slice_config,
        args.output_dir,
        config.get("vqa_eval_mode") if task == "vqa" else None,
    )
    logging_config = config.get("logging") or {}
    logger = setup_logging(output_dir, bool(logging_config.get("verbose", True)))
    apply_cuda_visible_devices_config(config, logger)

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
            "slice_inference": slice_inference_config_to_dict(slice_config),
            "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        }
    )
    save_yaml_or_json(output_dir / "run_config.yaml", runtime_config)

    package_status = get_package_availability()
    logger.info("Task name: %s", task)
    if task == "vqa":
        logger.info("VQA eval mode: %s", config.get("vqa_eval_mode"))
    logger.info("Command: %s", runtime_config["command"])
    logger.info("Config path: %s", config_path)
    logger.info("Dataset path: %s", dataset_path)
    logger.info("Image root: %s", image_root)
    logger.info("Split: %s", split if task == "cap" else "N/A")
    logger.info("Sample count: %s", sample_label)
    logger.info("Slice inference: %s", json.dumps(slice_inference_config_to_dict(slice_config), ensure_ascii=False))
    logger.info("%-32s %s", "slice_inference.requested_num_slices", slice_config.num_slices)
    logger.info("%-32s %s", "slice_inference.slice_strategy", slice_config.slice_strategy)
    logger.info("%-32s %s", "slice_inference.view", slice_config.view)
    logger.info("%-32s %s", "slice_inference.inference_mode", slice_config.inference_mode)
    logger.info("Result directory: %s", output_dir)
    logger.info("Log file: %s", output_dir / "log.txt")
    logger.info("Metric packages: %s", json.dumps(package_status, ensure_ascii=False))

    if task == "cap":
        samples, schema_info = load_caption_samples(config, dataset_path, image_root, split, max_samples, logger)
    else:
        samples, schema_info = load_vqa_samples(config, dataset_path, image_root, max_samples, logger)
    logger.info("Resolved sample count: %d", len(samples))
    log_sample_image_path_health(samples, logger)
    if task == "vqa":
        log_vqa_mode_constraints(samples, logger)

    if should_run_parallel(config, args, samples):
        return run_parallel_eval(
            args=args,
            config_path=config_path,
            config=config,
            task=task,
            split=split,
            sample_value=sample_value,
            sample_label=sample_label,
            slice_config=slice_config,
            output_dir=output_dir,
            schema_info=schema_info,
            samples=samples,
            logger=logger,
            overall_start=overall_start,
        )

    shard_index = int(getattr(args, "shard_index", 0) or 0)
    num_shards = int(getattr(args, "num_shards", 1) or 1)
    if num_shards > 1:
        original_count = len(samples)
        samples = apply_sample_shard(samples, shard_index, num_shards)
        logger.info(
            "Shard selection: shard_index=%d num_shards=%d selected=%d/%d",
            shard_index,
            num_shards,
            len(samples),
            original_count,
        )

    if args.dry_run:
        dry_benchmark = build_benchmark({}, [], 0, 0, 0, len(samples), time.perf_counter() - overall_start, 0.0)
        metrics = {
            "task": task,
            "num_successful_samples": 0,
            "dry_run": True,
            "slice_inference": slice_inference_config_to_dict(slice_config),
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
        write_eval_exports(output_dir, task, [], metrics, config, logger)
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
        slice_config=slice_config,
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
    if getattr(args, "skip_metrics", False):
        metrics = {
            "task": task,
            "skip_metrics": True,
            "num_successful_samples": len(rows),
            "num_requested_samples": len(samples),
            "num_failed_samples": len(errors),
            "slice_inference": slice_inference_config_to_dict(slice_config),
            "dataset_schema": schema_info,
            "Parameters": benchmark["parameters"],
            "FLOPs": benchmark["flops"],
            "Runtime": benchmark["runtime"],
        }
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "benchmark.json", benchmark)
        write_eval_exports(output_dir, task, rows, metrics, config, logger)
        logger.info("Saved predictions: %s", output_dir / "predictions.jsonl")
        logger.info("Saved metrics: %s", output_dir / "metrics.json")
        logger.info("Saved benchmark: %s", output_dir / "benchmark.json")
        logger.info("Saved errors: %s", output_dir / "errors.jsonl")
        logger.info("Skipped full metric computation for this shard/run.")
        return 0

    metrics = compute_task_metrics(task, rows, metrics_config, logger)
    metrics.update(
        {
            "task": task,
            "slice_inference": slice_inference_config_to_dict(slice_config),
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
    write_eval_exports(output_dir, task, rows, metrics, config, logger)
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
    parser.add_argument(
        "--vqa-mode",
        "--vqa_eval_mode",
        dest="vqa_eval_mode",
        choices=["open", "closed"],
        default=None,
        help="VQA prompt/eval mode. open hides answer choices; closed includes choices and scores choice accuracy.",
    )
    parser.add_argument(
        "--close_ended",
        "--close-ended",
        dest="close_ended",
        action="store_true",
        default=False,
        help="Med3DVLM-compatible alias for --vqa-mode closed.",
    )
    parser.add_argument(
        "--open_ended",
        "--open-ended",
        dest="open_ended",
        action="store_true",
        default=False,
        help="Explicitly run VQA as open-ended, hiding answer choices.",
    )
    parser.add_argument(
        "--num_slices",
        "--num_slice",
        "--num-slices",
        "--num-slice",
        dest="num_slices",
        default=None,
        help="Number of rule-selected slices from each 3D volume, or auto/max/all for the per-sample maximum.",
    )
    parser.add_argument(
        "--slice_strategy",
        "--slice-strategy",
        dest="slice_strategy",
        choices=["middle", "uniform", "center_uniform"],
        default=None,
        help="Rule-based slice selection strategy. Use center_uniform for N slices around the middle slice.",
    )
    parser.add_argument(
        "--view",
        choices=["axial", "sagittal", "coronal"],
        default=None,
        help="3D view axis used for rule-based slicing. Default: axial.",
    )
    parser.add_argument(
        "--inference_mode",
        "--inference-mode",
        dest="inference_mode",
        choices=["montage", "independent"],
        default=None,
        help="Run one montage image or run each selected slice independently. Default: montage.",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Internal/advanced: split selected samples into N shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Internal/advanced: run only this shard index.")
    parser.add_argument("--no-parallel", action="store_true", help="Disable config parallel_eval orchestration.")
    parser.add_argument("--skip-metrics", action="store_true", help="Internal/advanced: write predictions and benchmark only.")
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
