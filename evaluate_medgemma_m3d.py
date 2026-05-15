#!/usr/bin/env python3
"""Rule-based MedGemma baseline for M3D-Cap and M3D-VQA.

This entrypoint evaluates MedGemma on 3D medical volumes by sampling fixed 2D
slices and passing those slices as multi-image input. It is intentionally a
baseline: no adaptive slice selection, masks, entropy scoring, or model-based
slice selection are used.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import math
import os
import re
import string
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from evaluate import (
    CHOICE_LETTERS,
    PROJECT_ROOT,
    append_jsonl,
    bertscore_safe,
    cider_safe,
    collect_model_stats,
    corpus_bleu_scores,
    detect_caption_schema,
    detect_vqa_schema,
    get_requested_device,
    get_torch_dtype,
    load_config,
    meteor_score_safe,
    normalize_array_to_uint8,
    package_available,
    read_text_maybe_path,
    render_choices,
    resolve_relative_existing,
    rouge_scores,
    spice_safe,
    to_jsonable,
    validate_local_model_path,
    write_json,
)


@dataclass
class CapSample:
    sample_id: str
    image_path: str
    ground_truth: str
    split: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VqaSample:
    sample_id: str
    image_path: str
    question: str
    ground_truth: str
    answer_choice: str = ""
    question_type: str = ""
    choices: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelBundle:
    model: Any
    processor: Any
    tokenizer: Any
    torch: Any
    device: Any
    dtype: Any


def setup_run_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("medgemma_m3d_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def resolve_path(value: str, base: Optional[Path] = None) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if path.is_absolute():
        return path
    if base is not None:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def infer_image_roots(args: argparse.Namespace) -> List[Path]:
    roots: List[Path] = []
    if args.image_root:
        roots.append(resolve_path(args.image_root))
    if args.cap_json:
        cap_path = resolve_path(args.cap_json)
        roots.extend([cap_path.parent.parent, cap_path.parent])
    if args.vqa_csv:
        vqa_path = resolve_path(args.vqa_csv)
        roots.extend([vqa_path.parent.parent, vqa_path.parent])
    roots.extend([PROJECT_ROOT, Path.cwd()])

    unique: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def load_cap_samples(
    cap_json: Path,
    split: str,
    max_samples: Optional[int],
    image_roots: Sequence[Path],
    logger: logging.Logger,
) -> Tuple[List[CapSample], Dict[str, Any]]:
    with cap_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        available_splits = [str(key) for key in data.keys()]
        if split not in data:
            raise ValueError(f"Split {split!r} not found in CAP json. Available: {available_splits}")
        rows = data[split]
    elif isinstance(data, list):
        available_splits = []
        rows = data
    else:
        raise ValueError("CAP json must be a split mapping or a list of records.")

    if not isinstance(rows, list):
        raise ValueError(f"CAP split {split!r} must contain a list.")

    schema = detect_caption_schema(rows)
    if not schema.get("image") or not schema.get("text"):
        raise ValueError(f"Could not detect CAP schema from keys: {schema}")

    selected = rows[:max_samples] if max_samples is not None else rows
    roots = list(image_roots) + [cap_json.parent, cap_json.parent.parent]
    samples: List[CapSample] = []
    for idx, row in enumerate(selected):
        if not isinstance(row, dict):
            continue
        image_value = str(row.get(schema["image"], "")).strip()
        image_path = resolve_relative_existing(image_value, roots)
        gt, gt_path, gt_error = read_text_maybe_path(row.get(schema["text"]), roots)
        raw_id = row.get(schema["id"]) if schema.get("id") else None
        samples.append(
            CapSample(
                sample_id=str(raw_id or f"{split}_{idx:06d}"),
                image_path=str(image_path),
                ground_truth=gt,
                split=split,
                meta={
                    "row_index": idx,
                    "raw_image": image_value,
                    "ground_truth_path": gt_path,
                    "ground_truth_error": gt_error,
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
    logger.info("CAP schema: %s", json.dumps(schema_info, ensure_ascii=False))
    return samples, schema_info


def detect_question_type(row: Dict[str, Any], choices: Dict[str, str], answer: str) -> str:
    raw_type = str(row.get("Question Type", row.get("question_type", ""))).strip().lower()
    if choices:
        return "multiple_choice"
    normalized_answer = normalize_eval_text(answer)
    if normalized_answer in {"yes", "no"}:
        return "yes_no"
    if "yes" in raw_type or "binary" in raw_type:
        return "yes_no"
    if "choice" in raw_type or "closed" in raw_type:
        return "closed"
    return "open"


def load_vqa_samples(
    vqa_csv: Path,
    max_samples: Optional[int],
    image_roots: Sequence[Path],
    logger: logging.Logger,
) -> Tuple[List[VqaSample], Dict[str, Any]]:
    with vqa_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        schema = detect_vqa_schema(fieldnames, {})
        missing = [key for key in ("image", "question", "answer") if not schema.get(key)]
        if missing:
            raise ValueError(f"Could not detect VQA columns {missing}. Fieldnames: {fieldnames}")
        rows = list(reader)

    selected = rows[:max_samples] if max_samples is not None else rows
    roots = list(image_roots) + [vqa_csv.parent, vqa_csv.parent.parent]
    samples: List[VqaSample] = []
    for idx, row in enumerate(selected):
        image_value = str(row.get(schema["image"] or "", "")).strip()
        image_path = resolve_relative_existing(image_value, roots)
        question = str(row.get(schema["question"] or "", "")).strip()
        answer = str(row.get(schema["answer"] or "", "")).strip()
        answer_choice = str(row.get(schema["answer_choice"] or "", "")).strip() if schema.get("answer_choice") else ""
        choices: Dict[str, str] = {}
        for letter in CHOICE_LETTERS:
            column = schema.get(f"choice_{letter.lower()}")
            if column:
                value = str(row.get(column, "")).strip()
                if value:
                    choices[letter] = value
        if not answer and answer_choice in choices:
            answer = choices[answer_choice]
        raw_id = row.get(schema["id"]) if schema.get("id") else None
        samples.append(
            VqaSample(
                sample_id=str(raw_id or f"vqa_{idx:06d}"),
                image_path=str(image_path),
                question=question,
                ground_truth=answer,
                answer_choice=answer_choice,
                choices=choices,
                question_type=detect_question_type(row, choices, answer),
                meta={"row_index": idx, "raw_image": image_value},
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


def load_volume_or_image(path: Path) -> Tuple[Any, bool]:
    from PIL import Image  # type: ignore

    lowered = str(path).lower()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if lowered.endswith(".npy"):
        if not package_available("numpy"):
            raise ImportError("numpy is required to read .npy volumes.")
        import numpy as np  # type: ignore

        return np.load(path), True
    if lowered.endswith(".nii") or lowered.endswith(".nii.gz"):
        if not package_available("nibabel"):
            raise ImportError("nibabel is required to read NIfTI volumes.")
        import nibabel as nib  # type: ignore

        return nib.load(str(path)).get_fdata(), True
    return Image.open(path).convert("RGB"), False


def squeeze_volume(array: Any) -> Any:
    import numpy as np  # type: ignore

    arr = np.asarray(array)
    arr = np.squeeze(arr)
    while arr.ndim > 3:
        arr = np.take(arr, arr.shape[0] // 2, axis=0)
        arr = np.squeeze(arr)
    return arr


def plane_axis(plane: str) -> int:
    return {"axial": 0, "coronal": 1, "sagittal": 2}[plane]


def uniform_indices(depth: int, count: int) -> List[int]:
    if count <= 1:
        return [depth // 2]
    target_count = min(count, depth)
    step = (depth - 1) / max(target_count - 1, 1)
    return [max(0, min(depth - 1, int(round(step * idx)))) for idx in range(target_count)]


def center_uniform_indices(depth: int, count: int) -> List[int]:
    if count <= 1:
        return [depth // 2]
    target_count = min(count, depth)
    center = depth // 2
    start = center - (target_count // 2)
    end = start + target_count
    if start < 0:
        start = 0
        end = target_count
    if end > depth:
        end = depth
        start = max(0, end - target_count)
    return list(range(start, end))


def choose_slice_indices(depth: int, num_slices: int, strategy: str) -> List[int]:
    if depth <= 0:
        raise ValueError("Cannot sample slices from an empty axis.")
    if strategy == "middle" or num_slices <= 1:
        return [depth // 2]
    if strategy == "uniform":
        return uniform_indices(depth, num_slices)
    if strategy == "center_uniform":
        return center_uniform_indices(depth, num_slices)
    raise ValueError(f"Unsupported slice strategy: {strategy}")


def slice_to_rgb(slice_array: Any) -> Image.Image:
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    arr = np.asarray(slice_array)
    arr = np.squeeze(arr)
    while arr.ndim > 2:
        arr = np.take(arr, arr.shape[0] // 2, axis=0)
        arr = np.squeeze(arr)
    return Image.fromarray(normalize_array_to_uint8(arr), mode="L").convert("RGB")


def sample_slices(path: Path, num_slices: int, strategy: str, plane: str) -> Tuple[List[Image.Image], List[int]]:
    data, is_volume = load_volume_or_image(path)
    if not is_volume:
        if num_slices > 1:
            raise ValueError(f"Cannot sample {num_slices} slices from a 2D image: {path}")
        return [data.convert("RGB")], [0]

    arr = squeeze_volume(data)
    if arr.ndim == 2:
        return [slice_to_rgb(arr)], [0]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported volume shape after squeeze: {arr.shape}")

    axis = plane_axis(plane)
    indices = choose_slice_indices(int(arr.shape[axis]), num_slices, strategy)
    images = [slice_to_rgb(arr.take(index, axis=axis)) for index in indices]
    return images, indices


def grid_shape(count: int) -> Tuple[int, int]:
    cols = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / max(cols, 1)))
    return rows, cols


def build_slice_grid(images: Sequence[Image.Image], grid_size: int = 896) -> Image.Image:
    from PIL import Image  # type: ignore

    rows, cols = grid_shape(len(images))
    tile_w = max(grid_size // max(cols, 1), 1)
    tile_h = max(grid_size // max(rows, 1), 1)
    canvas = Image.new("RGB", (grid_size, grid_size), color=(0, 0, 0))
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        tile = image.convert("RGB").resize((tile_w, tile_h), resampling)
        canvas.paste(tile, (col * tile_w, row * tile_h))
    return canvas


def image_tokens(count: int) -> str:
    return " ".join("<start_of_image>" for _ in range(max(count, 1)))


def cap_prompt(num_images: int, num_slices: int, view: str, is_montage: bool) -> str:
    if is_montage:
        body = (
            f"This image is a montage of {num_slices} {view} slices sampled from a 3D medical volume, "
            "ordered from first to last slice. Generate a concise radiology-style caption describing "
            "only visible imaging findings. Do not infer patient history or findings not visible in the image."
        )
    else:
        body = (
            f"This image shows {num_slices} {view} slice(s) sampled from a 3D medical volume. Generate "
            "a concise radiology-style caption describing only visible imaging findings. Do not infer "
            "patient history or findings not visible in the image."
        )
    return f"{image_tokens(num_images)}\n{body}"


def vqa_prompt(sample: VqaSample, num_images: int, num_slices: int, view: str, is_montage: bool) -> str:
    if is_montage:
        prefix = (
            f"This image is a montage of {num_slices} {view} slices sampled from a 3D medical volume, "
            "ordered from first to last slice. Answer based only on visible findings."
        )
    else:
        prefix = (
            f"This image shows {num_slices} {view} slice(s) sampled from a 3D medical volume. "
            "Answer the question based only on visible findings."
        )
    if sample.choices:
        choices = render_choices(sample.choices)
        body = (
            f"{prefix}\n\n"
            f"Question: {sample.question}\n\n"
            f"{choices}\n\n"
            "Return only the final option label. Do not explain."
        )
    elif sample.question_type == "yes_no" or normalize_eval_text(sample.ground_truth) in {"yes", "no"}:
        body = (
            f"{prefix}\n\n"
            f"Question: {sample.question}\n\n"
            "Return only yes or no. Do not explain."
        )
    else:
        body = (
            f"{prefix}\n\n"
            f"Question: {sample.question}\n\n"
            "Answer concisely."
        )
    return f"{image_tokens(num_images)}\n{body}"


def clean_prediction(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^```(?:\w+)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"^(?:answer|report|caption)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def normalize_eval_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def normalize_yes_no(text: str) -> Optional[str]:
    normalized = normalize_eval_text(text)
    if normalized.startswith("yes") or re.search(r"\byes\b", normalized):
        return "yes"
    if normalized.startswith("no") or re.search(r"\bno\b", normalized):
        return "no"
    return None


def extract_option_label(text: str, choices: Dict[str, str]) -> str:
    raw = str(text).strip()
    match = re.search(r"\b([A-Ea-e])\b", raw)
    if match:
        letter = match.group(1).upper()
        if letter in choices:
            return letter
    normalized = normalize_eval_text(raw)
    for letter, choice_text in choices.items():
        if normalize_eval_text(choice_text) == normalized:
            return letter
    return normalized.upper() if len(normalized) == 1 else raw


def vqa_correct(sample: VqaSample, prediction: str) -> Tuple[Optional[bool], str, str]:
    if sample.choices:
        pred_label = extract_option_label(prediction, sample.choices)
        gt_label = sample.answer_choice.upper().strip()
        if not gt_label:
            for letter, choice_text in sample.choices.items():
                if normalize_eval_text(choice_text) == normalize_eval_text(sample.ground_truth):
                    gt_label = letter
                    break
        if gt_label:
            return pred_label == gt_label, pred_label, gt_label
        return normalize_eval_text(prediction) == normalize_eval_text(sample.ground_truth), pred_label, sample.ground_truth

    gt_yes_no = normalize_yes_no(sample.ground_truth)
    if sample.question_type == "yes_no" or gt_yes_no in {"yes", "no"}:
        pred_yes_no = normalize_yes_no(prediction)
        return pred_yes_no == gt_yes_no, pred_yes_no or normalize_eval_text(prediction), gt_yes_no or ""

    return None, normalize_eval_text(prediction), normalize_eval_text(sample.ground_truth)


def vqa_token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_eval_text(prediction).split()
    gt_tokens = normalize_eval_text(ground_truth).split()
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
    return 2.0 * precision * recall / (precision + recall)


def bertscore_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "bertscore_lang": args.bertscore_lang,
        "bertscore_model_type": args.bertscore_model_type,
        "bertscore_rescale_with_baseline": args.bertscore_rescale_with_baseline,
    }


def sample_text_metrics(
    prediction: str,
    reference: str,
    logger: logging.Logger,
    use_bertscore: bool,
    metrics_config: Dict[str, Any],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics.update(corpus_bleu_scores([prediction], [reference]))
    metrics.update(rouge_scores([prediction], [reference]))
    metrics["METEOR"] = meteor_score_safe([prediction], [reference], logger)
    if use_bertscore:
        metrics["BERTScore"] = bertscore_safe([prediction], [reference], metrics_config, logger)
    return metrics


def aggregate_text_metrics(
    rows: Sequence[Dict[str, Any]],
    logger: logging.Logger,
    use_bertscore: bool,
    metrics_config: Dict[str, Any],
) -> Dict[str, Any]:
    predictions = [str(row.get("prediction", "")) for row in rows]
    references = [str(row.get("ground_truth", "")) for row in rows]
    metrics: Dict[str, Any] = {"num_samples": len(rows)}
    metrics.update(corpus_bleu_scores(predictions, references))
    metrics.update(rouge_scores(predictions, references))
    metrics["METEOR"] = meteor_score_safe(predictions, references, logger)
    metrics["CIDEr"] = cider_safe(predictions, references, logger)
    metrics["SPICE"] = spice_safe(predictions, references, logger)
    if use_bertscore:
        metrics["BERTScore"] = bertscore_safe(predictions, references, metrics_config, logger)
    return metrics


def get_token_id(*values: Any) -> Optional[Any]:
    for value in values:
        if value is not None:
            return value
    return None


def load_model(args: argparse.Namespace, logger: logging.Logger) -> Tuple[ModelBundle, Dict[str, Any]]:
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

    model_path = resolve_path(args.model_path)
    validate_local_model_path(model_path, args.local_files_only)
    device_name = get_requested_device(torch, args.device)
    dtype = get_torch_dtype(torch, args.dtype)

    logger.info("Loading processor from %s", model_path)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast_processor,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None and AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )

    model_kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if device_name != "cpu" and args.device_map:
        model_kwargs["device_map"] = args.device_map

    logger.info("Loading model from %s", model_path)
    if AutoModelForImageTextToText is not None:
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), **model_kwargs)
    else:
        from transformers import AutoModelForVision2Seq  # type: ignore

        model = AutoModelForVision2Seq.from_pretrained(str(model_path), **model_kwargs)
    if device_name == "cpu":
        model.to(device_name)
    model.eval()

    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device(device_name)

    stats = collect_model_stats(model, torch)
    logger.info("Model loaded. Device=%s dtype=%s", device, dtype)
    return ModelBundle(model=model, processor=processor, tokenizer=tokenizer, torch=torch, device=device, dtype=dtype), stats


def move_inputs_to_device(inputs: Any, bundle: ModelBundle) -> Any:
    torch = bundle.torch
    if hasattr(inputs, "to"):
        try:
            return inputs.to(bundle.device, dtype=bundle.dtype)
        except TypeError:
            return inputs.to(bundle.device)
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if torch.is_tensor(value) and value.is_floating_point():
                moved[key] = value.to(device=bundle.device, dtype=bundle.dtype)
            else:
                moved[key] = value.to(device=bundle.device)
        else:
            moved[key] = value
    return moved


def generation_kwargs(args: argparse.Namespace, bundle: ModelBundle, task: str) -> Dict[str, Any]:
    max_new_tokens = args.cap_max_new_tokens if task == "cap" else args.vqa_max_new_tokens
    generation_config = getattr(bundle.model, "generation_config", None)
    model_config = getattr(bundle.model, "config", None)
    eos_token_id = get_token_id(
        getattr(bundle.tokenizer, "eos_token_id", None),
        getattr(generation_config, "eos_token_id", None),
        getattr(model_config, "eos_token_id", None),
    )
    pad_token_id = get_token_id(
        getattr(bundle.tokenizer, "pad_token_id", None),
        getattr(generation_config, "pad_token_id", None),
        getattr(model_config, "pad_token_id", None),
        eos_token_id,
    )
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
    }
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def generate(bundle: ModelBundle, images: Sequence[Image.Image], prompt: str, kwargs: Dict[str, Any]) -> Tuple[str, float, int]:
    image_input: Any = list(images)
    inputs = bundle.processor(text=prompt, images=image_input, return_tensors="pt")
    inputs = move_inputs_to_device(inputs, bundle)
    input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0

    if bundle.torch.cuda.is_available():
        bundle.torch.cuda.synchronize()
    start = time.perf_counter()
    with bundle.torch.inference_mode():
        output = bundle.model.generate(**inputs, **kwargs)
    if bundle.torch.cuda.is_available():
        bundle.torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    sequence = output[0]
    is_encoder_decoder = bool(getattr(getattr(bundle.model, "config", None), "is_encoder_decoder", False))
    if input_len and not is_encoder_decoder and int(sequence.shape[-1]) >= input_len:
        sequence = sequence[input_len:]
    generated_tokens = int(sequence.shape[-1])
    decoder = getattr(bundle.processor, "decode", None) or getattr(bundle.tokenizer, "decode", None)
    if decoder is None:
        raise RuntimeError("Neither processor nor tokenizer exposes decode().")
    return clean_prediction(decoder(sequence, skip_special_tokens=True)), elapsed, generated_tokens


def prepare_images(
    image_path: str,
    args: argparse.Namespace,
    prompt_builder: Any,
    logger: logging.Logger,
) -> Tuple[List[Image.Image], List[int], str, Optional[str]]:
    slices, indices = sample_slices(Path(image_path), args.num_slices, args.slice_strategy, args.plane)
    actual_num_slices = len(indices)
    if args.slice_grid:
        grid = build_slice_grid(slices, args.grid_size)
        prompt = prompt_builder(1, actual_num_slices, True)
        return [grid], indices, prompt, None
    prompt = prompt_builder(len(slices), actual_num_slices, False)
    return slices, indices, prompt, None


def evaluate_cap(
    samples: Sequence[CapSample],
    bundle: ModelBundle,
    args: argparse.Namespace,
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    predictions_path = output_dir / "predictions_cap.jsonl"
    use_bertscore = bool(args.bertscore)
    metrics_config = bertscore_config_from_args(args)
    with predictions_path.open("w", encoding="utf-8") as handle:
        for idx, sample in enumerate(samples):
            try:
                if sample.meta.get("ground_truth_error"):
                    raise FileNotFoundError(sample.meta["ground_truth_error"])
                images, slice_indices, prompt, _ = prepare_images(
                    sample.image_path,
                    args,
                    lambda image_count, slice_count, is_montage, a=args: cap_prompt(
                        image_count, slice_count, a.plane, is_montage
                    ),
                    logger,
                )
                actual_num_slices = len(slice_indices)
                model_input_images = len(images)
                logger.info(
                    "Slice selection | task=cap | sample_id=%s | requested_num_slices=%s | actual_num_slices=%d | model_input_images=%d | strategy=%s | plane=%s | slice_grid=%s | indices=%s",
                    sample.sample_id,
                    args.num_slices,
                    actual_num_slices,
                    model_input_images,
                    args.slice_strategy,
                    args.plane,
                    bool(args.slice_grid),
                    slice_indices,
                )
                prediction, elapsed, generated_tokens = generate(
                    bundle,
                    images,
                    prompt,
                    generation_kwargs(args, bundle, "cap"),
                )
                row = {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "num_slices": actual_num_slices,
                    "requested_num_slices": args.num_slices,
                    "model_input_images": model_input_images,
                    "slice_indices": slice_indices,
                    "slice_strategy": args.slice_strategy,
                    "plane": args.plane,
                    "slice_grid": bool(args.slice_grid),
                    "prompt": prompt,
                    "prediction": prediction,
                    "ground_truth": sample.ground_truth,
                    "inference_time": elapsed,
                    "generated_tokens": generated_tokens,
                    "metrics_per_sample": sample_text_metrics(
                        prediction, sample.ground_truth, logger, use_bertscore, metrics_config
                    ),
                }
                rows.append(row)
                append_jsonl(handle, row)
                if idx % max(args.preview_every, 1) == 0:
                    logger.info(
                        "\n[CAP SAMPLE]\nID: %s\nActual num slices: %d\nModel input images: %d\nSlice indices: %s\nGT: %s\nPR: %s",
                        sample.sample_id,
                        actual_num_slices,
                        model_input_images,
                        slice_indices,
                        sample.ground_truth,
                        prediction,
                    )
            except Exception as exc:
                error = {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(handle, error)
                logger.error("CAP sample %s failed: %s", sample.sample_id, exc, exc_info=True)

    metrics = aggregate_text_metrics(rows, logger, use_bertscore, metrics_config)
    write_json(output_dir / "metrics_cap.json", metrics)
    logger.info("Saved CAP predictions: %s", predictions_path)
    logger.info("Saved CAP metrics: %s", output_dir / "metrics_cap.json")
    return metrics


def evaluate_vqa(
    samples: Sequence[VqaSample],
    bundle: ModelBundle,
    args: argparse.Namespace,
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    open_rows: List[Dict[str, Any]] = []
    predictions_path = output_dir / "predictions_vqa.jsonl"
    use_bertscore = bool(args.bertscore)
    metrics_config = bertscore_config_from_args(args)
    correct_values: List[bool] = []
    type_correct: Dict[str, List[bool]] = {}

    with predictions_path.open("w", encoding="utf-8") as handle:
        for idx, sample in enumerate(samples):
            try:
                images, slice_indices, prompt, _ = prepare_images(
                    sample.image_path,
                    args,
                    lambda image_count, slice_count, is_montage, s=sample, a=args: vqa_prompt(
                        s, image_count, slice_count, a.plane, is_montage
                    ),
                    logger,
                )
                actual_num_slices = len(slice_indices)
                model_input_images = len(images)
                logger.info(
                    "Slice selection | task=vqa | sample_id=%s | requested_num_slices=%s | actual_num_slices=%d | model_input_images=%d | strategy=%s | plane=%s | slice_grid=%s | indices=%s",
                    sample.sample_id,
                    args.num_slices,
                    actual_num_slices,
                    model_input_images,
                    args.slice_strategy,
                    args.plane,
                    bool(args.slice_grid),
                    slice_indices,
                )
                prediction, elapsed, generated_tokens = generate(
                    bundle,
                    images,
                    prompt,
                    generation_kwargs(args, bundle, "vqa"),
                )
                correct, normalized_prediction, normalized_ground_truth = vqa_correct(sample, prediction)
                exact_match = normalized_prediction == normalized_ground_truth
                token_f1_score = vqa_token_f1(normalized_prediction, normalized_ground_truth)
                if correct is None:
                    open_rows.append(
                        {
                            "prediction": prediction,
                            "ground_truth": sample.ground_truth,
                            "exact_match": exact_match,
                            "token_f1": token_f1_score,
                        }
                    )
                else:
                    correct_values.append(bool(correct))
                    type_correct.setdefault(sample.question_type or "closed", []).append(bool(correct))
                row = {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "question": sample.question,
                    "question_type": sample.question_type,
                    "choices": sample.choices,
                    "answer_choice": sample.answer_choice,
                    "num_slices": actual_num_slices,
                    "requested_num_slices": args.num_slices,
                    "model_input_images": model_input_images,
                    "slice_indices": slice_indices,
                    "slice_strategy": args.slice_strategy,
                    "plane": args.plane,
                    "slice_grid": bool(args.slice_grid),
                    "prompt": prompt,
                    "prediction": prediction,
                    "ground_truth": sample.ground_truth,
                    "normalized_prediction": normalized_prediction,
                    "normalized_ground_truth": normalized_ground_truth,
                    "correct": correct,
                    "exact_match": exact_match,
                    "token_f1": token_f1_score,
                    "inference_time": elapsed,
                    "generated_tokens": generated_tokens,
                    "metrics_per_sample": sample_text_metrics(
                        prediction, sample.ground_truth, logger, use_bertscore, metrics_config
                    )
                    if correct is None
                    else {},
                }
                rows.append(row)
                append_jsonl(handle, row)
                if idx % max(args.preview_every, 1) == 0:
                    logger.info(
                        "\n[VQA SAMPLE]\nID: %s\nActual num slices: %d\nModel input images: %d\nSlice indices: %s\nQuestion: %s\nGT: %s\nPR: %s\nCorrect: %s",
                        sample.sample_id,
                        actual_num_slices,
                        model_input_images,
                        slice_indices,
                        sample.question,
                        sample.ground_truth,
                        prediction,
                        str(correct).lower() if correct is not None else "n/a",
                    )
            except Exception as exc:
                error = {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "question": sample.question,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(handle, error)
                logger.error("VQA sample %s failed: %s", sample.sample_id, exc, exc_info=True)

    metrics: Dict[str, Any] = {
        "num_samples": len(rows),
        "num_closed_samples": len(correct_values),
        "closed_accuracy": (sum(correct_values) / len(correct_values)) if correct_values else None,
        "exact_match_accuracy": (
            sum(bool(row.get("exact_match", False)) for row in rows) / len(rows) if rows else None
        ),
        "token_f1": (
            sum(float(row.get("token_f1", 0.0)) for row in rows) / len(rows) if rows else None
        ),
        "accuracy_by_type": {
            label: {
                "num_samples": len(values),
                "accuracy": (sum(values) / len(values)) if values else None,
            }
            for label, values in sorted(type_correct.items())
        },
        "num_open_samples": len(open_rows),
        "open_exact_match_accuracy": (
            sum(bool(row.get("exact_match", False)) for row in open_rows) / len(open_rows) if open_rows else None
        ),
        "open_token_f1": (
            sum(float(row.get("token_f1", 0.0)) for row in open_rows) / len(open_rows) if open_rows else None
        ),
    }
    if open_rows:
        metrics["open_text_metrics"] = aggregate_text_metrics(open_rows, logger, use_bertscore, metrics_config)
    else:
        metrics["open_text_metrics"] = None
    write_json(output_dir / "metrics_vqa.json", metrics)
    logger.info("Saved VQA predictions: %s", predictions_path)
    logger.info("Saved VQA metrics: %s", output_dir / "metrics_vqa.json")
    return metrics


def optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if str(value).strip().lower() in {"", "none", "null", "full"}:
        return None
    count = int(value)
    return count if count > 0 else None


def save_run_config(args: argparse.Namespace, output_dir: Path, model_stats: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "model_path": str(resolve_path(args.model_path)),
        "cap_json": str(resolve_path(args.cap_json)) if args.cap_json else None,
        "vqa_csv": str(resolve_path(args.vqa_csv)) if args.vqa_csv else None,
        "image_root": str(resolve_path(args.image_root)) if args.image_root else None,
        "task": args.task,
        "num_slices": args.num_slices,
        "slice_strategy": args.slice_strategy,
        "plane": args.plane,
        "slice_grid": bool(args.slice_grid),
        "max_samples": args.max_samples,
        "generation": {
            "temperature": 0.0,
            "do_sample": False,
            "cap_max_new_tokens": args.cap_max_new_tokens,
            "vqa_max_new_tokens": args.vqa_max_new_tokens,
        },
        "metrics": {
            "bertscore": bool(args.bertscore),
            "bertscore_lang": args.bertscore_lang,
            "bertscore_model_type": args.bertscore_model_type,
            "bertscore_rescale_with_baseline": args.bertscore_rescale_with_baseline,
        },
        "model_stats": model_stats,
        "command": " ".join(sys.argv),
    }
    write_json(output_dir / "config.json", payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MedGemma on M3D-Cap and M3D-VQA with fixed 2D slice sampling.")
    parser.add_argument("--model_path", required=True, help="Local MedGemma model directory.")
    parser.add_argument("--cap_json", default=None, help="Path to M3D_Cap.json.")
    parser.add_argument("--vqa_csv", default=None, help="Path to M3D_VQA_test5k.csv.")
    parser.add_argument("--image_root", default=None, help="Optional image root. Defaults are inferred from dataset paths.")
    parser.add_argument("--task", choices=["cap", "vqa", "both"], default="both")
    parser.add_argument("--cap_split", default="test1k", help="CAP split to evaluate.")
    parser.add_argument("--num_slices", type=int, default=9)
    parser.add_argument("--slice_strategy", choices=["middle", "uniform", "center_uniform"], default="center_uniform")
    parser.add_argument("--plane", choices=["axial", "sagittal", "coronal"], default="axial")
    parser.add_argument("--max_samples", default=None, help="Maximum samples per task, or full.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--slice_grid", action="store_true", help="Use one montage grid image instead of multi-image input.")
    parser.add_argument("--grid_size", type=int, default=896)
    parser.add_argument("--cap_max_new_tokens", type=int, default=128)
    parser.add_argument("--vqa_max_new_tokens", type=int, default=64)
    parser.add_argument("--preview_every", type=int, default=10)
    parser.add_argument("--bertscore", action="store_true", help="Compute BERTScore if bert_score and model cache are available.")
    parser.add_argument("--bertscore_lang", default="en")
    parser.add_argument("--bertscore_model_type", default="weight/roberta-large")
    parser.add_argument("--bertscore_rescale_with_baseline", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_processor", action=argparse.BooleanOptionalAction, default=False)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.task in {"cap", "both"} and not args.cap_json:
        raise ValueError("--cap_json is required when --task is cap or both.")
    if args.task in {"vqa", "both"} and not args.vqa_csv:
        raise ValueError("--vqa_csv is required when --task is vqa or both.")
    if args.num_slices < 1:
        raise ValueError("--num_slices must be >= 1.")
    if args.slice_strategy == "middle" and args.num_slices != 1:
        raise ValueError("--slice_strategy middle requires --num_slices 1. Use uniform or center_uniform for multiple slices.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        output_dir = resolve_path(args.output_dir)
        logger = setup_run_logging(output_dir)
        logger.info("Starting MedGemma M3D baseline evaluation")
        save_run_config(args, output_dir)

        image_roots = infer_image_roots(args)
        logger.info("Image roots: %s", [str(root) for root in image_roots])
        logger.info(
            "Slice config: num_slices=%s strategy=%s plane=%s slice_grid=%s",
            args.num_slices,
            args.slice_strategy,
            args.plane,
            args.slice_grid,
        )

        bundle, model_stats = load_model(args, logger)
        save_run_config(args, output_dir, model_stats)

        max_samples = optional_int(args.max_samples)
        if args.task in {"cap", "both"}:
            cap_samples, cap_schema = load_cap_samples(resolve_path(args.cap_json), args.cap_split, max_samples, image_roots, logger)
            write_json(output_dir / "schema_cap.json", cap_schema)
            evaluate_cap(cap_samples, bundle, args, output_dir, logger)

        if args.task in {"vqa", "both"}:
            vqa_samples, vqa_schema = load_vqa_samples(resolve_path(args.vqa_csv), max_samples, image_roots, logger)
            write_json(output_dir / "schema_vqa.json", vqa_schema)
            evaluate_vqa(vqa_samples, bundle, args, output_dir, logger)

        logger.info("Evaluation finished")
        return 0
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
