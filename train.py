#!/usr/bin/env python3
"""Train or fine-tune MedGemma on the local M3D CAP/VQA datasets."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import shlex
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from evaluate import (
    EvalSample,
    PROJECT_ROOT,
    collect_model_stats,
    get_package_availability,
    get_requested_device,
    get_torch_dtype,
    infer_task_name,
    load_caption_samples,
    load_config,
    load_image_as_rgb,
    load_vqa_samples,
    normalize_sample_spec,
    package_available,
    resolve_config_path,
    resolve_output_root,
    save_yaml_or_json,
    setup_logging,
    write_json,
)


TRAIN_MODE_ALIASES = {
    "lora": "lora",
    "peft": "lora",
    "adapter": "lora",
    "adapters": "lora",
    "fine_tune": "lora",
    "finetune": "lora",
    "fine-tune": "lora",
    "full": "full",
    "full_train": "full",
    "full-train": "full",
    "train_all": "full",
    "train-all": "full",
    "pipeline": "full",
    "full_pipeline": "full",
    "full-pipeline": "full",
}


def bool_config(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def optional_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or str(value).strip().lower() in {"", "null", "none"}:
        return default
    return int(value)


def normalize_string_list(value: Any, default: Sequence[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return [part for part in parts if part]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return list(default)


def normalize_train_mode(raw_mode: Any, training_config: Dict[str, Any]) -> str:
    if raw_mode is None:
        raw_mode = training_config.get("mode") or training_config.get("train_mode") or "lora"
    if bool_config(training_config.get("full_train"), False):
        raw_mode = "full"
    mode_key = str(raw_mode).strip().lower().replace(" ", "_")
    if mode_key not in TRAIN_MODE_ALIASES:
        allowed = ", ".join(sorted(set(TRAIN_MODE_ALIASES.values())))
        raise ValueError(f"Unsupported training mode {raw_mode!r}. Use one of: {allowed}.")
    return TRAIN_MODE_ALIASES[mode_key]


def command_string(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in argv)


def get_train_output_root(config: Dict[str, Any], training_config: Dict[str, Any]) -> Path:
    return resolve_output_root(training_config.get("output_root", config.get("output_root", "results")))


def build_train_output_dir(
    task: str,
    mode: str,
    output_root: Path,
    split: str,
    sample_label: str,
    training_config: Dict[str, Any],
    override: Optional[str],
) -> Path:
    if override:
        path = Path(os.path.expandvars(os.path.expanduser(override)))
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    run_name = str(training_config.get("run_name") or "").strip()
    if run_name:
        return output_root / run_name
    split_label = split or "all"
    return output_root / f"TRAIN_{task.upper()}_{mode}_{split_label}_{sample_label}"


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def package_status() -> Dict[str, bool]:
    status = get_package_availability()
    status.update(
        {
            "accelerate": package_available("accelerate"),
            "peft": package_available("peft"),
            "bitsandbytes": package_available("bitsandbytes"),
        }
    )
    return status


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def target_text_for_sample(sample: EvalSample, task: str, training_config: Dict[str, Any]) -> str:
    if task == "vqa":
        target_mode = str(training_config.get("vqa_target", "answer")).strip().lower()
        if target_mode in {"choice", "answer_choice", "letter"} and sample.answer_choice:
            return sample.answer_choice
    return sample.ground_truth


def join_prompt_and_target(prompt: str, target: str, eos_token: Optional[str], append_eos: bool) -> str:
    prompt = str(prompt).strip()
    target = str(target).strip()
    separator = "" if not prompt or prompt.endswith((" ", "\n")) else " "
    text = f"{prompt}{separator}{target}"
    if append_eos and eos_token and not text.endswith(eos_token):
        text += eos_token
    return text


def load_samples_for_task(
    task: str,
    config: Dict[str, Any],
    dataset_path: Path,
    image_root: Path,
    split: str,
    max_samples: Optional[int],
    logger: logging.Logger,
) -> Tuple[List[EvalSample], Dict[str, Any]]:
    if task == "cap":
        return load_caption_samples(config, dataset_path, image_root, split, max_samples, logger)
    return load_vqa_samples(config, dataset_path, image_root, max_samples, logger)


def filter_trainable_samples(
    samples: Sequence[EvalSample],
    task: str,
    training_config: Dict[str, Any],
) -> Tuple[List[EvalSample], List[Dict[str, Any]]]:
    kept: List[EvalSample] = []
    skipped: List[Dict[str, Any]] = []
    for sample in samples:
        reason = ""
        if sample.meta.get("ground_truth_error"):
            reason = str(sample.meta["ground_truth_error"])
        elif not target_text_for_sample(sample, task, training_config).strip():
            reason = "Target text is empty."
        elif not Path(sample.image_path).exists():
            reason = f"Image not found: {sample.image_path}"

        if reason:
            skipped.append(
                {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "reason": reason,
                }
            )
            continue
        kept.append(sample)
    return kept, skipped


class MedGemmaTrainDataset:
    def __init__(self, samples: Sequence[EvalSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EvalSample:
        return self.samples[index]


class MedGemmaDataCollator:
    def __init__(
        self,
        processor: Any,
        task: str,
        image_config: Dict[str, Any],
        training_config: Dict[str, Any],
        tokenizer: Any,
    ) -> None:
        self.processor = processor
        self.task = task
        self.image_config = image_config
        self.training_config = training_config
        self.tokenizer = tokenizer
        self.max_length = optional_int(training_config.get("max_seq_length"), None)
        self.response_only_loss = bool_config(training_config.get("response_only_loss"), True)
        self.append_eos = bool_config(training_config.get("append_eos_token"), True)
        self.label_pad_token_id = int(training_config.get("label_pad_token_id", -100))

    def _encode(self, text: Any, images: Any, padding: Any = True) -> Any:
        kwargs: Dict[str, Any] = {
            "text": text,
            "images": images,
            "return_tensors": "pt",
            "padding": padding,
        }
        if self.max_length is not None:
            kwargs["truncation"] = True
            kwargs["max_length"] = self.max_length
        return self.processor(**kwargs)

    def _prompt_length(self, prompt: str, image: Any) -> int:
        encoded = self._encode(prompt, image, padding=False)
        if "attention_mask" in encoded:
            return int(encoded["attention_mask"][0].sum().item())
        return int(encoded["input_ids"].shape[-1])

    def __call__(self, samples: Sequence[EvalSample]) -> Dict[str, Any]:
        images = [load_image_as_rgb(Path(sample.image_path), self.image_config) for sample in samples]
        eos_token = getattr(self.tokenizer, "eos_token", None)
        texts = [
            join_prompt_and_target(
                sample.prompt,
                target_text_for_sample(sample, self.task, self.training_config),
                eos_token,
                self.append_eos,
            )
            for sample in samples
        ]
        batch = self._encode(texts, images, padding=True)
        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        attention_mask = batch.get("attention_mask")
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, self.label_pad_token_id)
        elif pad_token_id is not None:
            labels = labels.masked_fill(labels == pad_token_id, self.label_pad_token_id)

        if self.response_only_loss:
            for row_index, sample in enumerate(samples):
                prompt_len = self._prompt_length(sample.prompt, images[row_index])
                prompt_len = min(prompt_len, labels.shape[-1])
                labels[row_index, :prompt_len] = self.label_pad_token_id

                has_label = bool((labels[row_index] != self.label_pad_token_id).any().item())
                if not has_label:
                    if attention_mask is not None:
                        valid_len = int(attention_mask[row_index].sum().item())
                    else:
                        valid_len = int(labels.shape[-1])
                    fallback_index = max(valid_len - 1, 0)
                    labels[row_index, fallback_index] = input_ids[row_index, fallback_index]

        batch["labels"] = labels
        return dict(batch)


def set_tokenizer_padding(tokenizer: Any, logger: logging.Logger) -> None:
    if tokenizer is None:
        return
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
            logger.info("Tokenizer pad_token was missing; using eos_token as padding.")


def maybe_enable_gradient_checkpointing(model: Any, training_config: Dict[str, Any], logger: logging.Logger) -> None:
    if not bool_config(training_config.get("gradient_checkpointing"), True):
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model_config = getattr(model, "config", None)
    if model_config is not None and hasattr(model_config, "use_cache"):
        model_config.use_cache = False


def freeze_named_parameters(model: Any, keys: Sequence[str], logger: logging.Logger, label: str) -> None:
    matched = 0
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if any(key in lowered for key in keys):
            parameter.requires_grad = False
            matched += int(parameter.numel())
    if matched:
        logger.info("Frozen %s parameters: %d", label, matched)


def apply_full_training_freezes(model: Any, training_config: Dict[str, Any], logger: logging.Logger) -> None:
    if bool_config(training_config.get("freeze_vision"), False):
        freeze_named_parameters(model, ("vision", "visual", "image_tower", "vision_tower"), logger, "vision")
    if bool_config(training_config.get("freeze_projector"), False):
        freeze_named_parameters(model, ("projector", "multi_modal_projector", "mm_projector"), logger, "projector")
    if bool_config(training_config.get("freeze_language"), False):
        freeze_named_parameters(model, ("language_model", "text_model", "model.layers", "lm_head"), logger, "language")


def get_lora_config(config: Dict[str, Any], training_config: Dict[str, Any]) -> Dict[str, Any]:
    fine_tuning = dict(config.get("fine_tuning") or {})
    lora_config = dict(fine_tuning.get("lora") or {})
    lora_config.update(dict(training_config.get("lora") or {}))
    return lora_config


def get_quantization_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(training_config.get("quantization") or {})


def load_processor_and_model(
    config: Dict[str, Any],
    training_config: Dict[str, Any],
    train_mode: str,
    model_path: Path,
    logger: logging.Logger,
) -> Tuple[Any, Any, Any, Any]:
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
    local_files_only = bool_config(config.get("local_files_only"), True)
    trust_remote_code = bool_config(config.get("trust_remote_code"), True)

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None and AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    set_tokenizer_padding(tokenizer, logger)

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }

    quantization_config = get_quantization_config(training_config)
    load_in_4bit = bool_config(quantization_config.get("load_in_4bit"), False)
    load_in_8bit = bool_config(quantization_config.get("load_in_8bit"), False)
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig  # type: ignore

        bnb_kwargs: Dict[str, Any] = {
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": load_in_8bit,
        }
        if load_in_4bit:
            bnb_kwargs.update(
                {
                    "bnb_4bit_quant_type": str(quantization_config.get("bnb_4bit_quant_type", "nf4")),
                    "bnb_4bit_use_double_quant": bool_config(
                        quantization_config.get("bnb_4bit_use_double_quant"), True
                    ),
                    "bnb_4bit_compute_dtype": dtype,
                }
            )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)

    device_map = training_config.get("device_map", config.get("device_map", "auto"))
    if device_name != "cpu" and device_map:
        model_kwargs["device_map"] = device_map

    logger.info("Loading model from %s", model_path)
    if AutoModelForImageTextToText is None:
        from transformers import AutoModelForVision2Seq  # type: ignore

        model = AutoModelForVision2Seq.from_pretrained(str(model_path), **model_kwargs)
    else:
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), **model_kwargs)

    if device_name == "cpu" and "quantization_config" not in model_kwargs:
        model.to(device_name)

    maybe_enable_gradient_checkpointing(model, training_config, logger)

    if train_mode == "lora":
        if not package_available("peft"):
            raise ImportError("peft is required for training.mode=lora. Install peft or use training.mode=full.")
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore

        if load_in_4bit or load_in_8bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool_config(training_config.get("gradient_checkpointing"), True),
            )

        lora_config = get_lora_config(config, training_config)
        target_modules = normalize_string_list(
            lora_config.get("target_modules"),
            ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        )
        peft_config = LoraConfig(
            r=int(lora_config.get("r", 16)),
            lora_alpha=int(lora_config.get("alpha", lora_config.get("lora_alpha", 32))),
            lora_dropout=float(lora_config.get("dropout", lora_config.get("lora_dropout", 0.05))),
            bias=str(lora_config.get("bias") or "none"),
            task_type=str(lora_config.get("task_type", "CAUSAL_LM")),
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)
        logger.info("LoRA enabled. Target modules: %s", ", ".join(target_modules))
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
        apply_full_training_freezes(model, training_config, logger)
        logger.info("Full training enabled for all unfrozen parameters.")

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    stats = collect_model_stats(model, torch)
    total = max(int(stats.get("total_params") or 0), 1)
    trainable = int(stats.get("trainable_params") or 0)
    logger.info("Trainable parameters: %d / %d (%.4f%%)", trainable, total, trainable * 100.0 / total)
    return processor, tokenizer, model, torch


def add_if_supported(target: Dict[str, Any], signature: inspect.Signature, name: str, value: Any) -> None:
    if name in signature.parameters and value is not None:
        target[name] = value


def build_training_arguments(
    training_config: Dict[str, Any],
    output_dir: Path,
    dtype_name: str,
    torch: Any,
    has_eval: bool,
) -> Any:
    from transformers import TrainingArguments  # type: ignore

    signature = inspect.signature(TrainingArguments.__init__)
    kwargs: Dict[str, Any] = {}

    add_if_supported(kwargs, signature, "output_dir", str(output_dir))
    add_if_supported(kwargs, signature, "overwrite_output_dir", bool_config(training_config.get("overwrite_output_dir"), False))
    add_if_supported(kwargs, signature, "per_device_train_batch_size", int(training_config.get("per_device_train_batch_size", 1)))
    add_if_supported(kwargs, signature, "per_device_eval_batch_size", int(training_config.get("per_device_eval_batch_size", 1)))
    add_if_supported(kwargs, signature, "gradient_accumulation_steps", int(training_config.get("gradient_accumulation_steps", 8)))
    add_if_supported(kwargs, signature, "num_train_epochs", float(training_config.get("num_train_epochs", 1)))
    add_if_supported(kwargs, signature, "max_steps", int(training_config.get("max_steps", -1)))
    add_if_supported(kwargs, signature, "learning_rate", float(training_config.get("learning_rate", 2e-4)))
    add_if_supported(kwargs, signature, "weight_decay", float(training_config.get("weight_decay", 0.0)))
    add_if_supported(kwargs, signature, "warmup_ratio", float(training_config.get("warmup_ratio", 0.03)))
    add_if_supported(kwargs, signature, "lr_scheduler_type", str(training_config.get("lr_scheduler_type", "cosine")))
    add_if_supported(kwargs, signature, "logging_steps", int(training_config.get("logging_steps", 10)))
    add_if_supported(kwargs, signature, "save_steps", int(training_config.get("save_steps", 100)))
    add_if_supported(kwargs, signature, "eval_steps", int(training_config.get("eval_steps", 100)) if has_eval else None)
    add_if_supported(kwargs, signature, "save_total_limit", int(training_config.get("save_total_limit", 2)))
    add_if_supported(kwargs, signature, "remove_unused_columns", False)
    add_if_supported(kwargs, signature, "gradient_checkpointing", bool_config(training_config.get("gradient_checkpointing"), True))
    add_if_supported(kwargs, signature, "dataloader_num_workers", int(training_config.get("dataloader_num_workers", 0)))
    add_if_supported(kwargs, signature, "dataloader_pin_memory", bool_config(training_config.get("dataloader_pin_memory"), True))
    add_if_supported(kwargs, signature, "report_to", training_config.get("report_to") or "none")
    add_if_supported(kwargs, signature, "optim", str(training_config.get("optim", "adamw_torch")))
    add_if_supported(kwargs, signature, "seed", int(training_config.get("seed", 42)))
    add_if_supported(kwargs, signature, "logging_dir", str(output_dir / "logs"))
    add_if_supported(kwargs, signature, "save_strategy", str(training_config.get("save_strategy", "steps")))

    eval_strategy = str(training_config.get("eval_strategy", "steps" if has_eval else "no"))
    if not has_eval:
        eval_strategy = "no"
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = eval_strategy
    elif "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = eval_strategy

    use_cuda = bool(torch.cuda.is_available())
    dtype_name = dtype_name.lower()
    if dtype_name in {"bfloat16", "bf16"}:
        add_if_supported(kwargs, signature, "bf16", bool(use_cuda and torch.cuda.is_bf16_supported()))
    if dtype_name in {"float16", "fp16"}:
        add_if_supported(kwargs, signature, "fp16", use_cuda)

    return TrainingArguments(**kwargs)


def build_trainer(
    model: Any,
    training_args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: Any,
    processor: Any,
) -> Any:
    from transformers import Trainer  # type: ignore

    signature = inspect.signature(Trainer.__init__)
    kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }
    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = processor
    elif "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = processor
    return Trainer(**kwargs)


def save_final_artifacts(
    trainer: Any,
    processor: Any,
    output_dir: Path,
    training_config: Dict[str, Any],
    train_mode: str,
    logger: logging.Logger,
) -> None:
    if not bool_config(training_config.get("save_final_model"), True):
        logger.info("Skipping final model save because training.save_final_model=false.")
        return

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(final_dir))
    logger.info("Saved final model artifacts: %s", final_dir)

    if train_mode == "lora" and bool_config(training_config.get("merge_lora"), False):
        model = trainer.model
        if not hasattr(model, "merge_and_unload"):
            logger.warning("merge_lora requested, but model does not expose merge_and_unload().")
            return
        merged_dir = output_dir / "merged"
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(str(merged_dir))
        logger.info("Saved merged LoRA model: %s", merged_dir)


def run(args: argparse.Namespace) -> int:
    overall_start = time.perf_counter()
    config_path = resolve_config_path(args.config, Path.cwd())
    if config_path is None:
        raise ValueError("--config is required")
    config_path = config_path.resolve()
    config = load_config(config_path)
    training_config = dict(config.get("training") or {})
    if args.no_eval:
        training_config["do_eval"] = False

    train_mode = normalize_train_mode(args.train_mode, training_config)
    task = infer_task_name(args.task, config, config_path)
    train_split = str(args.split or training_config.get("train_split") or ("train" if task == "cap" else "all"))
    eval_split = args.eval_split if args.eval_split is not None else training_config.get("eval_split")
    if eval_split is not None:
        eval_split = str(eval_split)

    sample_value = args.sample if args.sample is not None else training_config.get("sample", training_config.get("max_samples", "full"))
    max_samples, sample_label = normalize_sample_spec(sample_value)
    eval_sample_value = (
        args.eval_sample
        if args.eval_sample is not None
        else training_config.get("eval_sample", training_config.get("eval_max_samples", 100))
    )
    eval_max_samples, eval_sample_label = normalize_sample_spec(eval_sample_value)

    dataset_path = resolve_config_path(config.get("dataset_path"), config_path.parent)
    image_root = resolve_config_path(config.get("image_root"), config_path.parent)
    model_path = resolve_config_path(config.get("model_path"), config_path.parent)
    if dataset_path is None:
        raise ValueError("dataset_path is required in config.")
    if image_root is None:
        image_root = dataset_path.parent
    if model_path is None:
        raise ValueError("model_path is required in config.")

    output_root = get_train_output_root(config, training_config)
    output_dir = build_train_output_dir(task, train_mode, output_root, train_split, sample_label, training_config, args.output_dir)
    logger = setup_logging(output_dir, bool_config((config.get("logging") or {}).get("verbose"), True))

    runtime_config = dict(config)
    runtime_config["training"] = training_config
    runtime_config.update(
        {
            "mode": train_mode,
            "task": task,
            "config_path": str(config_path),
            "dataset_path": str(dataset_path),
            "image_root": str(image_root),
            "model_path": str(model_path),
            "output_dir": str(output_dir),
            "train_split": train_split,
            "eval_split": eval_split,
            "sample": sample_label,
            "max_samples": max_samples,
            "eval_sample": eval_sample_label,
            "eval_max_samples": eval_max_samples,
            "command": command_string(sys.argv),
        }
    )
    save_yaml_or_json(output_dir / "train_config.yaml", runtime_config)

    logger.info("Task: %s", task)
    logger.info("Mode: %s", train_mode)
    logger.info("Command: %s", runtime_config["command"])
    logger.info("Config path: %s", config_path)
    logger.info("Dataset path: %s", dataset_path)
    logger.info("Image root: %s", image_root)
    logger.info("Model path: %s", model_path)
    logger.info("Train split: %s", train_split if task == "cap" else "N/A")
    logger.info("Sample count: %s", sample_label)
    logger.info("Output directory: %s", output_dir)
    logger.info("Package status: %s", json.dumps(package_status(), ensure_ascii=False))

    train_samples_raw, train_schema = load_samples_for_task(
        task, config, dataset_path, image_root, train_split, max_samples, logger
    )
    train_samples, skipped_train = filter_trainable_samples(train_samples_raw, task, training_config)
    if skipped_train:
        write_jsonl(output_dir / "skipped_train_samples.jsonl", skipped_train)

    do_eval = bool_config(training_config.get("do_eval"), task == "cap")
    eval_samples: List[EvalSample] = []
    eval_schema: Dict[str, Any] = {}
    skipped_eval: List[Dict[str, Any]] = []
    if do_eval and eval_split and task == "cap":
        eval_samples_raw, eval_schema = load_samples_for_task(
            task, config, dataset_path, image_root, eval_split, eval_max_samples, logger
        )
        eval_samples, skipped_eval = filter_trainable_samples(eval_samples_raw, task, training_config)
        if skipped_eval:
            write_jsonl(output_dir / "skipped_eval_samples.jsonl", skipped_eval)
    elif do_eval and task != "cap":
        logger.info("Eval during VQA training is disabled unless a separate eval loader is configured.")

    dataset_summary = {
        "train_schema": train_schema,
        "eval_schema": eval_schema,
        "num_train_requested": len(train_samples_raw),
        "num_train_samples": len(train_samples),
        "num_train_skipped": len(skipped_train),
        "num_eval_samples": len(eval_samples),
        "num_eval_skipped": len(skipped_eval),
    }
    write_json(output_dir / "train_dataset_summary.json", dataset_summary)
    logger.info("Train samples: %d kept, %d skipped", len(train_samples), len(skipped_train))
    logger.info("Eval samples: %d kept, %d skipped", len(eval_samples), len(skipped_eval))

    if args.dry_run:
        write_json(
            output_dir / "train_metrics.json",
            {
                "dry_run": True,
                "ready_for_training": bool(train_samples),
                "mode": train_mode,
                "dataset": dataset_summary,
                "runtime_sec": time.perf_counter() - overall_start,
            },
        )
        logger.info("Dry run finished without loading model.")
        return 0

    if not train_samples:
        raise ValueError("No valid training samples found after filtering.")

    seed = int(training_config.get("seed", 42))
    set_seed(seed)

    processor, tokenizer, model, torch = load_processor_and_model(config, training_config, train_mode, model_path, logger)
    train_dataset = MedGemmaTrainDataset(train_samples)
    eval_dataset = MedGemmaTrainDataset(eval_samples) if eval_samples else None
    data_collator = MedGemmaDataCollator(
        processor=processor,
        task=task,
        image_config=dict(config.get("image") or {}),
        training_config=training_config,
        tokenizer=tokenizer,
    )

    training_args = build_training_arguments(
        training_config=training_config,
        output_dir=output_dir,
        dtype_name=str(config.get("dtype", "auto")),
        torch=torch,
        has_eval=eval_dataset is not None,
    )
    trainer = build_trainer(model, training_args, train_dataset, eval_dataset, data_collator, processor)

    resume_checkpoint = args.resume_from_checkpoint or training_config.get("resume_from_checkpoint")
    logger.info("Starting training%s", f" from {resume_checkpoint}" if resume_checkpoint else "")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_metrics = dict(getattr(train_result, "metrics", {}) or {})
    train_metrics.update(
        {
            "mode": train_mode,
            "task": task,
            "num_train_samples": len(train_samples),
            "num_eval_samples": len(eval_samples),
            "runtime_sec": time.perf_counter() - overall_start,
        }
    )

    trainer.save_state()
    save_final_artifacts(trainer, processor, output_dir, training_config, train_mode, logger)
    write_json(output_dir / "train_metrics.json", train_metrics)
    logger.info("Saved train metrics: %s", output_dir / "train_metrics.json")
    logger.info("Total runtime: %.3f sec", train_metrics["runtime_sec"])
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or fine-tune MedGemma on M3D-Cap/M3D-VQA.")
    parser.add_argument("--config", required=True, help="Path to CAP_task.yaml or VQA_task.yaml.")
    parser.add_argument("--task", choices=["cap", "caption", "captioning", "vqa"], default=None)
    parser.add_argument("--sample", default=None, help="Integer sample count or 'full' for the training split.")
    parser.add_argument("--split", default=None, help="Training split for captioning, for example train.")
    parser.add_argument("--eval-split", default=None, help="Evaluation split for captioning, for example validation.")
    parser.add_argument("--eval-sample", default=None, help="Integer eval sample count or 'full'.")
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory.")
    parser.add_argument("--train-mode", default=None, help="Training mode: lora/finetune or full/full_pipeline.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Checkpoint directory to resume from.")
    parser.add_argument("--no-eval", action="store_true", help="Disable eval during training.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/dataset and write skeleton outputs.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
