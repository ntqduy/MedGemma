# MedGemma Evaluation and Fine-tuning

This project evaluates and fine-tunes MedGemma on local medical image datasets:

- Captioning: `M3D-Cap/M3D_Cap/M3D_Cap.json`
- VQA: `M3D-VQA/M3D-VQA/M3D_VQA_test5k.csv`

The default configs use the instruction-tuned model at:

```text
weight/medgemma-1.5-4b-it
```

## Setup

Create an environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

For CUDA machines, install the PyTorch build that matches your CUDA version if
the default `pip` package is not suitable.

## Model Weights

The configs set `local_files_only: true`, so the full Hugging Face model must
exist locally before running evaluation or training.

Expected layout:

```text
MedGemma/
  weight/
    medgemma-1.5-4b-it/
      config.json
      model.safetensors.index.json
      model-00001-of-00002.safetensors
      model-00002-of-00002.safetensors
      tokenizer files...
```

If you use Hugging Face CLI and have access to the model:

```bash
huggingface-cli login
huggingface-cli download google/medgemma-1.5-4b-it --local-dir weight/medgemma-1.5-4b-it
```

If your model is stored somewhere else, update `model_path` in:

```text
config/CAP_task.yaml
config/VQA_task.yaml
```

## Dataset Paths

The default config paths are relative to this project directory:

```text
../M3D-Cap/M3D_Cap/M3D_Cap.json
../M3D-VQA/M3D-VQA/M3D_VQA_test5k.csv
../M3D-Cap
```

Edit `dataset_path` and `image_root` in the config files if your data is in a
different location.

## Evaluation

Run from this directory.

Captioning:

```bash
bash scripts/eval_CAP.sh 100 test1k
bash scripts/eval_CAP.sh full test1k
```

VQA:

```bash
bash scripts/eval_VQA.sh 100
bash scripts/eval_VQA.sh full
```

Direct CLI:

```bash
python evaluate.py --config config/CAP_task.yaml --task cap --sample 100 --split test1k
python evaluate.py --config config/VQA_task.yaml --task vqa --sample 100
```

Rule-based multi-slice baseline:

```bash
python evaluate.py --config config/CAP_task.yaml --task cap --sample 100 --split test1k \
  --num_slices 9 --slice_strategy uniform --view axial --inference_mode montage

python evaluate.py --config config/VQA_task.yaml --task vqa --sample 100 \
  --num_slices 5 --slice_strategy uniform --view coronal --inference_mode independent
```

You can also use the shared entrypoint:

```bash
python main.py eval --config config/CAP_task.yaml --task cap --sample 100 --split test1k
python main.py eval --config config/VQA_task.yaml --task vqa --sample 100
```

Each run writes `predictions.jsonl`, `metrics.json`, `benchmark.json`,
`run_config.yaml`, `log.txt`, `errors.jsonl`, and preview files under
`results/EVAL_*`.

### Slice Inference Baseline

The evaluator uses a fixed rule-based slice baseline for 3D volumes. The model
does not choose slices.

Defaults:

```text
--num_slices 1
--slice_strategy middle
--view axial
--inference_mode montage
```

Rules:

- `num_slices=1` selects the middle slice for the chosen view.
- `num_slices>1` selects uniformly spaced slices for the chosen view.
- Uniform selection skips the first and last 10% of the selected axis.
- No entropy, mask, model-based, or adaptive selection is used.

Views:

```text
axial    -> axis 0
coronal  -> axis 1
sagittal -> axis 2
```

`montage` mode builds a near-square grid, resizes it for the processor, saves
debug images under `results/EVAL_*/montages/`, and runs MedGemma once.

`independent` mode runs MedGemma once per selected slice. VQA uses majority vote
over per-slice answers. Captioning concatenates slice captions in slice order.

Each row in `predictions.jsonl` includes `view`, `num_slices`,
`selected_slice_indices`, `inference_mode`, `prompt`, `prediction`,
`ground_truth`, plus `montage_path` or `per_slice_predictions` depending on the
mode.

## Training / Fine-tuning

Both task configs include `training` and `fine_tuning` sections. Use
`training.mode: lora` for adapter fine-tuning or `training.mode: full` to train
all unfrozen parameters. The same value can be overridden from the CLI.

PowerShell:

```powershell
.\scripts\train_CAP.ps1 100 lora train
.\scripts\train_CAP.ps1 full full train
.\scripts\train_VQA.ps1 100 lora
```

Bash:

```bash
bash scripts/train_CAP.sh 100 lora train
bash scripts/train_CAP.sh full full train
bash scripts/train_VQA.sh 100 lora
```

Direct CLI:

```bash
python train.py --config config/CAP_task.yaml --task cap --sample 100 --train-mode lora --split train
python train.py --config config/CAP_task.yaml --task cap --sample full --train-mode full --split train
python train.py --config config/VQA_task.yaml --task vqa --sample 100 --train-mode lora
```

Training writes `train_config.yaml`, `train_dataset_summary.json`,
`train_metrics.json`, checkpoints, and final model artifacts under
`results/TRAIN_*`.

## Useful Checks

Validate config and dataset paths without loading the model:

```bash
python evaluate.py --config config/CAP_task.yaml --task cap --sample 5 --split test1k --dry-run
python train.py --config config/CAP_task.yaml --task cap --sample 5 --train-mode lora --split train --dry-run
```

Check Python syntax:

```bash
python -m py_compile main.py evaluate.py train.py
```

## Troubleshooting

If you see a missing shard error such as:

```text
FileNotFoundError: model-00001-of-00002.safetensors
```

the model directory is incomplete or `model_path` points to the wrong folder.
Download/copy all shard files listed in `model.safetensors.index.json`, or
update `model_path`.

If you see:

```text
HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'
```

Transformers did not recognize `model_path` as an existing local directory and
tried to parse it as a Hugging Face repo id. Check that the resolved path exists
on the machine where you run the command:

```bash
ls -ld weight/medgemma-1.5-4b-it
ls -lh weight/medgemma-1.5-4b-it
```

If you see:

```text
`torch_dtype` is deprecated! Use `dtype` instead!
```

update to the latest project code. The loader now passes `dtype` to
Transformers.

`bitsandbytes` is skipped on Windows by `requirements.txt`. Quantized 4-bit or
8-bit loading is recommended on Linux/CUDA environments.

`BERTScore` is disabled by default because it loads a separate text model such
as `roberta-large` from Hugging Face. Enable `metrics.bertscore: true` only when
that metric model is already cached locally or the machine has internet access.

`processor.use_fast` is set to `false` in the task configs to keep the saved
slow image processor behavior explicit and avoid Transformers version-change
warnings. Set it to `true` only if you intentionally want the fast processor.
