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

### Main Evaluation Path

Use the shell scripts for the normal workflow. They call `evaluate.py` with the
task YAML configs, so prompt, slice selection, generation, metrics, and logging
are controlled from `config/CAP_task.yaml` and `config/VQA_task.yaml`.

Slice strategies:

```text
middle          -> middle slice only, use with --num_slices 1
uniform         -> N slices uniformly across the volume axis
center_uniform  -> N slices centered around the middle slice
```

Captioning:

```bash
bash scripts/eval_CAP.sh 100 test1k
bash scripts/eval_CAP.sh full test1k
bash scripts/eval_CAP.sh 100 test1k 9 axial montage center_uniform
```

VQA:

```bash
bash scripts/eval_VQA.sh 100
bash scripts/eval_VQA.sh full
bash scripts/eval_VQA.sh 100 9 axial montage center_uniform
```

Multi-GPU:

```bash
# By default, edit cuda_visible_devices in config/CAP_task.yaml or config/VQA_task.yaml.
bash scripts/eval_CAP.sh 100 test1k
bash scripts/eval_VQA.sh 100

# Temporary env override for one run.
CUDA_DEVICE_IDS=0,1 bash scripts/eval_CAP.sh 100 test1k
CUDA_DEVICE_IDS=0,1 bash scripts/eval_VQA.sh 100

# If you run CAP and VQA at the same time, pin them to different GPUs.
CUDA_DEVICE_IDS=1 bash scripts/eval_CAP.sh 100 test1k
CUDA_DEVICE_IDS=0 bash scripts/eval_VQA.sh 100
```

The task configs use:

```yaml
cuda_visible_devices: "0,1,2,3,4,5,6,7"
parallel_eval: true
device: auto
device_map: auto
```

With `parallel_eval: true`, evaluation is split by sample across the GPUs in
`cuda_visible_devices`: one worker process per GPU, then `predictions.jsonl`,
`errors.jsonl`, and `metrics.json` are merged back into the main output folder.
If `CUDA_VISIBLE_DEVICES` or `CUDA_DEVICE_IDS` is set in the shell, that
environment value overrides the config for that run.

Set `parallel_eval: false` if you only want a single process with
Transformers/Accelerate `device_map: auto` model sharding.

Direct CLI:

```bash
python evaluate.py --config config/CAP_task.yaml --task cap --sample 100 --split test1k
python evaluate.py --config config/VQA_task.yaml --task vqa --sample 100
```

Rule-based multi-slice baseline:

```bash
python evaluate.py --config config/CAP_task.yaml --task cap --sample 100 --split test1k \
  --num_slices 9 --slice_strategy center_uniform --view axial --inference_mode montage

python evaluate.py --config config/VQA_task.yaml --task vqa --sample 100 \
  --num_slices 9 --slice_strategy center_uniform --view axial --inference_mode montage
```

The shell scripts support a short positional slice form and also forward normal
CLI flags:

```bash
bash scripts/eval_CAP.sh 100 test1k 9 axial montage center_uniform
bash scripts/eval_CAP.sh 100 test1k --num_slices 9 --slice_strategy center_uniform --view axial --inference_mode montage

bash scripts/eval_VQA.sh 100 9 axial montage center_uniform
bash scripts/eval_VQA.sh 100 --num_slices 9 --slice_strategy center_uniform --view axial --inference_mode montage
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

The current task configs use:

```text
num_slices: 9
slice_strategy: center_uniform
view: axial
inference_mode: montage
```

The current default uses 9 slices centered around the middle slice. Avoid
`num_slices: auto` for montage runs because deep volumes can produce very large
montages, for example 206 selected slices.

The evaluator fallback defaults, used only when neither CLI nor config provides
values, are:

```text
--num_slices 1
--slice_strategy middle
--view axial
--inference_mode montage
```

`--num_slice` and `--num-slice` are accepted as aliases for `--num_slices`.

Rules:

- `num_slices=1` selects the middle slice for the chosen view.
- `num_slices>1` selects uniformly spaced slices for the chosen view.
- `slice_strategy=center_uniform` selects N slices around the middle slice.
- `num_slices=auto` selects all valid slices after the 10% edge skip and should
  not be used for the central 9-slice montage baseline.
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

`BERTScore` loads a separate text model, not the MedGemma weights. The configs
point `metrics.bertscore_model_type` to `weight/roberta-large`; copy/download a
complete Hugging Face `roberta-large` folder there, or change that config value
to the absolute local path of your cached snapshot. When using local
`roberta-large`, keep `metrics.bertscore_num_layers: 17` so `bert-score` does
not try to look up the local path as a built-in model name.

`processor.use_fast` is set to `false` in the task configs to keep the saved
slow image processor behavior explicit and avoid Transformers version-change
warnings. Set it to `true` only if you intentionally want the fast processor.
