# MedGemma Evaluation

This project includes a self-contained evaluator for:

- Image captioning on `M3D-Cap/M3D-Cap/M3D_Cap.json`
- VQA on `M3D-VQA/M3D-VQA/M3D_VQA_test5k.csv`

Example commands from the repository root:

```bash
bash MedGemma/scripts/eval_CAP.sh 100 test1k
bash MedGemma/scripts/eval_CAP.sh full test1k

bash MedGemma/scripts/eval_VQA.sh 100
bash MedGemma/scripts/eval_VQA.sh full
```

Direct CLI:

```bash
python MedGemma/evaluate.py --config MedGemma/config/CAP_task.yaml --task cap --sample 100 --split test1k
python MedGemma/evaluate.py --config MedGemma/config/VQA_task.yaml --task vqa --sample 100
```

Each run writes `predictions.jsonl`, `metrics.json`, `benchmark.json`,
`run_config.yaml` or `run_config.json`, `log.txt`, `errors.jsonl`, and preview
files under `MedGemma/results/`.

## Training / fine-tuning

Both task configs include `training` and `fine_tuning` sections. Use
`training.mode: lora` for adapter fine-tuning or `training.mode: full` to train
all unfrozen parameters. The same can be overridden from the CLI.

PowerShell:

```powershell
.\MedGemma\scripts\train_CAP.ps1 100 lora train
.\MedGemma\scripts\train_CAP.ps1 full full train
.\MedGemma\scripts\train_VQA.ps1 100 lora
```

Bash:

```bash
bash MedGemma/scripts/train_CAP.sh 100 lora train
bash MedGemma/scripts/train_CAP.sh full full train
bash MedGemma/scripts/train_VQA.sh 100 lora
```

Direct CLI:

```bash
python MedGemma/train.py --config MedGemma/config/CAP_task.yaml --task cap --sample 100 --train-mode lora --split train
python MedGemma/train.py --config MedGemma/config/CAP_task.yaml --task cap --sample full --train-mode full --split train
python MedGemma/train.py --config MedGemma/config/VQA_task.yaml --task vqa --sample 100 --train-mode lora
```

Training writes `train_config.yaml`, `train_dataset_summary.json`,
`train_metrics.json`, checkpoints, and final model artifacts under
`MedGemma/results/TRAIN_*`.
