param(
    [string]$Sample = "100",
    [string]$Mode = "lora",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

python (Join-Path $ProjectRoot "train.py") `
    --config (Join-Path $ProjectRoot "config\VQA_task.yaml") `
    --task vqa `
    --sample $Sample `
    --train-mode $Mode `
    @ExtraArgs
