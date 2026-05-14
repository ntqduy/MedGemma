param(
    [string]$Sample = "100",
    [string]$Mode = "lora",
    [string]$Split = "train",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

python (Join-Path $ProjectRoot "train.py") `
    --config (Join-Path $ProjectRoot "config\CAP_task.yaml") `
    --task cap `
    --sample $Sample `
    --train-mode $Mode `
    --split $Split `
    @ExtraArgs
