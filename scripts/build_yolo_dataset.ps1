param(
  [Parameter(Mandatory = $true)]
  [string]$DatasetRoot,

  [Parameter(Mandatory = $true)]
  [string]$ProjectRef,

  [string]$OutputDir = ".\\dist-yolo",
  [string]$BucketName = "ml-datasets",
  [string]$DatasetName = "whitelinez",
  [string[]]$Classes = @("car")
)

$ErrorActionPreference = "Stop"

function Assert-Path([string]$p) {
  if (-not (Test-Path $p)) {
    throw "Missing required path: $p"
  }
}

$datasetRootAbs = (Resolve-Path $DatasetRoot).Path
$outputAbs = (Resolve-Path .).Path
$outputAbs = Join-Path $outputAbs $OutputDir

New-Item -ItemType Directory -Path $outputAbs -Force | Out-Null

$required = @(
  "images/train",
  "images/val",
  "labels/train",
  "labels/val"
)

foreach ($rel in $required) {
  Assert-Path (Join-Path $datasetRootAbs $rel)
}

$stageDir = Join-Path $outputAbs $DatasetName
if (Test-Path $stageDir) {
  Remove-Item -Recurse -Force $stageDir
}
New-Item -ItemType Directory -Path $stageDir | Out-Null

Copy-Item -Recurse -Force (Join-Path $datasetRootAbs "images") (Join-Path $stageDir "images")
Copy-Item -Recurse -Force (Join-Path $datasetRootAbs "labels") (Join-Path $stageDir "labels")

$zipPath = Join-Path $outputAbs "$DatasetName-yolo.zip"
if (Test-Path $zipPath) {
  Remove-Item -Force $zipPath
}

Compress-Archive -Path (Join-Path $stageDir "images"), (Join-Path $stageDir "labels") -DestinationPath $zipPath -CompressionLevel Optimal

$yamlPath = Join-Path $outputAbs "data.yaml"
$namesLines = @()
for ($i = 0; $i -lt $Classes.Count; $i++) {
  $namesLines += "  $i`: $($Classes[$i])"
}

$downloadUrl = "https://$ProjectRef.supabase.co/storage/v1/object/public/$BucketName/datasets/$DatasetName/$DatasetName-yolo.zip"

$yaml = @(
  "path: ./datasets/$DatasetName",
  "train: images/train",
  "val: images/val",
  "names:",
  $namesLines,
  "download: $downloadUrl"
) -join "`n"

Set-Content -Path $yamlPath -Value $yaml -Encoding UTF8

Write-Host "Created:"
Write-Host "  Zip:  $zipPath"
Write-Host "  YAML: $yamlPath"
Write-Host ""
Write-Host "Upload both to:"
Write-Host "  $BucketName/datasets/$DatasetName/"
Write-Host ""
Write-Host "Set Railway env:"
Write-Host "  TRAINER_DATASET_YAML_URL=https://$ProjectRef.supabase.co/storage/v1/object/public/$BucketName/datasets/$DatasetName/data.yaml"

