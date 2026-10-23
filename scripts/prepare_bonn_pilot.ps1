$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$dataRoot = Join-Path $projectRoot "data\bonn"
$sourceRoot = Join-Path $dataRoot "source"
$commit = "2ad85e77b6c1fea00124e055cedbc45181e9f6ba"
$baseUrl = "https://raw.githubusercontent.com/cesium-ml/cesium-data/$commit/andrzejak"
$sets = @(
    @{ Name = "F"; Size = 582967 },
    @{ Name = "S"; Size = 764359 }
)

New-Item -ItemType Directory -Force -Path $sourceRoot | Out-Null
foreach ($set in $sets) {
    $archive = Join-Path $sourceRoot "$($set.Name).zip"
    curl.exe --fail --location --retry 2 "$baseUrl/$($set.Name).zip" --output $archive
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to download the Bonn set $($set.Name) archive."
    }
    if ((Get-Item $archive).Length -ne $set.Size) {
        throw "Unexpected archive size for Bonn set $($set.Name)."
    }
    Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $dataRoot $set.Name) -Force
}

$readme = Join-Path $sourceRoot "README.txt"
curl.exe --fail --location --retry 2 "$baseUrl/README.txt" --output $readme
if ($LASTEXITCODE -ne 0) {
    throw "Failed to download the Bonn dataset description."
}

$manifestRows = foreach ($set in $sets) {
    $directory = Join-Path $dataRoot $set.Name
    $segments = @(Get-ChildItem -LiteralPath $directory -Filter "$($set.Name)*.txt" -File)
    if ($segments.Count -ne 100) {
        throw "Expected 100 Bonn set $($set.Name) segments; found $($segments.Count)."
    }
    foreach ($segment in $segments) {
        [PSCustomObject]@{
            path = Join-Path $set.Name $segment.Name
            label = $set.Name
            group_id = "$($set.Name)-$($segment.BaseName)"
        }
    }
}
$manifestRows | Export-Csv -NoTypeInformation -Encoding utf8 `
    -Path (Join-Path $dataRoot "manifest.csv")
Write-Output "Prepared 200 real Bonn EEG segments and data\bonn\manifest.csv."
