# PowerShell script to build thesis/main.pdf
$ErrorActionPreference = "Stop"
$ThesisDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ThesisDir "..\\thesis")

Write-Host "Building thesis PDF in $(Get-Location)"

pdflatex -interaction=nonstopmode main.tex | Out-Null
if ($LASTEXITCODE -ne 0) {
    pdflatex -interaction=nonstopmode main.tex
    throw "pdflatex failed on feature pass"
}

bibtex main | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null

if (Test-Path main.pdf) {
    Write-Host "Success: thesis/main.pdf"
} else {
    throw "main.pdf was not created"
}
