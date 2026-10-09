# Validate the generated docx by opening it in Word and exporting a PDF.
# This is the strongest check available: if Word opens it without a repair
# prompt and exports successfully, the OOXML is valid.
#
# NOTE: under the DSH sandbox, Word COM automation is REFUSED:
#   New-Object : CLSID {000209FF-...} 80080005 Server execution failed
# In that case this script reports COM_AUTOMATION_BLOCKED and tells you to
# open the file manually instead of pretending it passed.
#
# Keep this file ASCII-only: Windows PowerShell mis-parses UTF-8 Chinese
# text in .ps1 files that have no BOM.

$ErrorActionPreference = 'Continue'
$docx = Join-Path $PSScriptRoot '第一次作业_报告.docx'
$pdf  = Join-Path $PSScriptRoot '过程记录\报告预览.pdf'

if (-not (Test-Path -LiteralPath $docx)) {
    Write-Output "MISSING: $docx"
    exit 1
}

$word = $null
try {
    $word = New-Object -ComObject Word.Application
} catch {
    Write-Output 'COM_AUTOMATION_BLOCKED'
    Write-Output 'Word COM is not available in this sandbox; cannot export PDF automatically.'
    Write-Output ('Please open it manually: ' + $docx)
    exit 2
}

try {
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($docx, $false, $true)   # ReadOnly
    Write-Output ('PAGES=' + $doc.ComputeStatistics(2))       # wdStatisticPages
    Write-Output ('WORDS=' + $doc.ComputeStatistics(0))       # wdStatisticWords
    Write-Output ('CHARS=' + $doc.ComputeStatistics(3))       # wdStatisticCharacters
    Write-Output ('PARAGRAPHS=' + $doc.Paragraphs.Count)
    Write-Output ('INLINE_SHAPES=' + $doc.InlineShapes.Count)
    Write-Output ('TABLES=' + $doc.Tables.Count)
    $doc.ExportAsFixedFormat($pdf, 17)                        # wdExportFormatPDF
    $doc.Close($false)
    Write-Output ('PDF_OK=' + $pdf)
} finally {
    if ($word -ne $null) {
        $word.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
}
