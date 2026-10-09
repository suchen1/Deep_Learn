# 把最新版报告换回正式文件名。
# 旧文件被 Word 占着时（WinError 32）无法直接覆盖，所以生成脚本会写成 _v2/_v3/_v4。
# 关掉 Word 后运行本脚本即可完成替换。
# 注意：本文件必须带 UTF-8 BOM，否则 Windows PowerShell 5.1 会按 GBK 解码中文路径。

$ErrorActionPreference = 'Continue'
$here = $PSScriptRoot
$main = Join-Path $here '第一次作业_报告.docx'

# 找出所有 _vN 版本，取最新那个
$vers = Get-ChildItem -LiteralPath $here -Filter '第一次作业_报告_v*.docx' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime
if (-not $vers) { Write-Output '没有找到 _vN 版本，无需替换。'; exit 0 }
$new = $vers[-1]
Write-Output ('最新版本: ' + $new.Name)

if (Test-Path -LiteralPath $main) {
    try {
        Remove-Item -LiteralPath $main -Force -ErrorAction Stop
    } catch {
        Write-Output '旧文件仍被占用 —— 请先关闭 Word（以及资源管理器里的预览窗格），再运行本脚本。'
        exit 2
    }
}

Move-Item -LiteralPath $new.FullName -Destination $main -Force
Write-Output ('已替换: ' + $main)

# 清掉其余旧版本
foreach ($v in $vers) {
    if ($v.FullName -ne $new.FullName -and (Test-Path -LiteralPath $v.FullName)) {
        try {
            Remove-Item -LiteralPath $v.FullName -Force -ErrorAction Stop
            Write-Output ('已删除多余版本: ' + $v.Name)
        } catch {
            Write-Output ('无法删除（可能也开着）: ' + $v.Name)
        }
    }
}
Get-ChildItem -LiteralPath $here -Filter '第一次作业_报告*.docx' | Select-Object Name, Length | Format-Table -AutoSize
