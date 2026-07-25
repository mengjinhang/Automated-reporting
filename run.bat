@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================
echo   流感预警报告生成
echo ============================================
echo.
echo [检查依赖]
python -m pip install -q pandas openpyxl matplotlib numpy pyyaml requests
echo.
echo [1] 先用占位文本验证数据与图表 (不调用大模型)
python generate_report.py --dry-run
echo.
echo 如上一步正常, 请确认 config.yaml 中的 model 名与 ollama list 一致,
echo 然后按任意键运行完整流程 (调用本地 Ollama)...
pause >nul
python generate_report.py
echo.
echo 完成. 报告在 output 目录下.
pause
