@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================
echo   流感预警报告生成
echo ============================================
echo.
echo [检查依赖]
python -m pip install -q pandas openpyxl matplotlib numpy pyyaml requests reportlab
if errorlevel 1 goto error
echo.
echo [1] 先用规则底稿验证数据与图表 (不调用大模型)
python service.py --dry-run
if errorlevel 1 goto error
echo.
echo [2] 运行完整流程 (调用 config.yaml 中配置的 Ollama；失败会自动使用规则底稿)
python service.py
if errorlevel 1 goto error
echo.
echo 完成. 报告在 output 目录下.
exit /b 0

:error
echo.
echo 运行失败，请查看上方错误信息。
pause
exit /b 1
