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
echo [生成报告] 调用 config.yaml 中配置的 Ollama；失败会自动使用规则底稿
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
