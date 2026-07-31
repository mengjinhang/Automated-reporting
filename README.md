# 流感预警综合分析报告自动生成

根据流感监测数据（ILI + 搜索指数）和 CNE/TRCE/Fusion 三模型预警结果，自动生成
与样例一致的《传染病智能预警综合分析报告》（Markdown + 三张监测图表）。

**设计原则**：所有数值、表格、提前天数由代码精确计算，本地 LLM（Ollama）仅生成
叙述性段落（风险判断 / 图表解读 / 可信度分析 / 防控建议），避免数值幻觉。
四个叙述段落按同一条 Ollama chat 会话串行生成，后一段可参考前文，但每轮都会追加
本段专属任务和事实边界约束。

## 目录结构

```
Flu/
├── data/                    # 原始数据
│   ├── 测试数据全国_填充.xlsx
│   ├── cne_alerts.csv / trce_alerts.csv / critical_alerts.csv
├── config.yaml              # 配置（Ollama 地址、疾病、区域、峰值参数）
├── service.py               # 服务封装入口（供其他程序调用）
├── core/
│   ├── generate_report.py   # 报告生成核心逻辑
│   └── md_to_pdf.py         # Markdown 转 PDF
└── output/
    ├── report_YYYYMMDD_HHMM.md / .pdf
    └── charts/{cne,trce,fusion}_chart.png
```

## 使用

```bash
cd /Users/mac/Desktop/Flu

# 1. 先用规则底稿验证数据分析与图表（不调用 LLM）
python3 service.py --dry-run

# 2. 在 config.yaml 中填入远程 Ollama 地址与模型名后，运行完整流程
python3 service.py
```

### 作为服务调用

```python
from service import generate_report

result = generate_report(
    dry_run=False,
    enable_md2pdf=True,
    input_dir="data",
    output_dir="output",
    markdown_path="report.md",
    pdf_path="report.pdf",
    verbose=True,
)
print(result["markdown_path"])
print(result["pdf_path"])
```

也可以用命令行获取 JSON 结果：

```bash
python3 service.py --dry-run --json --no-md2pdf
python3 service.py --json --input-dir data --output-dir output --md2pdf
python3 service.py --json --output-dir output --markdown-path report.md --pdf-path report.pdf
```

### 配置远程 Ollama

编辑 `config.yaml`：

```yaml
ollama:
  base_url: "http://<远程IP>:11434"   # 远程服务器地址
  model: "qwen3.5:9b"     
```

> 远程 Ollama 需以 `OLLAMA_HOST=0.0.0.0 ollama serve` 启动才能被外部访问，
> 并确保 11434 端口对本机开放。

## 依赖

```
pandas openpyxl matplotlib numpy pyyaml requests
```

（峰值检测用内置纯 numpy 实现，无需 scipy。）

## 说明

- **历史事件表**由 ILI 峰值检测 + 预警点匹配自动生成，提前天数 = 峰值时间 − 首次预警时间。
  峰值灵敏度由 `config.yaml` 的 `peak_detection.distance/prominence` 控制。
- **CNE 网络熵 H、TRCE tRCE 指标**为可视化代理指标（由搜索指数/ILI 计算），
  图中预警点均来自真实 CSV。若后续有模型输出的真实指标序列，可替换
  `compute_network_entropy` / `compute_trce_index` 两个函数。
- 输出为 Markdown；如需转 docx，可用 `pandoc report.md -o report.docx`。
