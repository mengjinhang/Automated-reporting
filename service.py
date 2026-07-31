#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Service entrypoint for report generation."""

import argparse
import json

from core.generate_report import DEFAULT_CONFIG
from core.generate_report import generate_report as core_generate_report


# Defaults exposed for local/remote deployment adaptation.
CONFIG_PATH = DEFAULT_CONFIG  # 配置文件路径；通常保持默认 config.yaml。
INPUT_DIR = None  # 输入数据目录；例如 "E:/.../Automated-reporting/data"。
OUTPUT_DIR = None  # 报告输出目录；例如 "E:/.../Automated-reporting/output"。
MAIN_DATA = None  # 主数据 Excel 文件；None 表示使用 config.yaml 中的 main_data。
CNE_ALERTS = None  # CNE 预警 CSV；None 表示使用 config.yaml 中的 cne_alerts。
TRCE_ALERTS = None  # TRCE 预警 CSV；None 表示使用 config.yaml 中的 trce_alerts。
FUSION_ALERTS = None  # Fusion 预警 CSV；None 表示使用 config.yaml 中的 fusion_alerts。
MARKDOWN_PATH = None  # Markdown 输出路径；None 表示自动生成 report_时间.md。
PDF_PATH = None  # PDF 输出路径；None 表示与 Markdown 同名输出。
ENABLE_MD2PDF = True  # 是否启用 Markdown 转 PDF。
DRY_RUN = False  # 是否只生成规则底稿，不调用 Ollama。
VERBOSE = True  # 是否打印执行过程日志。


def generate_report(
    config_path=CONFIG_PATH,
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    enable_md2pdf=ENABLE_MD2PDF,
    main_data=MAIN_DATA,
    cne_alerts=CNE_ALERTS,
    trce_alerts=TRCE_ALERTS,
    fusion_alerts=FUSION_ALERTS,
    markdown_path=MARKDOWN_PATH,
    pdf_path=PDF_PATH,
    dry_run=DRY_RUN,
    verbose=VERBOSE,
    now=None,
):
    """Call core report generation with deployment-friendly explicit params."""
    return core_generate_report(
        config_path=config_path,
        dry_run=dry_run,
        md2pdf=enable_md2pdf,
        now=now,
        verbose=verbose,
        data_dir=input_dir,
        main_data=main_data,
        cne_alerts=cne_alerts,
        trce_alerts=trce_alerts,
        fusion_alerts=fusion_alerts,
        output_dir=output_dir,
        markdown_path=markdown_path,
        pdf_path=pdf_path,
    )


def main():
    parser = argparse.ArgumentParser(description="流感预警报告生成服务入口")
    parser.add_argument("--config", default=CONFIG_PATH, help="配置文件路径")
    parser.add_argument("--input-dir", "--data-dir", dest="input_dir", default=INPUT_DIR, help="输入数据目录")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--main-data", default=MAIN_DATA, help="主数据 Excel 文件名或路径")
    parser.add_argument("--cne-alerts", default=CNE_ALERTS, help="CNE 预警 CSV 文件名或路径")
    parser.add_argument("--trce-alerts", default=TRCE_ALERTS, help="TRCE 预警 CSV 文件名或路径")
    parser.add_argument("--fusion-alerts", default=FUSION_ALERTS, help="Fusion 预警 CSV 文件名或路径")
    parser.add_argument("--markdown-path", default=MARKDOWN_PATH, help="Markdown 输出路径")
    parser.add_argument("--pdf-path", default=PDF_PATH, help="PDF 输出路径")
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN, help="不调用 Ollama，使用规则底稿")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结构化结果")
    parser.add_argument("--quiet", action="store_true", help="不打印过程日志")
    pdf_group = parser.add_mutually_exclusive_group()
    pdf_group.add_argument("--md2pdf", dest="enable_md2pdf", action="store_true", default=ENABLE_MD2PDF, help="生成 PDF")
    pdf_group.add_argument("--no-md2pdf", "--no-pdf", dest="enable_md2pdf", action="store_false", help="只生成 Markdown")
    args = parser.parse_args()

    result = generate_report(
        config_path=args.config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        enable_md2pdf=args.enable_md2pdf,
        main_data=args.main_data,
        cne_alerts=args.cne_alerts,
        trce_alerts=args.trce_alerts,
        fusion_alerts=args.fusion_alerts,
        markdown_path=args.markdown_path,
        pdf_path=args.pdf_path,
        dry_run=args.dry_run,
        verbose=not args.quiet and not args.json,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"Markdown → {result['markdown_path']}")
    if result["pdf_path"]:
        print(f"PDF → {result['pdf_path']}")
    elif result["pdf_error"]:
        print(f"PDF 生成失败：{result['pdf_error']}")


if __name__ == "__main__":
    main()
