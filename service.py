#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Service wrapper for generating flu surveillance reports.

This module keeps orchestration code in one reusable entry point so other
programs can call the project without shelling out to core/generate_report.py.
"""
import argparse
import contextlib
import io
import json
import os
from datetime import datetime

from core import generate_report as report_core


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "config.yaml")


def _now_value(now=None):
    if now is None:
        return datetime.now()
    if isinstance(now, datetime):
        return now
    if isinstance(now, str):
        return datetime.fromisoformat(now)
    raise TypeError("now must be None, datetime, or ISO datetime string")


def _latest_summary(analysis):
    latest = analysis["latest"]
    return {
        "date": latest["date"],
        "t": latest["t"],
        "triggered": latest["triggered"],
        "untriggered": latest["untriggered"],
        "risk_color": latest["risk_color"],
        "risk_status": latest["risk_status"],
        "peak_desc": latest["peak_desc"],
        "near_peak_t": latest["near_peak_t"],
        "near_peak_date": latest["near_peak_date"],
    }


def _call_narratives(analysis, cfg, dry_run, verbose):
    if verbose:
        return report_core.gen_narratives(analysis, cfg, dry_run=dry_run)
    with contextlib.redirect_stdout(io.StringIO()):
        return report_core.gen_narratives(analysis, cfg, dry_run=dry_run)


def generate_report(config_path=None, dry_run=False, make_pdf=True, now=None, verbose=False):
    """Generate a report and return structured output paths and metadata.

    Args:
        config_path: Path to config YAML. Defaults to ./config.yaml.
        dry_run: Use deterministic rule-based narratives without calling Ollama.
        make_pdf: Convert the generated Markdown report to PDF.
        now: Optional datetime or ISO datetime string used in report timestamp
            and output file name.
        verbose: Print progress and LLM diagnostics while running.

    Returns:
        dict with report paths, chart paths, latest warning summary, and status.
    """
    config_path = config_path or DEFAULT_CONFIG
    report_core.setup_font()
    cfg = report_core.load_config(config_path)
    run_time = _now_value(now)
    cfg["_now"] = run_time.strftime("%Y-%m-%d %H:%M:%S")

    if verbose:
        print("[1/6] 加载数据 ...")
    data = report_core.load_data(cfg)

    if verbose:
        print("[2/6] 分析疫情事件 ...")
    analysis = report_core.analyze_flu_events(data, cfg)

    out_dir = cfg["output"]["dir"]
    chart_dir = os.path.join(out_dir, "charts")
    if verbose:
        print("[3/6] 生成图表 ...")
    chart_paths = report_core.generate_charts(data, chart_dir)

    if verbose:
        mode = "dry-run 规则底稿" if dry_run else "调用 Ollama"
        print(f"[4/6] 生成叙述文本（{mode}） ...")
    narratives = _call_narratives(analysis, cfg, dry_run=dry_run, verbose=verbose)

    if verbose:
        print("[5/6] 组装 Markdown ...")
    markdown = report_core.build_report(analysis, narratives, chart_paths, cfg, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, f"report_{run_time.strftime('%Y%m%d_%H%M')}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    pdf_path = None
    pdf_error = None
    if make_pdf:
        if verbose:
            print("[6/6] 生成 PDF ...")
        try:
            from core.md_to_pdf import export_md_pdf

            if verbose:
                pdf_path = export_md_pdf(md_path)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    pdf_path = export_md_pdf(md_path)
        except Exception as exc:
            pdf_error = str(exc)
            if verbose:
                print(f"  PDF 生成失败：{pdf_error}")

    return {
        "ok": True,
        "dry_run": dry_run,
        "generated_at": cfg["_now"],
        "config_path": os.path.abspath(config_path),
        "markdown_path": md_path,
        "pdf_path": pdf_path,
        "pdf_error": pdf_error,
        "chart_paths": chart_paths,
        "data_period": analysis["data_period"],
        "event_count": len(analysis["events"]),
        "latest": _latest_summary(analysis),
    }


def main():
    parser = argparse.ArgumentParser(description="流感预警报告生成服务入口")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="不调用 Ollama，使用规则底稿")
    parser.add_argument("--no-pdf", action="store_true", help="只生成 Markdown，不生成 PDF")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结构化结果")
    parser.add_argument("--quiet", action="store_true", help="不打印过程日志")
    args = parser.parse_args()

    result = generate_report(
        config_path=args.config,
        dry_run=args.dry_run,
        make_pdf=not args.no_pdf,
        verbose=not args.quiet and not args.json,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Markdown → {result['markdown_path']}")
        if result["pdf_path"]:
            print(f"PDF → {result['pdf_path']}")
        elif result["pdf_error"]:
            print(f"PDF 生成失败：{result['pdf_error']}")


if __name__ == "__main__":
    main()
