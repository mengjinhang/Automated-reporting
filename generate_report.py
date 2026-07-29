#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流感预警综合分析报告自动生成
- 数值与表格由代码精确计算，LLM 仅生成叙述性段落，避免数值幻觉。
用法:
    python3 generate_report.py                 # 完整流程（需 Ollama 可用）
    python3 generate_report.py --dry-run       # 用占位文本，仅验证数据/图表
    python3 generate_report.py --config x.yaml
"""
import argparse
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def find_peaks(x, distance=1, prominence=0):
    """轻量峰值检测（替代 scipy.signal.find_peaks，避免额外依赖）。
    - 局部极大值：x[i] > 两侧邻居
    - prominence：峰值相对左右最近谷底的最小高出量
    - distance：相邻峰值最小间隔，冲突时保留更高者
    返回 (峰值索引数组, {})。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    cand = [i for i in range(1, n - 1) if x[i] > x[i - 1] and x[i] >= x[i + 1]]

    def _prom(i):
        # 向左找到不低于当前峰的更高点或边界，取区间最小值作为左基
        j = i - 1
        left_min = x[i]
        while j >= 0 and x[j] < x[i]:
            left_min = min(left_min, x[j]); j -= 1
        j = i + 1
        right_min = x[i]
        while j < n and x[j] < x[i]:
            right_min = min(right_min, x[j]); j += 1
        base = max(left_min, right_min)
        return x[i] - base

    cand = [i for i in cand if _prom(i) >= prominence]
    cand.sort(key=lambda i: -x[i])
    kept = []
    for i in cand:
        if all(abs(i - k) >= distance for k in kept):
            kept.append(i)
    kept.sort()
    return np.array(kept, dtype=int), {}

# ----------------------------------------------------------------------------
# 中文字体
# ----------------------------------------------------------------------------
def setup_font():
    for name in ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "STHeiti", "Heiti TC", "Songti SC"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------------
# 配置与数据加载
# ----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve(p):
    """相对路径按脚本所在目录解析，保证跨机器/任意工作目录可运行。"""
    return p if os.path.isabs(p) else os.path.join(SCRIPT_DIR, p)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["base_dir"] = _resolve(cfg["data"]["base_dir"])
    cfg["output"]["dir"] = _resolve(cfg["output"]["dir"])
    return cfg


def load_data(cfg):
    d = cfg["data"]
    base = d["base_dir"]
    df = pd.read_excel(os.path.join(base, d["main_data"]))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["t"] = range(len(df))

    def _load_alerts(fname):
        a = pd.read_csv(os.path.join(base, fname))
        # 统一列名 -> index, date
        a.columns = ["index", "date"]
        a["date"] = pd.to_datetime(a["date"])
        return a.sort_values("index").reset_index(drop=True)

    return {
        "df": df,
        "cne": _load_alerts(d["cne_alerts"]),
        "trce": _load_alerts(d["trce_alerts"]),
        "fusion": _load_alerts(d["fusion_alerts"]),
    }


# ----------------------------------------------------------------------------
# 指标计算（用于图表下子图；预警点均来自真实 CSV）
# ----------------------------------------------------------------------------
def compute_network_entropy(df, window=14):
    """CNE 网络熵 H 代理：搜索词构成的分布熵的日变化。"""
    cols = [c for c in ["甲流", "奥司他韦", "流感", "玛巴沙洛韦"] if c in df.columns]
    m = df[cols].astype(float).clip(lower=0).values + 1e-9
    p = m / m.sum(axis=1, keepdims=True)
    ent = -(p * np.log(p)).sum(axis=1)             # 每日熵
    ent = pd.Series(ent).rolling(window, min_periods=1).mean()
    return ent.diff().fillna(0).values             # 熵的变化量，围绕 0 波动


def compute_trce_index(df, window=20, tau=1):
    """TRCE 时间不可逆指标代理：滚动窗口内增量的三阶矩不对称。"""
    x = df["ILI"].astype(float).values
    dx = np.zeros_like(x)
    dx[tau:] = x[tau:] - x[:-tau]
    s = pd.Series(dx)
    # 归一化的三阶矩（时间不可逆的经典度量），围绕 0 波动
    m3 = s.rolling(window, min_periods=3).apply(lambda v: np.mean(v**3), raw=True)
    scale = (s.rolling(window, min_periods=3).std() ** 3).replace(0, np.nan)
    idx = (m3 / scale / 50.0).fillna(0)
    return idx.values


# ----------------------------------------------------------------------------
# 疫情事件识别与分析
# ----------------------------------------------------------------------------
RISK_BY_MODEL_COUNT = {3: "高可信预警", 2: "风险持续增强", 1: "弱信号预警", 0: "未触发"}


def _fmt_date(ts):
    d = pd.Timestamp(ts)
    return f"{d.year}年{d.month}月{d.day}日"


def analyze_flu_events(data, cfg):
    df = data["df"]
    ili = df["ILI"].values
    pk = cfg["peak_detection"]
    peaks, _ = find_peaks(ili, distance=pk["distance"], prominence=pk["prominence"])

    models = {"CNE": data["cne"], "TRCE": data["trce"], "Fusion": data["fusion"]}
    t2date = dict(zip(df["t"], df["date"]))

    def nearest_date(t):
        t = int(np.clip(t, 0, len(df) - 1))
        return _fmt_date(t2date[t])

    events = []
    pre_window = 130  # 峰值前多少天内的预警视为对应本次事件
    for i, peak_t in enumerate(peaks):
        row = {"event_id": i + 1, "peak_t": int(peak_t),
               "peak_date": nearest_date(peak_t), "peak_ili": float(ili[peak_t])}
        first_alerts = []
        for mname, adf in models.items():
            win = adf[(adf["index"] >= peak_t - pre_window) & (adf["index"] <= peak_t)]
            triggered = len(win) > 0
            row[mname] = triggered
            if triggered:
                first_alerts.append(int(win["index"].min()))
        if first_alerts:
            row["first_alert_t"] = min(first_alerts)
            row["first_alert_date"] = nearest_date(row["first_alert_t"])
            row["lead_time"] = int(peak_t - row["first_alert_t"])
        else:
            row["first_alert_t"] = None
            row["first_alert_date"] = None
            row["lead_time"] = None
        n = sum(row[m] for m in models)
        row["risk_level"] = RISK_BY_MODEL_COUNT[n]
        events.append(row)

    # 仅保留至少有一个模型预警的事件
    events = [e for e in events if e["first_alert_t"] is not None]

    # ---- 最近一次预警 ----
    latest_t = max(m["index"].max() for m in models.values())
    triggered, untriggered = [], []
    for mname, adf in models.items():
        # 最近预警时间窗内 ±10 天是否有该模型预警
        if ((adf["index"] >= latest_t - 10) & (adf["index"] <= latest_t + 10)).any():
            triggered.append(mname)
        else:
            untriggered.append(mname)

    # 最近预警对应的（下一个）峰值
    future_peaks = [p for p in peaks if p >= latest_t]
    if future_peaks:
        near_peak = int(future_peaks[0])
        peak_gap = int(near_peak - latest_t)
        peak_desc = f"约{peak_gap}天"
        peak_date = nearest_date(near_peak)
    else:
        # 最近预警已晚于所有已知峰值：取最近的历史峰值作参照
        near_peak = int(peaks[-1]) if len(peaks) else None
        peak_gap = None
        peak_desc = "峰值尚未在监测区间内出现"
        peak_date = None

    # 风险等级：Fusion 触发 -> 橙色预警；仅 TRCE/CNE -> 黄色预警
    if "Fusion" in triggered:
        risk_color = "橙色预警"
    elif triggered:
        risk_color = "黄色预警"
    else:
        risk_color = "蓝色预警"

    latest = {
        "t": int(latest_t),
        "date": nearest_date(latest_t),
        "triggered": triggered,
        "untriggered": untriggered,
        "risk_color": risk_color,
        "near_peak_t": near_peak,
        "near_peak_date": peak_date,
        "peak_gap": peak_gap,
        "peak_desc": peak_desc,
    }

    dmin, dmax = df['date'].min(), df['date'].max()
    period = f"{dmin.year}年{dmin.month}月—{dmax.year}年{dmax.month}月"
    return {"events": events, "latest": latest, "data_period": period,
            "peaks": [int(p) for p in peaks]}


# ----------------------------------------------------------------------------
# 图表生成
# ----------------------------------------------------------------------------
def _alert_y_on_curve(df, alerts):
    """取预警点在 ILI 曲线上的 y 值。"""
    t2ili = dict(zip(df["t"], df["ILI"]))
    xs, ys = [], []
    for idx in alerts["index"]:
        idx = int(np.clip(idx, 0, len(df) - 1))
        xs.append(df["date"].iloc[idx])
        ys.append(t2ili.get(idx, np.nan))
    return xs, ys


def generate_cne_chart(data, path):
    df, alerts = data["df"], data["cne"]
    H = compute_network_entropy(df)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(df["date"], df["ILI"], color="#1f6fb2", lw=1.0, label="ILI")
    xs, ys = _alert_y_on_curve(df, alerts)
    ax1.scatter(xs, ys, color="red", s=45, zorder=5, label="预警点")
    ax1.set_ylabel("ILI病例数"); ax1.set_title("CNE监测结果"); ax1.legend(loc="upper right")

    ax2.plot(df["date"], H, color="#e8890c", lw=0.8, label="网络熵H")
    ax2.scatter(xs, [H[int(np.clip(i, 0, len(df)-1))] for i in alerts["index"]],
                color="red", s=45, zorder=5, label="预警点")
    ax2.axhline(0, color="grey", lw=0.5)
    ax2.set_ylabel("网络熵H"); ax2.set_xlabel("日期"); ax2.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def generate_trce_chart(data, path):
    df, alerts = data["df"], data["trce"]
    idx = compute_trce_index(df)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(df["date"], df["ILI"], color="#1f6fb2", lw=1.0, label="ILI")
    xs, ys = _alert_y_on_curve(df, alerts)
    ax1.scatter(xs, ys, color="#7b3fa0", marker="^", s=60, zorder=5, label="预警点")
    ax1.set_ylabel("ILI病例数"); ax1.set_title("tRCE监测结果"); ax1.legend(loc="upper right")

    ax2.plot(df["date"], idx, color="#2ca02c", lw=0.8, label="tRCE指标")
    ax2.scatter(xs, [idx[int(np.clip(i, 0, len(df)-1))] for i in alerts["index"]],
                color="#7b3fa0", marker="^", s=60, zorder=5, label="预警点")
    ax2.axhline(0, color="grey", lw=0.5)
    ax2.set_ylabel("tRCE指标"); ax2.set_xlabel("日期"); ax2.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def generate_fusion_chart(data, path):
    df, alerts = data["df"], data["fusion"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(df["date"], df["ILI"], color="#1f6fb2", lw=1.0, label="ILI")
    xs, ys = _alert_y_on_curve(df, alerts)
    # 融合图：重点(critical)预警统一按橙/红显示
    ax1.scatter(xs, ys, color="#e8890c", s=55, zorder=5, label="融合预警")
    ax1.set_ylabel("ILI病例值"); ax1.set_title("多源融合预警结果"); ax1.legend(loc="upper right")

    ax2.plot(df["date"], df["ILI"], color="#1f6fb2", lw=1.0, label="ILI")
    ax2.scatter(xs, ys, color="#e03b3b", s=55, zorder=5,
                label=f"重点预警({len(alerts)}个)")
    # 预警趋势线（相邻重点预警点连线）
    if len(xs) >= 2:
        ax2.plot(xs, ys, color="#e03b3b", ls="--", lw=0.8, alpha=0.6, label="预警趋势")
    ax2.set_ylabel("ILI病例值"); ax2.set_xlabel("日期"); ax2.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def generate_charts(data, chart_dir):
    os.makedirs(chart_dir, exist_ok=True)
    paths = {
        "cne": os.path.join(chart_dir, "cne_chart.png"),
        "trce": os.path.join(chart_dir, "trce_chart.png"),
        "fusion": os.path.join(chart_dir, "fusion_chart.png"),
    }
    generate_cne_chart(data, paths["cne"])
    generate_trce_chart(data, paths["trce"])
    generate_fusion_chart(data, paths["fusion"])
    return paths


# ----------------------------------------------------------------------------
# Ollama 调用
# ----------------------------------------------------------------------------
def call_ollama(prompt, cfg, dry_run=False):
    if dry_run:
        return "【占位文本 · dry-run 模式，未调用 LLM】"
    o = cfg["ollama"]
    try:
        r = requests.post(
            f"{o['base_url']}/api/generate",
            json={"model": o["model"], "prompt": prompt, "stream": False,
                  "options": {"temperature": o.get("temperature", 0.3)}},
            timeout=o.get("timeout", 120),
        )
        r.raise_for_status()
        text = r.json()["response"]
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text
    except Exception as e:
        return f"【LLM 调用失败：{e}】"


SYS = ("你是一名资深流行病学监测分析专家，擅长撰写传染病预警分析报告。"
       "请基于给定的数据事实撰写专业、严谨、书面化的中文分析文字，"
       "不要编造数据中未提供的数字，不要使用markdown标题或列表，只输出连续段落。")


def gen_narratives(analysis, cfg, dry_run=False):
    latest = analysis["latest"]
    trig = "、".join(latest["triggered"]) or "无"
    untrig = "、".join(latest["untriggered"]) or "无"

    p_risk = f"""{SYS}
【任务】撰写“当前风险判断”段落（150字以内）。
【事实】最近一次预警时间 {latest['date']}（t={latest['t']}）；
触发模型：{trig}；未触发模型：{untrig}；当前风险等级：{latest['risk_color']}；
距病例峰值情况：{latest['peak_desc']}。
请说明当前流感传播系统的风险状态与演化趋势。"""

    p_signal = f"""{SYS}
【任务】撰写“结合监测图表的模型信号解读”段落，分别解读 CNE、TRCE、Fusion 三张图（每个模型2-3句）。
【事实】触发模型：{trig}；未触发模型：{untrig}。
CNE为因果网络熵模型（看网络熵H是否冲高、预警红点是否密集）；
TRCE为时间不可逆模型（看tRCE指标偏移与三角预警）；
Fusion为多源融合模型（整合双模型输出，给出分级预警点与风险趋势）。
请结合本次触发情况解读三张图的信号含义。"""

    p_conf = f"""{SYS}
【任务】撰写“预警可信度分析”段落（120字以内）。
【事实】触发模型：{trig}；未触发模型：{untrig}。
请说明多模型协同/分歧所反映的风险性质（如动力学变化 vs 因果网络变化），以及本次预警的可信度。"""

    p_advice = f"""{SYS}
【任务】撰写“防控建议”段落（150字以内）。
【事实】当前风险等级：{latest['risk_color']}；触发模型：{trig}；{latest['peak_desc']}。
请给出针对性的疫情监测与防控建议。"""

    return {
        "risk": call_ollama(p_risk, cfg, dry_run),
        "signal": call_ollama(p_signal, cfg, dry_run),
        "confidence": call_ollama(p_conf, cfg, dry_run),
        "advice": call_ollama(p_advice, cfg, dry_run),
    }


# ----------------------------------------------------------------------------
# Markdown 报告组装
# ----------------------------------------------------------------------------
def _mark(b):
    return "√" if b else "×"


def build_report(analysis, narr, chart_paths, cfg, out_dir):
    r = cfg["report"]
    latest = analysis["latest"]
    trig = "、".join(latest["triggered"]) or "无"
    untrig = "、".join(latest["untriggered"]) or "无"
    rel = lambda p: os.path.relpath(p, out_dir)

    L = []
    L.append("# 传染病智能预警综合分析报告\n")

    L.append("## 一、基本信息\n")
    L.append("| 项目 | 内容 |")
    L.append("|------|------|")
    L.append(f"| 监测疾病 | {r['disease']} |")
    L.append(f"| 监测区域 | {r['region']} |")
    L.append(f"| 数据周期 | {analysis['data_period']} |")
    L.append(f"| 数据来源 | {r['data_source']} |")
    L.append(f"| 分析模型 | {'、'.join(r['models'])} |\n")

    L.append("## 二、当前风险概况\n")
    L.append("### 1. 最近一次预警信息\n")
    L.append("| 项目 | 分析结果 |")
    L.append("|------|----------|")
    L.append(f"| 最近一次预警时间 | {latest['date']} |")
    L.append(f"| 对应时间节点 | t={latest['t']} |")
    L.append(f"| 触发模型 | {trig} |")
    L.append(f"| 未触发模型 | {untrig} |")
    L.append(f"| 当前风险等级 | {latest['risk_color']} |")
    L.append(f"| 距病例峰值 | {latest['peak_desc']} |\n")

    L.append("### 2. 当前风险判断\n")
    L.append(narr["risk"] + "\n")

    L.append("### 3. 结合监测图表的模型信号解读\n")
    L.append(f"![CNE监测结果]({rel(chart_paths['cne'])})\n")
    L.append(f"![TRCE监测结果]({rel(chart_paths['trce'])})\n")
    L.append(f"![Fusion融合预警结果]({rel(chart_paths['fusion'])})\n")
    L.append(narr["signal"] + "\n")

    L.append("### 4. 预警可信度分析\n")
    L.append("| 模型 | 分析结果 |")
    L.append("|------|----------|")
    L.append(f"| CNE因果网络熵模型 | {'✓ 检测到异常变化' if 'CNE' in latest['triggered'] else '未触发'} |")
    L.append(f"| TRCE时间不可逆模型 | {'✓ 检测到异常变化' if 'TRCE' in latest['triggered'] else '未触发'} |")
    L.append(f"| Fusion多源融合模型 | {'✓ 输出' + latest['risk_color'] if 'Fusion' in latest['triggered'] else '未触发'} |\n")
    L.append(narr["confidence"] + "\n")

    L.append("### 5. 历史预警一致性参考\n")
    L.append("| 疫情事件 | 首次预警时间 | 峰值时间 | CNE | TRCE | Fusion | 综合判断 | 提前时间 |")
    L.append("|----------|-------------|---------|-----|------|--------|---------|---------|")
    cn = ["", "第一次", "第二次", "第三次", "第四次", "第五次", "第六次",
          "第七次", "第八次", "第九次", "第十次"]
    for e in analysis["events"]:
        name = (cn[e["event_id"]] if e["event_id"] < len(cn) else f"第{e['event_id']}") + "疫情"
        L.append(f"| {name} | t={e['first_alert_t']} | t={e['peak_t']} | "
                 f"{_mark(e['CNE'])} | {_mark(e['TRCE'])} | {_mark(e['Fusion'])} | "
                 f"{e['risk_level']} | {e['lead_time']}天 |")
    L.append("")

    L.append("## 三、防控建议\n")
    L.append(narr["advice"] + "\n")

    L.append(f"\n---\n*报告生成时间：{cfg.get('_now', '')}*")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="不调用 LLM，用占位文本")
    args = ap.parse_args()

    setup_font()
    cfg = load_config(args.config)
    now = datetime.now()
    cfg["_now"] = now.strftime("%Y-%m-%d %H:%M:%S")

    print("[1/5] 加载数据 ...")
    data = load_data(cfg)
    print(f"      主数据 {len(data['df'])} 行，CNE/TRCE/Fusion 预警 "
          f"{len(data['cne'])}/{len(data['trce'])}/{len(data['fusion'])} 条")

    print("[2/5] 分析疫情事件 ...")
    analysis = analyze_flu_events(data, cfg)
    print(f"      识别疫情事件 {len(analysis['events'])} 次；"
          f"最近预警 {analysis['latest']['date']}（{analysis['latest']['risk_color']}）")

    print("[3/5] 生成图表 ...")
    out_dir = cfg["output"]["dir"]
    chart_paths = generate_charts(data, os.path.join(out_dir, "charts"))

    print("[4/5] 生成叙述文本" + ("（dry-run 占位）" if args.dry_run else "（调用 Ollama）") + " ...")
    narr = gen_narratives(analysis, cfg, dry_run=args.dry_run)

    print("[5/5] 组装报告 ...")
    md = build_report(analysis, narr, chart_paths, cfg, out_dir)
    out_path = os.path.join(out_dir, f"report_{now.strftime('%Y%m%d_%H%M')}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"完成 → {out_path}")


if __name__ == "__main__":
    main()
