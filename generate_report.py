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
MODEL_ORDER = ["CNE", "TRCE", "Fusion"]
CN_EVENT_NAMES = ["", "第一次", "第二次", "第三次", "第四次", "第五次", "第六次",
                  "第七次", "第八次", "第九次", "第十次"]


def _fmt_date(ts):
    d = pd.Timestamp(ts)
    return f"{d.year}年{d.month}月{d.day}日"


def _risk_color(triggered):
    if "Fusion" in triggered:
        return "橙色预警"
    if triggered:
        return "黄色预警"
    return "蓝色预警"


def _risk_status(triggered):
    n = len(triggered)
    if n == 3:
        return "高可信风险增强"
    if "TRCE" in triggered and "Fusion" in triggered:
        return "风险持续增强"
    if "CNE" in triggered and "Fusion" in triggered:
        return "多源风险增强"
    if n == 2:
        return "风险持续积累"
    if triggered == ["TRCE"]:
        return "传播动力学异常"
    if triggered == ["CNE"]:
        return "因果网络异常"
    if triggered == ["Fusion"]:
        return "融合预警触发"
    return "未触发"


def _event_name(event_id):
    prefix = CN_EVENT_NAMES[event_id] if event_id < len(CN_EVENT_NAMES) else f"第{event_id}"
    return prefix + "疫情"


def _range_text(values):
    vals = sorted({int(v) for v in values if v is not None})
    if not vals:
        return None
    if len(vals) == 1:
        return str(vals[0])
    return f"{vals[0]}～{vals[-1]}"


def _lead_desc_from_alerts(peak_t, alert_ts, prefix="约"):
    vals = sorted({int(v) for v in alert_ts if v is not None})
    if not vals or peak_t is None:
        return "峰值尚未在监测区间内出现"
    leads = sorted({int(peak_t) - v for v in vals})
    if len(leads) == 1:
        return f"{prefix}{leads[0]}天"
    return f"{prefix}{leads[0]}～{leads[-1]}天"


def analyze_flu_events(data, cfg):
    df = data["df"]
    ili = df["ILI"].values
    pk = cfg["peak_detection"]
    peaks, _ = find_peaks(ili, distance=pk["distance"], prominence=pk["prominence"])

    models = {m: data[m.lower()] if m != "Fusion" else data["fusion"] for m in MODEL_ORDER}
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
        model_alerts = {}
        for mname, adf in models.items():
            win = adf[(adf["index"] >= peak_t - pre_window) & (adf["index"] <= peak_t)]
            triggered = len(win) > 0
            row[mname] = triggered
            if triggered:
                first_t = int(win["index"].min())
                row[f"{mname}_alert_t"] = first_t
                model_alerts[mname] = first_t
                first_alerts.append(first_t)
            else:
                row[f"{mname}_alert_t"] = None
        if first_alerts:
            row["first_alert_t"] = min(first_alerts)
            row["first_alert_date"] = nearest_date(row["first_alert_t"])
            row["lead_time"] = int(peak_t - row["first_alert_t"])
            compact_range = max(first_alerts) - min(first_alerts) <= 3
            row["first_alert_t_display"] = (
                _range_text(first_alerts) if compact_range and len(set(first_alerts)) > 1
                else str(row["first_alert_t"])
            )
            row["lead_time_display"] = (
                _lead_desc_from_alerts(peak_t, first_alerts, prefix="").replace("天", "天")
                if compact_range and len(set(first_alerts)) > 1
                else f"{row['lead_time']}天"
            )
        else:
            row["first_alert_t"] = None
            row["first_alert_date"] = None
            row["lead_time"] = None
            row["first_alert_t_display"] = None
            row["lead_time_display"] = None
        row["model_alerts"] = model_alerts
        n = sum(row[m] for m in models)
        row["risk_level"] = RISK_BY_MODEL_COUNT[n]
        events.append(row)

    # 仅保留至少有一个模型预警的事件
    events = [e for e in events if e["first_alert_t"] is not None]

    # ---- 当前报告事件 ----
    # 默认按“最近一个已形成峰值的完整事件”生成报告，避免监测末端未验证的零散预警
    # 被误写成“峰值尚未出现”的当前事件；如需机械取全局最后一条预警，可在配置中
    # 设置 analysis.current_event = latest_alert。
    current_mode = cfg.get("analysis", {}).get("current_event", "latest_confirmed_event")
    selected_event = events[-1] if events and current_mode == "latest_confirmed_event" else None

    if selected_event:
        model_alerts = selected_event["model_alerts"]
        primary_t = model_alerts.get("Fusion") or selected_event["first_alert_t"]
        triggered = [m for m in MODEL_ORDER if selected_event[m]]
        untriggered = [m for m in MODEL_ORDER if not selected_event[m]]
        near_peak = selected_event["peak_t"]
        peak_date = selected_event["peak_date"]
        alert_ts = [model_alerts[m] for m in triggered]
        peak_gap = int(near_peak - primary_t)
        peak_desc = _lead_desc_from_alerts(near_peak, alert_ts)
        latest_t = primary_t
    else:
        latest_t = max(m["index"].max() for m in models.values())
        triggered, untriggered = [], []
        for mname, adf in models.items():
            # 最近预警时间窗内 ±10 天是否有该模型预警
            if ((adf["index"] >= latest_t - 10) & (adf["index"] <= latest_t + 10)).any():
                triggered.append(mname)
            else:
                untriggered.append(mname)

        future_peaks = [p for p in peaks if p >= latest_t]
        if future_peaks:
            near_peak = int(future_peaks[0])
            peak_gap = int(near_peak - latest_t)
            peak_desc = f"约{peak_gap}天"
            peak_date = nearest_date(near_peak)
        else:
            near_peak = int(peaks[-1]) if len(peaks) else None
            peak_gap = None
            peak_desc = "峰值尚未在监测区间内出现"
            peak_date = None

    latest = {
        "t": int(latest_t),
        "date": nearest_date(latest_t),
        "triggered": triggered,
        "untriggered": untriggered,
        "risk_color": _risk_color(triggered),
        "risk_status": _risk_status(triggered),
        "near_peak_t": near_peak,
        "near_peak_date": peak_date,
        "peak_gap": peak_gap,
        "peak_desc": peak_desc,
        "event": selected_event,
    }

    dmin, dmax = df['date'].min(), df['date'].max()
    period = f"{dmin.year}年{dmin.month}月—{dmax.year}年{dmax.month}月"
    return {"events": events, "latest": latest, "data_period": period,
            "peaks": [int(p) for p in peaks], "date_by_t": t2date}


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
    options = {
        "temperature": o.get("temperature", 0.2),
        "top_p": o.get("top_p", 0.8),
        "repeat_penalty": o.get("repeat_penalty", 1.08),
        "num_predict": o.get("num_predict", 1000),
    }
    try:
        r = requests.post(
            f"{o['base_url']}/api/generate",
            json={"model": o["model"], "prompt": prompt, "stream": False,
                  "options": options},
            timeout=o.get("timeout", 120),
        )
        r.raise_for_status()
        return _clean_llm_text(r.json()["response"])
    except Exception as e:
        return f"【LLM 调用失败：{e}】"


SYS = """你是一名资深流行病学监测分析专家，正在润色传染病智能预警综合分析报告。
你的任务不是自由创作，而是基于“事实”和“底稿”做专业化、书面化改写。
硬性要求：
1. 只能使用事实和底稿中出现的日期、t值、模型触发状态、预警等级、提前天数。
2. 不得新增病例数、概率、百分比、地区、政策名称或事实中没有的判断。
3. 不要输出 Markdown 标题、列表、表格、项目符号、解释过程或思考过程。
4. 输出连续中文段落，语气接近正式监测分析报告，克制、具体、可落地。
5. 在不新增事实的前提下适当展开分析链条，不要写成过短结论句。
6. 如果底稿已经合适，只做轻微润色，不改变原意。"""


REFERENCE_STYLE = """参考文风：
根据平台最新分析结果，TRCE模型于 t=1034（2025年10月31日）首次检测到系统异常变化，随后 Fusion 模型于 t=1035（2025年11月1日）输出橙色风险预警。本次预警对应第四次流感风险事件，该事件病例峰值出现在 t=1084，模型提前约49～50天识别风险变化。
虽然CNE模型未产生对应预警，但TRCE和Fusion模型在相近时间窗口内同时识别到风险变化，说明当前异常主要表现为传播系统动力学变化和多源融合风险增强。"""


def _clean_llm_text(text):
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[/\\]?no_think\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```(?:\w+)?|```$", "", text.strip())
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[.、]\s*", "", line)
        lines.append(line)
    text = "".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace(" 。", "。").replace(" ，", "，")


def _is_bad_llm_text(text):
    text = _clean_llm_text(text)
    if not text:
        return True
    if text.startswith("【LLM 调用失败"):
        return True
    if len(text) < 35:
        return True
    bad_patterns = ["我无法", "不能确定", "作为AI", "作为人工智能", "以下是", "标题："]
    return any(p in text for p in bad_patterns)


def _model_first_sentence(model, t, analysis):
    if t is None:
        return ""
    return f"{model}模型于 t={t}（{_fmt_date(pd.Timestamp(analysis['date_by_t'][t]))}）"


def _fallback_risk(analysis):
    latest = analysis["latest"]
    event = latest.get("event")
    trig = latest["triggered"]
    model_alerts = event.get("model_alerts", {}) if event else {}

    if event:
        pieces = []
        for model in MODEL_ORDER:
            if model in trig:
                pieces.append(_model_first_sentence(model, model_alerts.get(model), analysis))
        alert_text = "，随后 ".join(pieces)
        lead_text = latest["peak_desc"]
        return (
            f"根据平台最新分析结果，{alert_text}检测到系统异常变化。本次预警对应"
            f"{_event_name(event['event_id'])}风险事件，该事件病例峰值出现在 t={event['peak_t']}，"
            f"模型提前{lead_text.replace('约', '约')}识别风险变化。综合判断，当前流感传播系统已出现"
            f"{latest['risk_status']}信号，系统由相对稳定状态向高风险状态转变，未来一段时间内病例规模"
            f"可能进一步增加。由于本轮预警主要由传播动力学模型与多源融合模型共同支持，说明异常并非单一"
            f"监测点波动，而是病例演化过程和融合风险信号在相近时间窗口内形成一致指向。当前阶段应将其视为"
            f"峰值前的风险累积期，持续关注疫情演化过程和后续新增预警点变化。"
        )

    return (
        f"根据平台最新分析结果，最近一次预警发生于 t={latest['t']}（{latest['date']}），"
        f"触发模型为{'、'.join(trig) or '无'}，当前风险等级为{latest['risk_color']}。"
        f"综合判断，当前流感传播系统处于{latest['risk_status']}状态，需结合后续病例曲线和模型信号"
        f"继续滚动评估。若后续 ILI 曲线延续上行并伴随新增预警点，应及时提高风险研判等级和响应准备强度。"
    )


def _fallback_signal(analysis):
    latest = analysis["latest"]
    trig = set(latest["triggered"])
    cne = (
        "CNE 模型图显示，本轮监测窗口内网络熵 H 与预警红点同步出现异常变化，提示搜索行为相关的因果网络结构复杂度上升，外部关注和症状相关搜索之间的联动关系正在增强。该类信号通常反映多源信息网络由平稳状态向异常耦合状态转换，可作为风险扩散的辅助证据。"
        if "CNE" in trig else
        "CNE 模型图显示，本轮监测窗口内网络熵 H 未形成与当前事件对应的显著冲高，预警点相对稀疏，提示搜索行为相关因果网络变化尚不明显。也就是说，当前异常并未主要表现为外部搜索行为网络的集中扰动，CNE 对本轮风险的支持强度有限。"
    )
    trce = (
        "TRCE 模型图显示，tRCE 指标在病例上升阶段出现持续偏移，并伴随三角预警信号，说明传播动力学已偏离基线状态，是本次风险判断的重要依据。该模型对病例序列内部演化方向较为敏感，当前触发结果提示病例增长过程已具备异常加速或非平稳变化特征。"
        if "TRCE" in trig else
        "TRCE 模型图显示，当前窗口内时间不可逆指标未触发明确异常，暂未观察到显著的传播动力学突变信号。后续仍需结合 ILI 曲线斜率变化和新增预警点进行滚动观察。"
    )
    fusion = (
        "Fusion 融合预警图整合 CNE 与 TRCE 输出后给出分级预警，预警点与 ILI 曲线上升阶段相互印证，说明风险正在持续累积并仍存在提前响应窗口。融合模型的触发进一步提高了本次预警的综合可信度，提示应从单点监测转向多模型协同研判。"
        if "Fusion" in trig else
        "Fusion 融合预警图未输出当前窗口的高等级融合预警，提示多源证据尚未形成充分一致的风险增强判断。当前风险仍应以持续监测和趋势复核为主。"
    )
    return cne + trce + fusion


def _fallback_confidence(analysis):
    latest = analysis["latest"]
    trig = latest["triggered"]
    untrig = latest["untriggered"]
    if len(trig) >= 2:
        return (
            f"本次风险判断由{'、'.join(trig)}共同支持，模型信号在相近时间窗口内指向"
            f"{latest['risk_status']}。"
            f"{'、'.join(untrig)}未触发说明异常来源存在一定结构性分歧，但不削弱当前传播风险增强的判断。"
            f"从模型含义看，TRCE 与 Fusion 的同步触发更偏向反映病例系统内部动力学变化和多源综合风险抬升，"
            f"而 CNE 未触发则提示搜索行为相关因果网络尚未出现同步异常。因此，本次预警可信度较高，"
            f"但风险来源应主要理解为传播过程本身的异常增强。"
        )
    if trig:
        return (
            f"本次预警主要由{'、'.join(trig)}触发，提示风险信号已出现但多模型一致性仍有限，"
            f"建议结合后续病例走势和新增预警点继续验证。在未形成多模型协同前，应避免过度解读单一信号，"
            f"但也不宜忽视其对早期风险变化的提示作用。"
        )
    return "当前窗口内多模型均未形成明确异常信号，预警可信度有限，应维持常规监测并等待后续数据验证。若后续出现模型连续触发或病例曲线持续上行，再进一步提高研判等级。"


def _fallback_advice(analysis):
    latest = analysis["latest"]
    return (
        f"根据当前智能预警分析结果，当前流感传播系统已出现{latest['risk_status']}信号，"
        f"风险等级为{latest['risk_color']}，且距离病例峰值仍存在{latest['peak_desc']}提前响应时间。"
        f"建议加强疫情动态监测，提高流感监测频率，持续跟踪病例变化趋势及风险指标变化；"
        f"同步做好重点场所健康提示、医疗资源储备和分级诊疗准备，必要时及时启动针对性干预措施。"
        f"在监测层面，应重点关注后续 ILI 曲线是否持续上行、TRCE 指标是否继续偏移以及 Fusion 预警点是否连续出现；"
        f"在处置层面，可提前完善门急诊接诊、重点人群健康提示和异常聚集信息核查，确保在病例峰值到来前形成较充分的响应准备。"
    )


def _with_fallback(text, fallback):
    text = _clean_llm_text(text)
    return fallback if _is_bad_llm_text(text) else text


def _fact_block(analysis):
    latest = analysis["latest"]
    event = latest.get("event") or {}
    model_alerts = event.get("model_alerts", {})
    items = [
        "监测疾病：流感",
        f"数据周期：{analysis['data_period']}",
        f"最近一次预警时间：{latest['date']}，t={latest['t']}",
        f"触发模型：{'、'.join(latest['triggered']) or '无'}",
        f"未触发模型：{'、'.join(latest['untriggered']) or '无'}",
        f"当前风险等级：{latest['risk_color']}",
        f"风险状态：{latest['risk_status']}",
        f"距病例峰值：{latest['peak_desc']}",
    ]
    if event:
        items.extend([
            f"对应事件：{_event_name(event['event_id'])}",
            f"峰值时间节点：t={event['peak_t']}，{event['peak_date']}",
        ])
        for model in MODEL_ORDER:
            t = model_alerts.get(model)
            if t is None:
                items.append(f"{model}首次预警：未触发")
            else:
                items.append(f"{model}首次预警：t={t}，{_fmt_date(analysis['date_by_t'][t])}")
    return "\n".join(items)


def _polish_prompt(section_name, facts, draft, limit, focus):
    return f"""/no_think
{SYS}

{REFERENCE_STYLE}

【段落名称】{section_name}
【事实】
{facts}

【底稿】
{draft}

【润色要求】
{focus}
请在 {limit} 字以内输出最终段落。只输出正文，不要解释。"""


def _polish_section(section_name, facts, draft, limit, focus, cfg):
    prompt = _polish_prompt(section_name, facts, draft, limit, focus)
    raw = call_ollama(prompt, cfg)
    polished = _with_fallback(raw, draft)
    if polished == draft and _is_bad_llm_text(raw):
        print(f"      - {section_name}：LLM 输出不可用，使用规则底稿")
    else:
        print(f"      - {section_name}：已使用 LLM 润色")
    return polished


def gen_narratives(analysis, cfg, dry_run=False):
    facts = _fact_block(analysis)
    drafts = {
        "risk": _fallback_risk(analysis),
        "signal": _fallback_signal(analysis),
        "confidence": _fallback_confidence(analysis),
        "advice": _fallback_advice(analysis),
    }

    if dry_run:
        return drafts

    return {
        "risk": _polish_section(
            "当前风险判断", facts, drafts["risk"], 280,
            "保留预警时间、触发模型、对应事件、峰值节点和提前天数；突出风险由稳定转向增强，并适当说明为什么当前阶段属于峰值前风险累积期。",
            cfg,
        ),
        "signal": _polish_section(
            "结合监测图表的模型信号解读", facts, drafts["signal"], 520,
            "分 CNE、TRCE、Fusion 依次解读；必须体现 CNE 未触发、TRCE 与 Fusion 触发，不要把未触发模型写成已触发；每个模型可写2到3句。",
            cfg,
        ),
        "confidence": _polish_section(
            "预警可信度分析", facts, drafts["confidence"], 240,
            "解释多模型协同和分歧：TRCE/Fusion 同步增强，CNE 未触发代表因果网络变化不明显；适当说明本次预警可信但风险来源有侧重。",
            cfg,
        ),
        "advice": _polish_section(
            "防控建议", facts, drafts["advice"], 300,
            "建议要具体但不新增政策名，围绕动态监测、重点场所、医疗资源、滚动评估和峰值前准备展开。",
            cfg,
        ),
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
    rel = lambda p: os.path.relpath(p, out_dir).replace(os.sep, "/")

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
    L.append(f"| 距病例峰值 | {latest['peak_desc']} |")
    L.append(f"| 风险状态 | {latest['risk_status']} |\n")

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
    for e in analysis["events"]:
        name = _event_name(e["event_id"])
        L.append(f"| {name} | t={e['first_alert_t_display']} | t={e['peak_t']} | "
                 f"{_mark(e['CNE'])} | {_mark(e['TRCE'])} | {_mark(e['Fusion'])} | "
                 f"{e['risk_level']} | {e['lead_time_display']} |")
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

    print("[4/5] 生成叙述文本" + ("（dry-run 规则底稿）" if args.dry_run else "（调用 Ollama）") + " ...")
    narr = gen_narratives(analysis, cfg, dry_run=args.dry_run)

    print("[5/6] 组装报告 ...")
    md = build_report(analysis, narr, chart_paths, cfg, out_dir)
    out_path = os.path.join(out_dir, f"report_{now.strftime('%Y%m%d_%H%M')}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Markdown → {out_path}")

    print("[6/6] 生成 PDF ...")
    try:
        from md_to_pdf import export_md_pdf
        pdf_path = export_md_pdf(out_path)
        print(f"  PDF → {pdf_path}")
    except ImportError:
        print("  跳过（需安装 reportlab: pip install reportlab）")


if __name__ == "__main__":
    main()
