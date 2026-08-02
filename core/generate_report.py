#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流感预警综合分析报告自动生成
- 数值与表格由代码精确计算，LLM 仅生成叙述性段落，避免数值幻觉。
用法:
    python3 service.py                         # 推荐入口（需 Ollama 可用）
    python3 service.py --dry-run               # 用规则底稿验证数据/图表
    python3 core/generate_report.py --config x.yaml
"""
import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime

# 报告输出文件名统一前缀:产物形如 <STEM>_<时间戳>.md/.pdf,如 流感预警综合分析报告_20260802_1130.pdf
REPORT_FILE_STEM = "流感预警综合分析报告"

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
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CORE_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config.yaml")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def _resolve(p, base_dir):
    """相对路径按配置文件所在目录解析，保证跨机器/任意工作目录可运行。"""
    return p if os.path.isabs(p) else os.path.join(base_dir, p)


def load_config(path):
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    config_dir = os.path.dirname(path)
    cfg["data"]["base_dir"] = _resolve(cfg["data"]["base_dir"], config_dir)
    cfg["output"]["dir"] = _resolve(cfg["output"]["dir"], config_dir)
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
OLLAMA_LAST_META = {}


def call_ollama_chat(messages, cfg, dry_run=False, think=None):
    global OLLAMA_LAST_META
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
        payload = {
            "model": o["model"],
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if think is not None:
            payload["think"] = think
        elif "think" in o:
            payload["think"] = o.get("think", False)
        started = time.monotonic()
        r = requests.post(
            f"{o['base_url']}/api/chat",
            json=payload,
            timeout=o.get("timeout", 120),
        )
        elapsed = time.monotonic() - started
        r.raise_for_status()
        result = r.json()
        message = result.get("message", {}) or {}
        text = message.get("content", "") or result.get("response", "")
        thinking = message.get("thinking", "") or result.get("thinking", "")
        OLLAMA_LAST_META = {
            "endpoint": "chat",
            "elapsed": elapsed,
            "think": payload.get("think"),
            "response_len": len(text),
            "thinking_len": len(thinking),
            "response_preview": _clean_llm_text(text)[:120],
            "thinking_preview": thinking.replace("\n", " ")[:120],
            "total_duration": result.get("total_duration"),
            "eval_count": result.get("eval_count"),
        }
        if not text and thinking:
            return "【LLM 响应正文为空，仅返回 thinking 字段】"
        return text
    except Exception as e:
        OLLAMA_LAST_META = {"endpoint": "chat", "error": str(e)}
        return f"【LLM 调用失败：{e}】"


SYS = """你是一名 CDC 流感/传染病监测分析专家，正在撰写传染病智能预警综合分析报告。
你的任务是基于结构化事实和模型信号进行专业分析。
硬性要求：
1. 日期、t值、模型触发状态、预警等级、提前天数等事实类内容，只能使用事实材料中已经出现的信息。
2. 不得新增病例数、概率、百分比、地区、政策名称或事实中没有的判断。
3. 不要输出 Markdown 标题、列表、表格、项目符号、解释过程或思考过程。
4. 输出连续中文段落，语气接近正式监测分析报告，克制、具体、可落地。
5. 可以依据 CNE、TRCE、Fusion 的模型含义解释风险来源、模型一致性、信号分歧和处置重点。
6. 必须区分“已观测/已识别事实”和“研判建议”，不要把已识别峰值写成预测峰值。"""


MODEL_MEANING = """模型含义：
CNE：侧重识别搜索行为和多源信息之间的因果网络结构变化，适合作为外部关注度和网络复杂性异常的辅助证据。
TRCE：侧重识别病例时间序列传播动力学的非平稳变化，适合捕捉病例演化方向和增长过程异常。
Fusion：整合多模型/多源信号并输出分级预警，适合判断多源证据是否形成一致风险指向。"""


SECTION_BADCASE_RULES = {
    "risk": [
        (r"当前.{0,12}(处于|仍处于).{0,12}(峰值前|高峰前|窗口期)", "将历史预警事件误写为当前峰值前状态"),
        (r"(仍有|尚有|还有).{0,10}(47|48|约47|约48).{0,10}(天|窗口)", "将事件提前量误写为当前剩余时间"),
    ],
    "signal": [
        (r"CNE(?![^。；，,]*未触发).{0,30}(已触发|发出预警|输出预警|检测到异常变化)", "将 CNE 未触发误写为已触发"),
        (r"外部关注度.{0,12}(明确|显著).{0,12}(异常|变化|升高)", "在 CNE 未触发时过度确认外部关注度异常"),
    ],
    "confidence": [
        (r"(三模型|CNE、TRCE、Fusion).{0,20}(一致|同步|共同触发)", "将两模型触发误写为三模型一致触发"),
        (r"(完全一致|无.{0,8}分歧|不存在.{0,8}分歧)", "忽略 CNE 未触发造成的模型分歧"),
    ],
    "advice": [
        (r"立即启动", "将事件复盘建议误写为实时响应命令"),
        (r"(仍有|尚有|还有|利用).{0,10}(提前量|47|48).{0,12}窗口", "将事件提前量误写为当前可处置窗口"),
        (r"峰值前.{0,12}(准备|处置|响应|降低传播)", "将已识别峰值误写为未来峰值前处置"),
    ],
}


SECTION_BADCASE_PROMPTS = {
    "risk": [
        "不要写：当前仍处于峰值前窗口期。",
        "不要写：尚有约47～48天响应窗口。",
        "推荐写：本轮预警相对对应峰值提前约47～48天，体现模型对风险上升过程的提前识别。",
    ],
    "signal": [
        "不要写：CNE 已触发或检测到显著异常变化。",
        "不要写：外部关注度已经出现明确异常升高。",
        "推荐写：CNE 未触发，说明搜索行为相关因果网络尚未形成与本轮事件一致的异常证据。",
    ],
    "confidence": [
        "不要写：三模型同步触发、三模型完全一致或不存在分歧。",
        "不要写：CNE、TRCE、Fusion 共同确认本轮风险。",
        "推荐写：TRCE 与 Fusion 形成相互支持，CNE 未触发提示证据来源存在分歧。",
    ],
    "advice": [
        "不要写：建议立即启动动态监测机制。",
        "不要写：利用约47～48天提前量窗口开展峰值前处置。",
        "不要写：即将到来的峰值、峰值来临前、峰值显现前。",
        "推荐写：建议在同类橙色预警场景中，将 TRCE 与 Fusion 的连续信号作为加强监测、资源准备和滚动复核的重要依据。",
    ],
}


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


def _bad_llm_reason(text, facts=None, section_key=None):
    text = _clean_llm_text(text)
    if not text:
        return "清洗后为空"
    if text.startswith("【LLM 调用失败"):
        return text
    if text.startswith("【LLM 响应正文为空"):
        return text
    if len(text) < 35:
        return f"有效正文过短（{len(text)}字）"
    if not re.search(r"[。！？.!?）】”]$", text):
        return "正文疑似截断（未以完整句末标点结束）"
    bad_patterns = ["我无法", "不能确定", "作为AI", "作为人工智能", "以下是", "标题："]
    for pattern in bad_patterns:
        if pattern in text:
            return f"包含异常模板语：{pattern}"
    if facts and "不要写成预测峰值或预计峰值" in facts:
        peak_forecast = (
            r"(预计|预测|将于|将在).{0,12}(达到|出现|到来|形成).{0,8}(峰值|高峰)|"
            r"(峰值|高峰).{0,12}(预计|预测|将于|将在).{0,12}(达到|出现|到来|形成|显现)|"
            r"(即将|来临前|到来前).{0,12}(峰值|高峰)|(峰值|高峰).{0,12}(即将|来临|到来)"
        )
        if re.search(peak_forecast, text):
            return "将已识别峰值误写为预测峰值"
        response_window = r"(立即启动|仍有.{0,8}窗口|利用.{0,8}提前量窗口|峰值前.{0,12}(准备|处置|响应|降低传播))"
        if re.search(response_window, text):
            return "将事件提前量误写为当前可处置窗口"
    if facts and "CNE首次预警：未触发" in facts:
        rules = SECTION_BADCASE_RULES.get(section_key, [])
        for pattern, reason in rules:
            if re.search(pattern, text):
                return reason
    return None


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
            f"模型提前{lead_text.replace('约', '约')}识别风险变化。综合判断，本轮流感传播系统已出现"
            f"{latest['risk_status']}信号，系统由相对稳定状态向高风险状态转变。由于本轮预警主要由"
            f"传播动力学模型与多源融合模型共同支持，说明异常并非单一监测点波动，而是病例演化过程和融合"
            f"风险信号在相近时间窗口内形成一致指向。从事件过程看，该预警位于对应峰值前的风险累积阶段，"
            f"可为同类场景下的早期识别和响应准备提供参考。"
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
        f"根据本轮智能预警分析结果，流感传播系统已出现{latest['risk_status']}信号，"
        f"预警等级为{latest['risk_color']}，预警相对对应峰值提前{latest['peak_desc']}。"
        f"建议在同类预警场景中加强疫情动态监测，持续跟踪病例变化趋势及风险指标变化；"
        f"同步完善医疗资源储备、健康提示和分级诊疗准备，并结合新增数据开展滚动复核。"
        f"在监测层面，应重点关注 ILI 曲线变化、TRCE 指标偏移以及 Fusion 预警点连续性；"
        f"在处置层面，应将多模型证据作为调整响应强度的重要参考，避免过度依赖单一模型信号。"
    )


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
        f"预警风险等级：{latest['risk_color']}",
        f"风险状态：{latest['risk_status']}",
        f"相对对应峰值提前量：{latest['peak_desc']}",
        "时间语义：本报告按监测区间内最近一个已确认预警事件出报告，提前量表示预警相对对应峰值的提前识别时间，不代表报告生成日仍处于峰值前或仍有可利用响应窗口",
    ]
    if event:
        items.extend([
            f"对应事件：{_event_name(event['event_id'])}",
            f"峰值时间节点：t={event['peak_t']}，{event['peak_date']}",
            "峰值性质：该峰值为监测区间内已识别峰值，不要写成预测峰值或预计峰值",
        ])
        for model in MODEL_ORDER:
            t = model_alerts.get(model)
            if t is None:
                items.append(f"{model}首次预警：未触发")
            else:
                items.append(f"{model}首次预警：t={t}，{_fmt_date(analysis['date_by_t'][t])}")
    return "\n".join(items)


def _conversation_system_prompt(facts):
    return f"""{SYS}

{MODEL_MEANING}

【事实材料】
{facts}

【全局写作边界】
1. 本报告会分多轮生成四个正文部分：当前风险判断、结合监测图表的模型信号解读、预警可信度分析、防控建议。
2. 后续每轮只输出指定章节正文，不要输出标题、编号、表格或解释过程。
3. 可以承接前文已经生成的判断，但不要机械重复上一轮内容。
4. 表格中的“距病例峰值/提前量”表示预警相对对应峰值的提前识别时间，不表示报告生成日距离峰值的时间。
5. 如果事实材料显示峰值已识别，只能写“峰值出现在/对应峰值为”，不要写“预计峰值、即将到来、峰值来临前”。
6. 防控建议应基于本轮预警经验提出监测和处置要点，不要写成报告生成日需要立即启动的实时响应命令。"""


def _section_badcase_prompt(section_key):
    badcases = SECTION_BADCASE_PROMPTS.get(section_key, [])
    if not badcases:
        return ""
    lines = ["【本轮 badcase 强约束】", "以下写法会被视为不合格，请主动避开；推荐表达只提供语义方向，不要求照抄。"]
    lines.extend(f"{i}. {item}" for i, item in enumerate(badcases, 1))
    return "\n".join(lines)


def _retry_prompt(reason, limit):
    if not reason:
        return ""
    return (
        "【上次输出被拒原因】\n"
        f"{reason}\n"
        f"请基于同一事实材料完整重写，优先保证事实准确和句子完整；本轮上限已缩短为 {limit} 字，不必写满。\n"
    )


def _section_task_prompt(section_key, section_name, guidance, limit, retry_reason=None):
    retry_block = _retry_prompt(retry_reason, limit)
    badcase_block = _section_badcase_prompt(section_key)
    return f"""/no_think
【本轮输出章节】{section_name}

{retry_block}
【本轮分析任务】
{guidance}

{badcase_block}

【本轮注意事项】
1. 只输出本章节正文，一个连续中文段落。
2. 可参考前文已经形成的表述保持报告一致性，但应补充本章节独有分析，不要大段复述前文。
3. 不要新增事实材料以外的日期、t值、病例数、百分比、地区、政策名称或模型触发结论。
4. 不要把 CNE 未触发写成已触发；不要把已识别峰值写成预测或未来事件。
5. 不必写满字数上限；宁可短一些，也必须自然收束并以完整句号、问号或感叹号结束。
6. 控制在 {limit} 字以内。"""


def _simple_analysis_prompt(section_key, section_name, facts, guidance, limit, retry_reason=None):
    retry_block = _retry_prompt(retry_reason, limit)
    badcase_block = _section_badcase_prompt(section_key)
    return f"""请作为 CDC 流感/传染病监测分析专家，基于事实材料撰写《传染病智能预警综合分析报告》的“{section_name}”正文。
要求：只输出一个连续中文段落；不要标题、列表、表格；不要新增事实材料以外的数字、日期、地点、模型触发结论；可以基于模型含义作专业解释；不必写满字数上限，必须以完整句末标点结束；控制在 {limit} 字以内。

{retry_block}

【事实材料】
{facts}

【分析任务】
{guidance}

{badcase_block}"""


def _preview(text, limit=90):
    text = _clean_llm_text(text).replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def _meta_text(meta):
    if not meta:
        return ""
    if meta.get("error"):
        return f"错误={meta['error']}"
    parts = [
        f"{meta.get('elapsed', 0):.1f}s",
        f"think={meta.get('think')}",
        f"response={meta.get('response_len', 0)}字",
        f"thinking={meta.get('thinking_len', 0)}字",
    ]
    if meta.get("eval_count"):
        parts.append(f"tokens={meta['eval_count']}")
    return "，".join(parts)


def _debug_meta(meta, label, cfg):
    if not cfg.get("ollama", {}).get("debug_response", False):
        return
    print(f"        {label}详情：{_meta_text(meta)}")
    if meta.get("response_preview"):
        print(f"        {label}正文预览：{meta['response_preview']}")
    if meta.get("thinking_preview"):
        print(f"        {label}thinking预览：{meta['thinking_preview']}")


def _generate_section_turn(messages, section_key, section_name, facts, guidance, fallback, limit, cfg):
    user_message = {"role": "user", "content": _section_task_prompt(section_key, section_name, guidance, limit)}
    turn_messages = messages + [user_message]
    raw = call_ollama_chat(turn_messages, cfg)
    first_meta = dict(OLLAMA_LAST_META)
    reason = _bad_llm_reason(raw, facts, section_key)
    if reason:
        retry_limit = max(120, int(limit * 0.75))
        retry_message = {
            "role": "user",
            "content": _section_task_prompt(section_key, section_name, guidance, retry_limit, reason),
        }
        retry_messages = messages + [retry_message]
        retry = call_ollama_chat(retry_messages, cfg, think=False)
        retry_meta = dict(OLLAMA_LAST_META)
        retry_reason = _bad_llm_reason(retry, facts, section_key)
        if retry_reason:
            simple_limit = max(100, int(limit * 0.6))
            simple_message = {
                "role": "user",
                "content": _simple_analysis_prompt(
                    section_key, section_name, facts, guidance, simple_limit, retry_reason
                ),
            }
            simple_messages = messages + [simple_message]
            simple_retry = call_ollama_chat(simple_messages, cfg, think=False)
            simple_meta = dict(OLLAMA_LAST_META)
            simple_reason = _bad_llm_reason(simple_retry, facts, section_key)
            if simple_reason:
                print(f"      - {section_name}：LLM 输出不可用，使用规则兜底（{simple_reason}；{_meta_text(simple_meta)}）")
                _debug_meta(first_meta, "首次", cfg)
                _debug_meta(retry_meta, "缩短重试", cfg)
                _debug_meta(simple_meta, "简化重试", cfg)
                return fallback, messages + [simple_message, {"role": "assistant", "content": fallback}]
            print(f"      - {section_name}：已使用 LLM 分析（简化重试；{_meta_text(simple_meta)}）")
            _debug_meta(first_meta, "首次", cfg)
            _debug_meta(retry_meta, "缩短重试", cfg)
            _debug_meta(simple_meta, "简化重试", cfg)
            text = _clean_llm_text(simple_retry)
            return text, messages + [simple_message, {"role": "assistant", "content": text}]
        print(f"      - {section_name}：已使用 LLM 分析（关闭 thinking 缩短重试；{_meta_text(retry_meta)}）")
        _debug_meta(first_meta, "首次", cfg)
        _debug_meta(retry_meta, "缩短重试", cfg)
        text = _clean_llm_text(retry)
        return text, messages + [retry_message, {"role": "assistant", "content": text}]
    print(f"      - {section_name}：已使用 LLM 分析（{_meta_text(first_meta)}）")
    _debug_meta(first_meta, "首次", cfg)
    text = _clean_llm_text(raw)
    return text, messages + [user_message, {"role": "assistant", "content": text}]


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

    messages = [{"role": "system", "content": _conversation_system_prompt(facts)}]
    narratives = {}
    sections = [
        (
            "risk",
            "当前风险判断", facts,
            "围绕最近一次预警、触发模型、风险等级、对应事件、峰值节点和提前天数作综合研判。需要说明当前风险是由哪些模型证据支撑、风险状态代表什么、为什么需要关注峰值前窗口。若事实显示峰值已识别，不要写成预计峰值。",
            drafts["risk"], 420,
        ),
        (
            "signal",
            "结合监测图表的模型信号解读", facts,
            "结合三张监测图表分别解释 CNE、TRCE、Fusion 的信号含义。必须尊重事实材料中的触发/未触发状态，不要把未触发模型写成已触发。需要说明各模型从信息网络、传播动力学、多源融合角度分别提供了什么证据，以及这些证据之间是否一致。",
            drafts["signal"], 720,
        ),
        (
            "confidence",
            "预警可信度分析", facts,
            "从多模型一致性、模型分工、未触发模型的含义、历史事件提前量参考等角度判断本次预警可信度。既要指出支持证据，也要说明不确定性和需要继续复核的部分。",
            drafts["confidence"], 360,
        ),
        (
            "advice",
            "防控建议", facts,
            "基于本轮橙色预警、触发模型和预警提前量提出监测处置建议。建议应表述为对同类预警场景和后续滚动监测的处置要点，覆盖动态监测、医疗资源准备、风险沟通、模型复核等方向；不得写成报告生成日仍有47～48天窗口或需要立即启动的实时命令；不得新增具体政策名称、地区或事实材料中没有的数据。",
            drafts["advice"], 520,
        ),
    ]
    for key, section_name, section_facts, guidance, fallback, limit in sections:
        narratives[key], messages = _generate_section_turn(
            messages, key, section_name, section_facts, guidance, fallback, limit, cfg
        )
    return narratives


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
    L.append(f"| 预警风险等级 | {latest['risk_color']} |")
    L.append(f"| 相对峰值提前量 | {latest['peak_desc']} |")
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
        return gen_narratives(analysis, cfg, dry_run=dry_run)
    with contextlib.redirect_stdout(io.StringIO()):
        return gen_narratives(analysis, cfg, dry_run=dry_run)


def _abs_path(path, base_dir):
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(base_dir, path))


def _data_file_path(cfg, key):
    base_dir = cfg["data"]["base_dir"]
    fname = cfg["data"][key]
    return fname if os.path.isabs(fname) else os.path.join(base_dir, fname)


def _input_paths(cfg):
    return {
        "data_dir": cfg["data"]["base_dir"],
        "main_data": _data_file_path(cfg, "main_data"),
        "cne_alerts": _data_file_path(cfg, "cne_alerts"),
        "trce_alerts": _data_file_path(cfg, "trce_alerts"),
        "fusion_alerts": _data_file_path(cfg, "fusion_alerts"),
    }


def _apply_path_overrides(
    cfg,
    config_path,
    data_dir=None,
    main_data=None,
    cne_alerts=None,
    trce_alerts=None,
    fusion_alerts=None,
    output_dir=None,
):
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if data_dir is not None:
        cfg["data"]["base_dir"] = _abs_path(data_dir, config_dir)
    for key, value in {
        "main_data": main_data,
        "cne_alerts": cne_alerts,
        "trce_alerts": trce_alerts,
        "fusion_alerts": fusion_alerts,
    }.items():
        if value is not None:
            cfg["data"][key] = value
    if output_dir is not None:
        cfg["output"]["dir"] = _abs_path(output_dir, config_dir)
    return cfg


def generate_report(
    config_path=None,
    dry_run=False,
    md2pdf=True,
    now=None,
    verbose=False,
    data_dir=None,
    main_data=None,
    cne_alerts=None,
    trce_alerts=None,
    fusion_alerts=None,
    output_dir=None,
    markdown_path=None,
    pdf_path=None,
    make_pdf=None,
):
    """Generate a report and return output paths plus summary metadata.

    Args:
        config_path: Path to config YAML.
        dry_run: Use rule-based narratives without calling Ollama.
        md2pdf: Whether to convert the Markdown report to PDF.
        now: Optional datetime or ISO datetime string used in timestamps.
        verbose: Print progress and LLM diagnostics while running.
        data_dir: Override data.base_dir from config.
        main_data: Override main Excel input filename/path.
        cne_alerts: Override CNE alert CSV filename/path.
        trce_alerts: Override TRCE alert CSV filename/path.
        fusion_alerts: Override Fusion alert CSV filename/path.
        output_dir: Override output directory.
        markdown_path: Override generated Markdown path.
        pdf_path: Override generated PDF path.
        make_pdf: Backward-compatible alias for md2pdf.
    """
    config_path = config_path or DEFAULT_CONFIG
    if make_pdf is not None:
        md2pdf = make_pdf
    setup_font()
    cfg = load_config(config_path)
    cfg = _apply_path_overrides(
        cfg,
        config_path,
        data_dir=data_dir,
        main_data=main_data,
        cne_alerts=cne_alerts,
        trce_alerts=trce_alerts,
        fusion_alerts=fusion_alerts,
        output_dir=output_dir,
    )
    run_time = _now_value(now)
    cfg["_now"] = run_time.strftime("%Y-%m-%d %H:%M:%S")

    if verbose:
        print("[1/6] 加载数据 ...")
    data = load_data(cfg)

    if verbose:
        print("[2/6] 分析疫情事件 ...")
    analysis = analyze_flu_events(data, cfg)

    out_dir = cfg["output"]["dir"]
    chart_dir = os.path.join(out_dir, "charts")
    if verbose:
        print("[3/6] 生成图表 ...")
    chart_paths = generate_charts(data, chart_dir)

    if verbose:
        mode = "dry-run 规则底稿" if dry_run else "调用 Ollama"
        print(f"[4/6] 生成叙述文本（{mode}） ...")
    narratives = _call_narratives(analysis, cfg, dry_run=dry_run, verbose=verbose)

    if verbose:
        print("[5/6] 组装 Markdown ...")
    markdown = build_report(analysis, narratives, chart_paths, cfg, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    md_path = _abs_path(
        markdown_path or f"{REPORT_FILE_STEM}_{run_time.strftime('%Y%m%d_%H%M')}.md",
        out_dir,
    )
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    actual_pdf_path = None
    pdf_error = None
    if md2pdf:
        if verbose:
            print("[6/6] 生成 PDF ...")
        try:
            from core.md_to_pdf import export_md_pdf

            requested_pdf_path = _abs_path(pdf_path, out_dir) if pdf_path else None
            if requested_pdf_path:
                os.makedirs(os.path.dirname(requested_pdf_path), exist_ok=True)
            if verbose:
                actual_pdf_path = export_md_pdf(md_path, requested_pdf_path)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    actual_pdf_path = export_md_pdf(md_path, requested_pdf_path)
        except Exception as exc:
            pdf_error = str(exc)
            if verbose:
                print(f"  PDF 生成失败：{pdf_error}")

    # 额外输出:把报告 PDF 平铺复制一份到 config.output.extra_pdf_dir(若已配置)。
    extra_pdf_path = None
    extra_pdf_dir = (cfg.get("output", {}).get("extra_pdf_dir") or "").strip()
    if actual_pdf_path and extra_pdf_dir:
        try:
            os.makedirs(extra_pdf_dir, exist_ok=True)
            dst = os.path.join(extra_pdf_dir, os.path.basename(actual_pdf_path))
            if os.path.abspath(dst) != os.path.abspath(actual_pdf_path):
                shutil.copy2(actual_pdf_path, dst)
                extra_pdf_path = dst
                if verbose:
                    print(f"  PDF 已额外复制到：{dst}")
        except Exception as exc:
            if verbose:
                print(f"  额外 PDF 复制失败：{exc}")

    return {
        "ok": True,
        "dry_run": dry_run,
        "md2pdf": md2pdf,
        "generated_at": cfg["_now"],
        "config_path": os.path.abspath(config_path),
        "input_paths": _input_paths(cfg),
        "output_dir": out_dir,
        "markdown_path": md_path,
        "pdf_path": actual_pdf_path,
        "extra_pdf_path": extra_pdf_path,
        "pdf_error": pdf_error,
        "chart_paths": chart_paths,
        "data_period": analysis["data_period"],
        "event_count": len(analysis["events"]),
        "latest": _latest_summary(analysis),
    }


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--dry-run", action="store_true", help="不调用 LLM，用占位文本")
    ap.add_argument("--data-dir", help="覆盖数据目录")
    ap.add_argument("--main-data", help="覆盖主数据 Excel 文件名或路径")
    ap.add_argument("--cne-alerts", help="覆盖 CNE 预警 CSV 文件名或路径")
    ap.add_argument("--trce-alerts", help="覆盖 TRCE 预警 CSV 文件名或路径")
    ap.add_argument("--fusion-alerts", help="覆盖 Fusion 预警 CSV 文件名或路径")
    ap.add_argument("--output-dir", help="覆盖输出目录")
    ap.add_argument("--markdown-path", help="覆盖 Markdown 输出路径")
    ap.add_argument("--pdf-path", help="覆盖 PDF 输出路径")
    pdf_group = ap.add_mutually_exclusive_group()
    pdf_group.add_argument("--md2pdf", dest="md2pdf", action="store_true", default=True, help="生成 PDF")
    pdf_group.add_argument("--no-md2pdf", "--no-pdf", dest="md2pdf", action="store_false", help="只生成 Markdown，不生成 PDF")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结构化结果")
    ap.add_argument("--quiet", action="store_true", help="不打印过程日志")
    args = ap.parse_args()

    result = generate_report(
        config_path=args.config,
        dry_run=args.dry_run,
        md2pdf=args.md2pdf,
        data_dir=args.data_dir,
        main_data=args.main_data,
        cne_alerts=args.cne_alerts,
        trce_alerts=args.trce_alerts,
        fusion_alerts=args.fusion_alerts,
        output_dir=args.output_dir,
        markdown_path=args.markdown_path,
        pdf_path=args.pdf_path,
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
        if result.get("extra_pdf_path"):
            print(f"额外 PDF → {result['extra_pdf_path']}")


if __name__ == "__main__":
    main()
