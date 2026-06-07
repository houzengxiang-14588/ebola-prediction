"""Ebola Virus Trend Forecast - Dual-Cycle (Short 7d + Long 21d) + LoRA + RAG + Backtest

Based on Qwen2-7B-Instruct + LoRA fine-tuning + ChromaDB vector knowledge base.
Supports short-cycle (7-day) / long-cycle (21-day) dual-mode prediction.
"""

import sys, json, re, os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_font_paths = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]
_zh_font = None
for fp in _font_paths:
    if os.path.exists(fp):
        _zh_font = fm.FontProperties(fname=fp)
        break
if _zh_font:
    plt.rcParams["font.family"] = _zh_font.get_name()
plt.rcParams["axes.unicode_minus"] = False

from datetime import datetime, timedelta

sys.path.insert(0, ".")
sys.path.insert(0, "src")


# ============================================================
# 1. Ebola Knowledge Base
# ============================================================

def load_ebola_knowledge():
    df = pd.read_csv("data/ebola_africa.csv")
    years = [2019, 2020, 2021, 2022, 2023]
    yearly_totals = {}
    for y in years:
        yearly_totals[y] = int(df[f"cases_{y}"].sum())

    country_profiles = []
    for _, row in df.iterrows():
        avg_cases = np.mean([row[f"cases_{y}"] for y in years])
        if avg_cases > 0:
            country_profiles.append({
                "name": row["country"],
                "avg_cases": avg_cases,
                "level": "Historical Hotspot" if avg_cases > 500 else (
                    "Recent Affected Zone" if avg_cases > 10 else "Sporadic/Surveillance"),
                "trend": "Falling" if row["cases_2023"] < row["cases_2022"] * 0.8 else (
                    "Rising" if row["cases_2023"] > row["cases_2022"] * 1.2 else "Stable"),
                "cases_2023": int(row["cases_2023"]),
            })

    return {
        "country_profiles": country_profiles,
        "yearly_totals": yearly_totals,
        "all_cases_2023": int(df["cases_2023"].sum()),
        "total_5y": sum(yearly_totals.values()),
    }


def load_user_cases():
    df = pd.read_excel("病例时间统计.xlsx")
    df.columns = ["date", "cases"]
    df["date"] = pd.to_datetime(df["date"])
    return df


# ============================================================
# 2. RAG Knowledge Retrieval
# ============================================================

_rag_retriever = None


def get_rag_retriever():
    global _rag_retriever
    if _rag_retriever is not None:
        return _rag_retriever
    try:
        from src.rag.vector_store import VectorStore
        from src.rag.retriever import KnowledgeRetriever
        vs = VectorStore(persist_dir="chroma_db")
        _rag_retriever = KnowledgeRetriever(vs, top_k=5)
        return _rag_retriever
    except Exception as e:
        print(f"RAG init failed: {e}")
        return None


def retrieve_knowledge(user_cases, metadata=None):
    retriever = get_rag_retriever()
    if retriever is None:
        return None
    try:
        return retriever.retrieve_for_prediction(user_cases, metadata)
    except Exception:
        return None


# ============================================================
# 3. Dual-Cycle Prompt
# ============================================================

def build_dual_cycle_prompt(user_df, knowledge, rag_results=None):
    cases = user_df["cases"].values.astype(int)
    dates = user_df["date"].values
    total_cases = int(cases.sum())
    max_case = int(cases.max())

    half = len(cases) // 2
    recent_avg = cases[-3:].mean() if len(cases) >= 3 else cases.mean()
    early_avg = cases[:half].mean() if half > 0 else 0
    late_avg = cases[half:].mean() if half > 0 else 0
    if late_avg > early_avg * 1.3:
        recent_trend = "Rising"
    elif late_avg < early_avg * 0.7:
        recent_trend = "Falling"
    else:
        recent_trend = "Stable / fluctuating"

    history_text = "\n".join(
        f"{pd.Timestamp(d).strftime('%m-%d')}: {cv} cases"
        for d, cv in zip(dates, cases)
    )

    top_countries = sorted(
        knowledge["country_profiles"],
        key=lambda x: x["avg_cases"], reverse=True
    )[:5]
    country_context = "; ".join(
        f"{c['name']}({c['level']}, 2023: {c['cases_2023']} cases)"
        for c in top_countries
    )

    rag_text = ""
    if rag_results and rag_results.get("knowledge"):
        rag_text = "\n--- Vector Knowledge Base References ---\n"
        for i, r in enumerate(rag_results["knowledge"][:3], 1):
            source = r["metadata"].get("source", r["metadata"].get("category", ""))
            rag_text += f"\n[Ref {i}] {source}: {r['content'][:250]}\n"

    prompt = f"""You are an Ebola virus epidemiological analysis expert. Predict daily new cases for the next 21 days based on surveillance data.

--- Background ---
Ebola virus disease (EVD) is transmitted through direct contact with bodily fluids of infected individuals.
Outbreaks occur primarily in West and Central Africa. 2023 total: {knowledge['all_cases_2023']} cases.
5-year cumulative across monitored countries: {knowledge['total_5y']} cases.
Key affected regions: {country_context}
{rag_text}
--- Surveillance Data ({len(cases)} days) ---
Total: {total_cases} cases | Daily mean: {cases.mean():.1f} | Peak: {max_case} | Recent trend: {recent_trend}

{history_text}

--- Prediction ---
Predict daily new cases (integers) for the next 21 days. Consider recent 7-day trend continuity, outbreak transmission dynamics, and background sporadic case patterns.

Output as JSON array of 21 values:
[{{"day": 1, "cases": N}}, {{"day": 2, "cases": N}}, ... 21 total ...]

JSON:"""

    return prompt


# ============================================================
# 4. Chart Generation
# ============================================================

def make_dual_chart(user_df, short_pred, long_weekly, granularity="Daily"):
    dates = list(user_df["date"].dt.strftime("%m-%d"))
    values = list(user_df["cases"].values.astype(int))
    last_date = user_df["date"].iloc[-1]
    n_hist = len(dates)

    s_vals = [p.get("cases", 0) for p in short_pred[:7]]
    pred_dates = [(last_date + timedelta(days=i + 1)).strftime("%m-%d") for i in range(21)]

    hist_weeks = []
    for w in range(max(1, len(values) // 7)):
        wk_vals = values[w * 7:(w + 1) * 7]
        hist_weeks.append(sum(wk_vals))
    hist_weeks = hist_weeks[-4:] if len(hist_weeks) >= 4 else ([0] * (4 - len(hist_weeks)) + hist_weeks)

    lw = [w.get("cases", 0) for w in long_weekly[:3]] if long_weekly else [0, 0, 0]

    if granularity == "Weekly":
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        week_labels = [f"W-{w}" for w in range(len(hist_weeks), 0, -1)] + \
                      ["Pred W1\n(7d)", "Pred W2\n(8-14d)", "Pred W3\n(15-21d)"]
        week_values = hist_weeks + lw
        colors = (["#93c5fd"] * len(hist_weeks) + ["#f59e0b"] + ["#fca5a5"] * 2)
        edge_colors = (["#2563eb"] * len(hist_weeks) + ["#d97706"] + ["#dc2626"] * 2)

        ax.bar(range(len(week_labels)), week_values, color=colors,
               edgecolor=edge_colors, linewidth=1.5, alpha=0.85)
        ax.set_xticks(range(len(week_labels)))
        ax.set_xticklabels(week_labels, fontsize=10)
        ax.set_ylabel("Weekly New Cases", fontsize=12)
        ax.set_title("Ebola Weekly Forecast (Blue=History / Orange=Short / Red=Long)",
                     fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        sep_x = len(hist_weeks) - 0.5
        ax.axvline(x=sep_x, color="#94a3b8", linestyle="-", linewidth=2, alpha=0.5)
        ax.text(sep_x, ax.get_ylim()[1] * 0.92, "History  |  Forecast",
                ha="center", fontsize=9, color="#64748b")
        ax.axvspan(len(hist_weeks) - 0.4, len(hist_weeks) + 0.4,
                   facecolor="#fbbf24", alpha=0.15, edgecolor="none")
        ax.text(len(hist_weeks), ax.get_ylim()[1] * 0.85, "7d Node",
                ha="center", fontsize=8, color="#d97706", fontweight="bold")

        for i, v in enumerate(week_values):
            ax.text(i, v + max(1, max(week_values) * 0.03), str(v),
                    ha="center", fontsize=10, fontweight="bold")
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                        gridspec_kw={"height_ratios": [3, 1]})
        n_pred = len(s_vals)
        ax1.bar(range(n_hist), values, color="#2563eb", alpha=0.85, label="Historical")
        ax1.bar(range(n_hist, n_hist + n_pred), s_vals,
                color="#f59e0b", alpha=0.9, label="Short-term Forecast (Next 7d)")

        ax1.axhline(y=0, color="#94a3b8", linewidth=0.5)
        ax1.axvline(x=n_hist - 0.5, color="#94a3b8", linestyle="-", linewidth=1.8, alpha=0.6)
        y_top = max(max(values + s_vals), 1) * 1.15
        ax1.set_ylim(0, y_top)
        ax1.set_xlim(-0.8, n_hist + n_pred - 1 + 0.5)
        ax1.margins(x=0)

        all_labels = dates + pred_dates[:n_pred]
        tick_step = max(1, len(all_labels) // 10)
        tick_pos = list(range(0, len(all_labels), tick_step))
        ax1.set_xticks(tick_pos)
        ax1.set_xticklabels([all_labels[i] for i in tick_pos], rotation=45, fontsize=8)
        ax1.set_ylabel("Daily New Cases", fontsize=11)
        ax1.set_title("Ebola Short-term Forecast (Next 7d) - See Report for Long-term",
                      fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax1.grid(axis="y", alpha=0.25)

        week_labels = [f"W-{w}" for w in range(len(hist_weeks), 0, -1)] + \
                      ["Short\n(W1)", "Long W2\n(8-14d)", "Long W3\n(15-21d)"]
        week_values = hist_weeks + lw
        week_colors = (["#93c5fd"] * len(hist_weeks) + ["#f59e0b"] + ["#fca5a5"] * 2)

        ax2.bar(range(len(week_labels)), week_values, color=week_colors, alpha=0.85, edgecolor="white")
        ax2.set_xticks(range(len(week_labels)))
        ax2.set_xticklabels(week_labels, fontsize=9)
        ax2.set_ylabel("Weekly Total Cases", fontsize=11)
        ax2.set_title("Weekly Comparison - Recent History + Forecast Weeks", fontsize=12)
        ax2.grid(axis="y", alpha=0.3)
        ax2.axvline(x=len(hist_weeks) - 0.5, color="#94a3b8", linestyle="-", linewidth=1.5)

        for i, v in enumerate(week_values):
            if v > 0:
                ax2.text(i, v + max(1, max(week_values) * 0.03), str(v),
                         ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = "data/chart_advanced.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# 5. Core Prediction Logic
# ============================================================

_model_client = None
_model_mode_cache = None
_knowledge = None


def get_model_client(model_mode="lora"):
    global _model_client, _model_mode_cache
    if _model_client is not None and _model_mode_cache != model_mode:
        del _model_client
        _model_client = None
        import torch
        torch.cuda.empty_cache()

    if _model_client is not None:
        return _model_client

    from src.model.lora_model import LoraModelClient

    lora_path = "lora_weights/final" if model_mode == "lora" else None
    print(f"Loading model (mode={model_mode})...")

    _model_client = LoraModelClient(
        model_path="D:/llm_models/qwen/Qwen2-7B-Instruct",
        lora_path=lora_path,
        temperature=0.7 if model_mode == "lora" else 0.3,
        max_tokens=512,
        load_in_4bit=True,
    )
    _model_mode_cache = model_mode
    return _model_client


def get_knowledge():
    global _knowledge
    if _knowledge is None:
        _knowledge = load_ebola_knowledge()
    return _knowledge


def parse_dual_json(text: str):
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if not arr_match:
        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match:
            try:
                obj = json.loads(obj_match.group())
                daily = obj.get("predictions", obj.get("daily", []))
                if not (isinstance(daily, list) and len(daily) > 0):
                    return None
            except json.JSONDecodeError:
                return None
        else:
            return None

    try:
        daily_raw = json.loads(arr_match.group())
    except json.JSONDecodeError:
        return None

    if not isinstance(daily_raw, list) or len(daily_raw) < 7:
        return None

    all_daily = []
    for i, item in enumerate(daily_raw[:21]):
        if isinstance(item, dict):
            all_daily.append({"day": i + 1, "cases": int(item.get("cases", 0))})
        elif isinstance(item, (int, float)):
            all_daily.append({"day": i + 1, "cases": int(item)})
        else:
            all_daily.append({"day": i + 1, "cases": 0})

    short = all_daily[:7]
    full_21 = all_daily[:21]

    w1 = sum(d["cases"] for d in full_21[0:7])
    w2 = sum(d["cases"] for d in full_21[7:14])
    w3 = sum(d["cases"] for d in full_21[14:21])

    long_weekly = [
        {"week": 1, "cases": w1},
        {"week": 2, "cases": w2},
        {"week": 3, "cases": w3},
    ]

    first_7 = sum(d["cases"] for d in full_21[0:7])
    last_7 = sum(d["cases"] for d in full_21[14:21])
    if last_7 > first_7 * 1.3:
        trend = "Rising"
    elif last_7 < first_7 * 0.7:
        trend = "Falling"
    else:
        trend = "Stable"

    return {
        "short": short,
        "long_daily": full_21[7:21],
        "long_weekly": long_weekly,
        "full_21": full_21,
        "trend": trend,
    }


def predict_dual(user_input_str, model_mode="lora", use_rag=True, granularity="Daily"):
    user_df = load_user_cases()

    if user_input_str and user_input_str.strip():
        try:
            extra = [int(x.strip()) for x in user_input_str.replace(",", " ").split() if x.strip()]
            last_date = user_df["date"].iloc[-1]
            for i, v in enumerate(extra):
                user_df = pd.concat([user_df, pd.DataFrame({
                    "date": [last_date + timedelta(days=i + 1)], "cases": [v],
                })], ignore_index=True)
        except ValueError:
            pass

    knowledge = get_knowledge()
    cases = user_df["cases"].values.astype(int)

    rag_results = None
    if use_rag:
        metadata = {
            "total_cases": int(cases.sum()),
            "max_case": int(cases.max()),
        }
        rag_results = retrieve_knowledge(cases, metadata)

    prompt = build_dual_cycle_prompt(user_df, knowledge, rag_results)
    client = get_model_client(model_mode)
    raw = client.generate(prompt)
    result = parse_dual_json(raw)

    if not result:
        return None, f"Parse failed\nRaw output:\n{raw[:1000]}", None, None, prompt, rag_results

    short_pred = result["short"]
    long_weekly = result["long_weekly"]
    full_21 = result["full_21"]
    model_trend = result["trend"]
    long_daily_for_chart = full_21[7:21]

    chart_path = make_dual_chart(user_df, short_pred, long_daily_for_chart, granularity)

    s_vals = [p.get("cases", 0) for p in short_pred]
    s_total = sum(s_vals)
    s_daily = s_total / 7

    l_weekly_vals = [w.get("cases", 0) for w in long_weekly] if long_weekly else [0, 0, 0]
    l_total = sum(l_weekly_vals)
    l_weekly_avg = l_total / 3 if l_total > 0 else 0

    if len(s_vals) >= 4:
        if np.mean(s_vals[3:]) > np.mean(s_vals[:3]) * 1.3:
            short_trend = "Rising"
        elif np.mean(s_vals[3:]) < np.mean(s_vals[:3]) * 0.7:
            short_trend = "Falling"
        else:
            short_trend = "Stable"
    else:
        short_trend = "-"

    model_tag = f"Qwen2-7B {'+LoRA' if model_mode == 'lora' else '(Base)'}"
    rag_tag = "+RAG" if use_rag else ""

    if granularity == "Weekly":
        report = (
            f"## Ebola Weekly Forecast Report\n\n"
            f"**{model_tag} {rag_tag}** | Trend: {model_trend}\n\n"
            f"| Cycle | Days | New Cases | Daily Avg | Trend |\n"
            f"|------|:--:|:--:|:--:|:--:|\n"
            f"| Short-term | Day 1-7 | **{s_total} cases** | {s_daily:.1f} | {short_trend} |\n"
            f"| Long-term W1 | Day 1-7 | **{l_weekly_vals[0]} cases** | {l_weekly_vals[0] / 7:.1f} | - |\n"
            f"| Long-term W2 | Day 8-14 | **{l_weekly_vals[1]} cases** | {l_weekly_vals[1] / 7:.1f} | - |\n"
            f"| Long-term W3 | Day 15-21 | **{l_weekly_vals[2]} cases** | {l_weekly_vals[2] / 7:.1f} | - |\n"
            f"| **21-Day Total** | | **{l_total} cases** | {l_total / 21:.1f} | {model_trend} |\n\n"
            f"### Reference Nodes\n\n"
            f"| 7-Day Node | 14-Day Node | 21-Day Node |\n"
            f"|:--:|:--:|:--:|\n"
            f"| **{s_total} cases** | **{s_total + l_weekly_vals[0] if len(l_weekly_vals) > 0 else s_total} cases** | **{l_total} cases** |\n"
        )
    else:
        report = (
            f"## Ebola Dual-Cycle Forecast Report\n\n"
            f"**{model_tag} {rag_tag}** | Trend: {model_trend}\n\n"
            f"---\n### Short-term (Next 7 Days)\n\n"
            f"| Metric | Value |\n|------|----|\n"
            f"| 7-Day Total | **{s_total} cases** |\n"
            f"| Daily Average | {s_daily:.1f} cases |\n"
            f"| Trend | {short_trend} |\n\n"
        )
        for i, p in enumerate(short_pred[:7]):
            report += f"- Day {i + 1}: {p.get('cases', '?')} cases\n"

        report += (
            f"\n---\n### Long-term (Next 21 Days / 3 Weeks)\n\n"
            f"| Metric | Value |\n|------|----|\n"
            f"| 21-Day Total | **{l_total} cases** |\n"
            f"| Weekly Average | {l_weekly_avg:.1f} cases/week |\n"
            f"| Daily Average | {l_total / 21:.1f} cases/day |\n\n"
        )
        if long_weekly:
            for w in long_weekly:
                report += f"- Week {w.get('week', '?')}: {w.get('cases', 0)} cases\n"

        report += (
            f"\n---\n### Reference Nodes\n\n"
            f"| 7-Day | 14-Day | 21-Day |\n"
            f"|:--:|:--:|:--:|\n"
            f"| **{s_total} cases** | **{s_total + l_weekly_vals[0] if len(l_weekly_vals) > 0 else s_total} cases** | **{l_total} cases** |\n"
        )

    return chart_path, report, short_pred, long_weekly, prompt, rag_results


# ============================================================
# 6. Gradio Interface
# ============================================================

import gradio as gr


def gradio_handler(user_input, model_mode, use_rag, granularity):
    chart, report, short, long_wk, prompt, rag = predict_dual(
        user_input, model_mode=model_mode, use_rag=use_rag, granularity=granularity,
    )
    if chart is None:
        return None, report, prompt or "", ""

    rag_display = ""
    if rag and rag.get("knowledge"):
        rag_display = "### Vector Knowledge Base Retrieval Results\n\n"
        for i, r in enumerate(rag["knowledge"][:5], 1):
            source = r["metadata"].get("source", r["metadata"].get("category", "Unknown"))
            rag_display += (
                f"**[{i}] {source}** (Relevance: {r['score']:.2f})\n"
                f"> {r['content'][:250]}...\n\n"
            )
    return chart, report, prompt or "", rag_display


def run_backtest_ui(model_mode):
    import json
    try:
        json_path = f"data/backtest_{model_mode}.json"
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            from src.eval.backtest import generate_backtest_report
            fig_path, _ = generate_backtest_report(result, f"data/backtest_{model_mode}.png")
            m = result["metrics"]
            acc = m.get('accuracy', m.get('direction_accuracy', 0))
            report = (
                f"## Backtest Report (Model: {model_mode})\n\n"
                f"| Metric | Value |\n|------|----|\n"
            )
            if 'accuracy' in m:
                report += f"| Accuracy | {acc:.1%} |\n"
                report += f"| Random Baseline | 33.3% |\n"
                for cls in ['Rise', 'Fall', 'Stable']:
                    cm = m.get('class_metrics', {}).get(cls, {})
                    if cm:
                        report += f"| {cls}-F1 | {cm.get('f1', 0):.1%} |\n"
            report += (
                f"| Samples | {m.get('total_countries', m.get('total_samples', '?'))} |\n\n"
                f"> Sliding-window evaluation on WHO weekly outbreak data"
            )
            return fig_path, report

        from src.eval.backtest import run_backtest, generate_backtest_report
        client = get_model_client(model_mode)
        result = run_backtest(client)
        fig_path, txt_path = generate_backtest_report(result, f"data/backtest_{model_mode}.png")
        m = result["metrics"]
        report = (
            f"## Backtest Report (Live)\n\n"
            f"| Metric | Value |\n|------|----|\n"
            f"| Accuracy | {m.get('accuracy', 0):.1%} |\n"
        )
        return fig_path, report
    except Exception as e:
        return None, f"Backtest failed: {str(e)}"


CSS = """
.gradio-container { max-width: 1200px !important; }
.prompt-box textarea { font-family: 'Consolas', monospace; font-size: 12px; }
.knowledge-box { font-size: 13px; }
"""

with gr.Blocks(title="Ebola Trend Forecast (Dual-Cycle LoRA + RAG)", css=CSS) as demo:
    gr.Markdown("# Ebola Dual-Cycle Trend Forecast System")
    gr.Markdown(
        "**Qwen2-7B-Instruct + LoRA + ChromaDB RAG** | "
        "Short-term 7d | Long-term 21d / 3 Weeks | "
        "7d / 14d Reference Nodes | Daily & Weekly Granularity"
    )

    with gr.Tabs():
        with gr.TabItem("Dual-Cycle Forecast"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Control Panel")
                    user_input = gr.Textbox(
                        label="Add Latest Cases (optional, comma-separated)",
                        placeholder="Leave empty to use Excel data\nor enter: 0, 2, 0, 1, 0",
                        lines=2,
                    )
                    model_mode = gr.Radio(
                        choices=["lora", "base"],
                        value="lora",
                        label="Model Mode",
                    )
                    granularity = gr.Radio(
                        choices=["Daily", "Weekly"],
                        value="Daily",
                        label="Forecast Granularity",
                    )
                    use_rag = gr.Checkbox(
                        value=True,
                        label="Enable Vector Knowledge Base (RAG)",
                    )
                    gr.Markdown("""
                    **Forecast Notes**:
                    - Short-term: Next 7 days
                    - Long-term: Next 21 days
                    - Single inference produces both cycles
                    """)
                    submit_btn = gr.Button("Start Forecast", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("### Dual-Cycle Forecast Results")
                    chart_output = gr.Image(label="Trend Chart", show_label=False)
                    report_output = gr.Markdown(label="Forecast Report")

            with gr.Accordion("View Full Prompt", open=False):
                prompt_output = gr.Textbox(
                    label="Prompt Sent to LLM", lines=15, max_lines=30,
                    elem_classes=["prompt-box"],
                )
            with gr.Accordion("Vector Knowledge Base Retrieval Results", open=False):
                rag_output = gr.Markdown(elem_classes=["knowledge-box"])

            submit_btn.click(
                fn=gradio_handler,
                inputs=[user_input, model_mode, use_rag, granularity],
                outputs=[chart_output, report_output, prompt_output, rag_output],
            )

        with gr.TabItem("Backtest"):
            gr.Markdown("### Historical Data Backtest Validation")
            gr.Markdown("Sliding-window evaluation on WHO weekly outbreak data.")
            with gr.Row():
                bt_model_mode = gr.Radio(choices=["lora", "base"], value="lora", label="Model Mode")
                run_bt_btn = gr.Button("View Backtest Results", variant="primary")
            with gr.Row():
                bt_chart = gr.Image(label="Confusion Matrix + Class Metrics")
                bt_report = gr.Markdown(label="Metrics Report")
            run_bt_btn.click(
                fn=run_backtest_ui, inputs=[bt_model_mode],
                outputs=[bt_chart, bt_report],
            )

        with gr.TabItem("Knowledge Base"):
            gr.Markdown("### Vector Knowledge Base Overview")
            gr.Markdown(
                "Knowledge Sources:\n"
                "- **WHO Outbreak Database**: 17,585 records x 12 countries x 36 indicators (2014-2016)\n"
                "- **cmrivers/ebola**: 5 daily-resolution CSVs, West Africa 2014-2015 outbreak\n"
                "- **13 Expert-curated Knowledge Chunks**: Transmission dynamics, outbreak patterns, virology\n"
                "- **20 Country Profiles**: Historical outbreak data by African nation\n\n"
                "Embedding: `all-MiniLM-L6-v2` | Vector DB: ChromaDB (local persistent)\n"
                "Retrieval: Top-5 semantic matching based on case features (magnitude/trend)"
            )


if __name__ == "__main__":
    import os as _os
    _os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    demo.launch(server_name="0.0.0.0", server_port=8080, share=False, quiet=True, show_error=True)
