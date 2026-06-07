"""埃博拉病毒趋势预测 — Gradio 交互演示
基于 WHO 疫情数据 + 本地大模型，专用埃博拉预测方案。
"""

import sys, json, re, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_zh_font = None
for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(fp):
        _zh_font = fm.FontProperties(fname=fp)
        break
if _zh_font:
    plt.rcParams["font.family"] = _zh_font.get_name()
plt.rcParams["axes.unicode_minus"] = False

from datetime import datetime, timedelta

sys.path.insert(0, "src")
from model.llm_client import LLMClient


def load_ebola_knowledge():
    """加载非洲国家埃博拉历史数据，返回结构化知识。"""
    df = pd.read_csv("data/ebola_africa.csv")

    years = [2019, 2020, 2021, 2022, 2023]
    yearly_totals = {}
    for y in years:
        yearly_totals[y] = int(df[f"cases_{y}"].sum())

    country_profiles = []
    for _, row in df.iterrows():
        total_5y = sum(int(row[f"cases_{y}"]) for y in years)
        if total_5y > 0:
            country_profiles.append({
                "name": row["country"],
                "total_5y": total_5y,
                "level": "历史重灾区" if total_5y > 1000 else (
                    "近期疫区" if total_5y > 10 else "散发疫区"),
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


def build_ebola_prompt(user_df, predict_days, knowledge):
    """构建埃博拉专用的填空式预测 prompt。

    核心特征：
    - 果蝠天然宿主，人传人为主，无空气传播
    - 间歇性暴发，非全年散发
    - 病死率约50%（扎伊尔型可达90%）
    - 潜伏期2-21天（常见5-9天）
    """
    cases = user_df["cases"].values.astype(int)
    dates = user_df["date"].values
    total_cases = int(cases.sum())
    non_zero_days = int((cases > 0).sum())
    max_case = int(cases.max())

    if max_case >= 20:
        level = "活跃暴发期"
    elif max_case >= 3:
        level = "小型聚集性"
    else:
        level = "零星散发/监测期"

    similar_countries = []
    for c in knowledge["country_profiles"]:
        if c["total_5y"] > 0:
            similar_countries.append(c)
    similar_countries.sort(key=lambda x: x["total_5y"])

    ref_text = ""
    for sc in similar_countries[:3]:
        ref_text += f"参考: {sc['name']} (5年累计{sc['total_5y']}例, {sc['level']})\n"

    history_text = "\n".join(
        f"{pd.Timestamp(d).strftime('%Y-%m-%d')}: {cv} 例"
        for d, cv in zip(dates, cases)
    )

    prompt = f"""你是埃博拉病毒病（Ebola Virus Disease, EVD）疫情趋势分析工具。根据当前监测数据，客观分析并预测未来走势。

--- 背景参考（WHO埃博拉疫情特征）---
1. 传播途径: 直接接触患者体液（血液、唾液、呕吐物、粪便等），葬礼习俗为高风险环节
2. 天然宿主: 果蝠（狐蝠科），溢出事件后启动人传人传播链
3. 暴发模式: 间歇性暴发，非全年散发。单一溢出事件可触发大规模传播
4. 病死率: 扎伊尔型平均约50%（可达90%），苏丹型约50%
5. 潜伏期: 2-21天（常见5-9天），潜伏期不具传染性
6. 疫情结束标准: 末例后42天无新病例

--- 当前监测数据 ---
监测天数: {len(cases)} 天 ({dates[0]} 至 {dates[-1]})
累计病例: {total_cases} 例
日均病例: {cases.mean():.1f} 例
非零天数占比: {non_zero_days}/{len(cases)}
单日最高: {max_case} 例
疫情等级: {level}

每日详细记录:
{history_text}

--- 历史对照（非洲疫区）---
{ref_text}
--- 分析要求 ---
1. 根据当前数据的趋势（上升/下降/平稳）做外推，不预设结论
2. 埃博拉传染性极强，传播链一旦启动可快速扩散；但有效干预也可迅速清零
3. 每日数据应有自然波动，反映追踪检测的日间差异
4. 输出值应为整数，可为0

请预测未来 {predict_days} 天每日新增病例数。
以JSON数组输出: [{{"cases": 0}}, {{"cases": 1}}, ...]

预测JSON数组："""

    return prompt


def make_ebola_chart(user_df, predictions):
    plt.figure(figsize=(10, 5))

    dates = list(user_df["date"].dt.strftime("%m-%d"))
    values = list(user_df["cases"].values.astype(int))

    last_date = user_df["date"].iloc[-1]
    pred_dates = [(last_date + timedelta(days=i + 1)).strftime("%m-%d")
                  for i in range(len(predictions))]
    pred_values = [p.get("cases", 0) for p in predictions]

    plt.bar(range(len(dates)), values, color="#dc2626", alpha=0.7, label="已确诊病例")
    pred_x = range(len(dates), len(dates) + len(predictions))
    plt.bar(pred_x, pred_values, color="#f59e0b", alpha=0.7, label="预测病例")

    plt.axhline(y=0, color="#94a3b8", linewidth=0.5)

    all_dates = dates + pred_dates
    tick_pos = list(range(0, len(all_dates), max(1, len(all_dates) // 10)))
    plt.xticks(tick_pos, [all_dates[i] for i in tick_pos], rotation=45, fontsize=8)

    plt.ylabel("每日新增病例", fontsize=12)
    plt.title("埃博拉病毒病趋势预测", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = "data/chart.png"
    plt.savefig(path, dpi=100)
    plt.close()
    return path


_client = None
_knowledge = None


def get_client():
    global _client
    if _client is None:
        _client = LLMClient(
            model_path="D:/llm_models/qwen/Qwen2-1___5B-Instruct",
            temperature=0.9,
            max_tokens=512,
            device="cpu",
        )
    return _client


def get_knowledge():
    global _knowledge
    if _knowledge is None:
        _knowledge = load_ebola_knowledge()
    return _knowledge


def parse_json(text: str):
    code = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    target = code.group(1) if code else text
    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        m = re.search(pattern, target)
        if m:
            try:
                val = json.loads(m.group())
                return val if isinstance(val, list) else [val]
            except json.JSONDecodeError:
                pass
    return None


def predict_ebola(user_input_str, predict_days):
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
    prompt = build_ebola_prompt(user_df, predict_days, knowledge)

    client = get_client()
    raw = client.generate(prompt)
    predictions = parse_json(raw)

    if not predictions:
        return None, f"解析失败\n原始输出:\n{raw}", None

    predictions = predictions[:predict_days]
    chart_path = make_ebola_chart(user_df, predictions)

    cases = user_df["cases"].values.astype(int)
    pred_vals = [p.get("cases", 0) for p in predictions]
    total_pred = sum(pred_vals)

    if len(pred_vals) >= 3:
        first_half = np.mean(pred_vals[:len(pred_vals)//2])
        second_half = np.mean(pred_vals[len(pred_vals)//2:])
        if second_half > first_half * 1.3:
            trend_judgment = "上升趋势 — 传播链可能正在扩大，需加强追踪和隔离"
        elif second_half < first_half * 0.7:
            trend_judgment = "下降趋势 — 干预措施可能正在生效"
        else:
            trend_judgment = "平稳 — 疫情处于可控状态或散发阶段"
    else:
        trend_judgment = "数据不足"

    if cases.max() >= 20:
        alert_level = "活跃暴发"
    elif cases.max() >= 3:
        alert_level = "小型聚集"
    else:
        alert_level = "散发监测"

    report = (
        f"## 埃博拉病毒病趋势预测\n\n"
        f"**当前等级**: {alert_level}\n\n"
        f"**趋势判断**: {trend_judgment}\n\n"
        f"**预测摘要**: 未来 {predict_days} 天预计共 {total_pred} 例，日均 {total_pred / predict_days:.1f} 例\n\n"
        f"**预测详情**:\n\n"
    )
    for i, p in enumerate(predictions):
        report += f"- 第{i + 1}天: {p.get('cases', '?')} 例\n"

    report += (
        f"\n**参考依据**:\n"
        f"- WHO数据: 非洲多国2019-2023年累计约 {knowledge['total_5y']} 例\n"
        f"- 传播特征: 人传人为主，果蝠天然宿主，无空气传播\n"
        f"- 潜伏期2-21天，当前数据趋势为主要判断依据\n"
    )

    return chart_path, report, prompt


import gradio as gr


def gradio_handler(user_input, predict_days):
    chart, report, prompt = predict_ebola(user_input, int(predict_days))
    if chart is None:
        return None, report, prompt or ""
    return chart, report, prompt or ""


CSS = """
.gradio-container { max-width: 1100px !important; }
.prompt-box textarea { font-family: 'Consolas', monospace; font-size: 12px; }
"""

with gr.Blocks(title="埃博拉病毒趋势预测", css=CSS) as demo:
    gr.Markdown("# 埃博拉病毒病（Ebola Virus Disease）趋势预测")
    gr.Markdown(
        "基于 WHO 疫情数据 + 本地大模型。"
        "根据当前监测数据趋势，客观预测埃博拉病毒病未来走势。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 控制面板")
            user_input = gr.Textbox(
                label="追加最新病例（可选，用逗号分隔）",
                placeholder="留空则使用 Excel 中已有数据\n或输入: 0, 2, 0, 1, 0",
                lines=2,
            )
            predict_days = gr.Slider(
                minimum=3, maximum=30, value=14, step=1,
                label="预测天数",
            )
            gr.Markdown("""
            **数据来源**: `病例时间统计.xlsx`

            **埃博拉背景**:
            - 果蝠天然宿主，人传人为主
            - 病死率约50%（扎伊尔型）
            - 间歇性暴发模式
            - 潜伏期2-21天
            """)
            submit_btn = gr.Button("开始预测", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### 预测结果")
            chart_output = gr.Image(label="趋势图", show_label=False)
            report_output = gr.Markdown(label="预测报告")

    with gr.Accordion("查看完整 Prompt", open=False):
        prompt_output = gr.Textbox(label="发送给 LLM 的完整 Prompt", lines=20,
                                   max_lines=40, elem_classes=["prompt-box"])

    submit_btn.click(
        fn=gradio_handler,
        inputs=[user_input, predict_days],
        outputs=[chart_output, report_output, prompt_output],
    )


if __name__ == "__main__":
    import os as _os
    _os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    demo.launch(server_name="0.0.0.0", server_port=8080, share=False, quiet=True, show_error=True)
