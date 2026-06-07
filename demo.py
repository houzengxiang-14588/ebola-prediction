"""疫情填空式预测 — 完整演示脚本"""

import sys, json, re, os, pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, "src")
from model.llm_client import LLMClient
from processor.cleaner import clean_series


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


def series_to_text(df: pd.DataFrame) -> str:
    """将 DataFrame 转为紧凑文本行。"""
    lines = []
    metric_cols = [c for c in df.columns if c != "date"]
    for _, row in df.iterrows():
        d = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
        parts = [f"{d}:"]
        for col in metric_cols:
            parts.append(f"{col}={int(row[col]):,}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def build_fill_prompt(df: pd.DataFrame, mask_days: int = 3) -> str:
    """构建填空式预测 prompt。"""
    metric_cols = [c for c in df.columns if c != "date"]
    metrics_str = ", ".join(metric_cols)
    history_text = series_to_text(df.tail(90))

    latest = df.iloc[-1]
    summary = (
        f"--- 数据摘要 ---\n"
        f"时间范围: {df['date'].iloc[0]} 至 {latest['date']}\n"
    )
    for col in metric_cols:
        summary += f"最新 {col}: {int(latest[col]):,}\n"

    # 构建完整示例
    example = "{" + ", ".join(f'"{c}": {int(df[c].iloc[-1])}' for c in metric_cols) + "}"

    return f"""你是一个疫情数据数值预测工具。仔细观察历史数据的数值变化规律和趋势，预测接下来 {mask_days} 天的值。

注意：预测值应与历史数据保持相同的数值量级。

历史数据（每一天的指标值）：
{history_text}
{summary}
请预测接下来 {mask_days} 天每天的 {metrics_str}。
输出一个JSON数组，数组中每个元素代表一天，包含全部 {len(metric_cols)} 个字段。
不要输出日期、不要解释文字，只输出JSON数组。

示例格式（每个元素包含全部字段）：
[{example}, {example}]

JSON数组："""


def demo():
    print("=" * 60)
    print("  疫情趋势填空式预测工具 — 演示")
    print("  数据: 纽约市 2020-2024 真实疫情时序数据")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载真实疫情数据...")
    df = pd.read_csv("data/nyc_epidemic.csv", parse_dates=["date"])
    df = clean_series(df, min_length=30, smooth_window=7)
    metric_cols = [c for c in df.columns if c != "date"]
    print(f"  数据: {len(df)} 天 ({df['date'].min().date()} ~ {df['date'].max().date()})")
    print(f"  指标: {len(metric_cols)} 维 ({', '.join(metric_cols)})")
    print(f"  最新值:")
    for col in metric_cols:
        print(f"    {col}: {int(df[col].iloc[-1]):,}")

    # 2. 加载模型
    print("\n[2/4] 加载本地大模型 (Qwen2-1.5B-Instruct)...")
    client = LLMClient(
        model_path="D:/llm_models/qwen/Qwen2-1___5B-Instruct",
        temperature=0.3,
        max_tokens=256,
        device="cpu",
    )

    # 3. 构建填空式 prompt
    print("\n[3/4] 构建填空式预测 Prompt...")
    MASK_DAYS = 3
    context_df = df.tail(60)
    prompt = build_fill_prompt(df, mask_days=MASK_DAYS)
    print(f"  Prompt 长度: {len(prompt)} 字符")
    print(f"  上下文: {len(context_df)} 天已知 → [MASK] {MASK_DAYS} 天待填空")
    print(f"\n  --- Prompt 关键部分（尾部）---")
    print(prompt[-600:])

    # 4. LLM 填空预测
    print(f"\n[4/4] LLM 填空预测...")
    raw = client.generate(prompt)
    print(f"  模型原始输出:\n{raw}")

    predictions = parse_json(raw)
    if predictions:
        print(f"\n  [OK] 解析出 {len(predictions)} 天预测:")
        last_date = pd.Timestamp(df["date"].iloc[-1])
        for i, pred in enumerate(predictions[:MASK_DAYS]):
            d = last_date + pd.Timedelta(days=i + 1)
            vals = [f"{col}={pred.get(col, '?')}" for col in metric_cols]
            print(f"    {d.strftime('%Y-%m-%d')}: {', '.join(vals)}")
    else:
        print("  [FAIL] 未能解析预测结果")

    # 5. 趋势分析
    print(f"\n{'=' * 60}")
    print("  趋势判断")
    print(f"{'=' * 60}")
    main_metric = metric_cols[0]
    last_val = df[main_metric].iloc[-1]
    if predictions:
        pred_val = predictions[0].get(main_metric, last_val)
        delta = pred_val - last_val
        pct = delta / last_val * 100 if last_val else 0
        if pct > 5:
            direction = "/\\ 明显上升"
        elif pct > 1:
            direction = "/\\ 小幅上升"
        elif pct < -5:
            direction = "\\/ 明显下降"
        elif pct < -1:
            direction = "\\/ 小幅下降"
        else:
            direction = "-> 平稳"
        print(f"  主要指标 {main_metric}: {int(last_val):,}")
        print(f"  预测走向: {direction} ({delta:+,.0f}, {pct:+.1f}%)")
    print(f"\n  1.5B轻量化小模型侧重视趋势走向研判，不侧重精准数值输出。")


if __name__ == "__main__":
    demo()
