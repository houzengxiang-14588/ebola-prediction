"""Prompt 构建器 — 将时序数据转化为填空式自然语言 prompt。"""

import pandas as pd


TEMPLATE = """你是一个疫情数据数值预测工具。仔细观察历史数据的数值变化规律和趋势，预测接下来 {mask_days} 天的值。

注意：预测值应与历史数据保持相同的数值量级。

历史数据（每一天的指标值）：
{history}

请预测接下来 {mask_days} 天每天的 {metrics}。
输出一个JSON数组，数组中每个元素代表一天，包含全部 {num_metrics} 个字段。
不要输出日期、不要解释文字，只输出JSON数组。

示例格式（每个元素包含全部字段）：
[{example}, {example}]

JSON数组："""


def _series_to_text(df: pd.DataFrame) -> str:
    """将 DataFrame 转为紧凑的文本行。"""
    metric_cols = [c for c in df.columns if c != "date"]
    lines = []
    for _, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
        parts = [f"{date_str}:"]
        for col in metric_cols:
            parts.append(f"{col}={int(row[col]):,}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def build_fill_prompt(
    context_df: pd.DataFrame,
    mask_days: int = 7,
    template: str | None = None,
) -> str:
    """构建填空式预测 prompt。

    Args:
        context_df: 已知的历史时序数据（已清洗）。
        mask_days: 需要预测的天数。
        template: 自定义 prompt 模板。

    Returns:
        完整的 prompt 字符串，可直接送入 LLM。
    """
    if template is None:
        template = TEMPLATE

    metric_cols = [c for c in context_df.columns if c != "date"]
    metrics = ", ".join(metric_cols)
    num_metrics = len(metric_cols)

    latest = context_df.iloc[-1]
    example = "{" + ", ".join(f'"{col}": {int(latest[col])}' for col in metric_cols) + "}"

    history_text = _series_to_text(context_df.tail(90))

    summary_lines = [
        "--- 数据摘要 ---",
        f"时间范围: {context_df['date'].iloc[0]} 至 {latest['date']}",
    ]
    for col in metric_cols:
        summary_lines.append(f"最新 {col}: {int(latest[col]):,}")
    summary = "\n".join(summary_lines)

    return template.format(
        mask_days=mask_days,
        metrics=metrics,
        num_metrics=num_metrics,
        example=example,
        history=history_text + "\n" + summary,
    )
