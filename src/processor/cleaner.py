"""数据处理模块 — 清洗、标准化、滑窗切分。"""

import pandas as pd
import numpy as np


def clean_series(
    df: pd.DataFrame,
    min_length: int = 30,
    smooth_window: int = 7,
    fill_method: str = "linear",
) -> pd.DataFrame:
    """清洗时序数据：去重、排序、缺失值填充、可选平滑。

    Args:
        df: 含 date, confirmed, deaths, recovered 列的 DataFrame。
        min_length: 少于该天数的序列将被丢弃。
        smooth_window: 滑动平均窗口大小，0 表示不平滑。
        fill_method: linear（线性插值）/ forward（前向填充）/ zero。

    Returns:
        清洗后的 DataFrame，按 date 排序。
    """
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    if len(df) < min_length:
        raise ValueError(f"数据仅 {len(df)} 天，少于最小要求 {min_length} 天")

    for col in ["confirmed", "deaths", "recovered"]:
        if col in df.columns:
            df[col] = _fill_missing(df[col], method=fill_method)
            if smooth_window > 0:
                df[col] = df[col].rolling(smooth_window, min_periods=1, center=True).mean()

    return df


def _fill_missing(series: pd.Series, method: str) -> pd.Series:
    if method == "linear":
        return series.interpolate(method="linear").fillna(0)
    elif method == "forward":
        return series.ffill().fillna(0)
    else:
        return series.fillna(0)


def sliding_windows(df: pd.DataFrame, context_days: int, mask_days: int):
    """将长时序切分为 (上下文, 填空答案) 片段，用于评估或微调。

    Yields:
        (context_df, answer_df): 上下文片段和答案片段。
    """
    total = context_days + mask_days
    for i in range(0, len(df) - total + 1):
        yield (
            df.iloc[i : i + context_days],
            df.iloc[i + context_days : i + total],
        )
