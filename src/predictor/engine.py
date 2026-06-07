"""预测引擎 — 滚动填空式多步预测。

核心流程：
1. 取最近 context_days 天数据构建 prompt，mask 最后 mask_days 天。
2. 调用 LLM 填空，解析预测值。
3. 将预测值拼回序列尾部，窗口前移，重复预测。
4. 每步可多次采样取中位数以降低随机性。
"""

import json
import re
import statistics

import pandas as pd

from ..model.llm_client import LLMClient
from ..model.prompt_builder import build_fill_prompt


def _parse_json_array(text: str) -> list[dict] | None:
    """从 LLM 输出中提取 JSON，容错解析。支持数组和单个对象。"""
    # 先提取 ```json ... ``` 代码块
    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    target = code_match.group(1) if code_match else text

    # 尝试匹配 [...] 数组
    arr_match = re.search(r"\[[\s\S]*\]", target)
    if arr_match:
        try:
            return json.loads(arr_match.group())
        except json.JSONDecodeError:
            pass

    # 尝试匹配单个 {...} 对象
    obj_match = re.search(r"\{[\s\S]*\}", target)
    if obj_match:
        try:
            obj = json.loads(obj_match.group())
            return [obj] if isinstance(obj, dict) else obj
        except json.JSONDecodeError:
            pass

    return None


def roll_predict(
    client: LLMClient,
    df: pd.DataFrame,
    context_days: int = 60,
    mask_days: int = 3,
    roll_steps: int = 14,
    samples: int = 3,
) -> list[dict]:
    """滚动多步预测 — 填空式。

    每一步用最近 context_days 天数据做上下文，预测 mask_days 天，
    然后将预测值拼回历史序列，窗口前移，继续预测下一步。

    Args:
        client: LLM 客户端。
        df: 历史时序数据（已清洗）。
        context_days: 每次给模型的上下文窗口天数。
        mask_days: 每次填空预测的天数。
        roll_steps: 滚动总步数（总预测天数 = roll_steps * mask_days）。
        samples: 每步采样次数，取中位数以稳定结果。

    Returns:
        预测结果列表，每项 {date, confirmed, deaths, ...}。
    """
    history = df.copy()
    predictions = []

    for step in range(roll_steps):
        # 取最近 context_days 天作为上下文窗口
        window = history.tail(context_days)
        step_preds = []

        for _ in range(samples):
            prompt = build_fill_prompt(window, mask_days=mask_days)
            raw = client.generate(prompt)
            parsed = _parse_json_array(raw)
            if parsed and len(parsed) >= 1:
                step_preds.append(parsed)

        if not step_preds:
            break

        # 按位置取中位数，仅保留目标字段
        TARGET_FIELDS = {"confirmed", "deaths", "recovered"}
        best = []
        for i in range(min(len(step_preds[0]), mask_days)):
            values: dict = {}
            for pred in step_preds:
                if i < len(pred):
                    for k, v in pred[i].items():
                        if k not in TARGET_FIELDS:
                            continue
                        if isinstance(v, (int, float)) and v >= 0:
                            values.setdefault(k, []).append(v)
            median_entry = {}
            for k, v_list in values.items():
                try:
                    median_entry[k] = int(statistics.median([float(v) for v in v_list]))
                except (ValueError, TypeError):
                    pass
            if median_entry:
                best.append(median_entry)

        # 生成日期并拼回历史
        last_date = pd.Timestamp(history["date"].iloc[-1])
        new_rows = []
        for j, entry in enumerate(best):
            entry["date"] = (last_date + pd.Timedelta(days=j + 1)).strftime("%Y-%m-%d")
            new_rows.append(entry)

        predictions.extend(new_rows)

        # 将预测值拼回历史序列，供下一步使用
        new_df = pd.DataFrame(new_rows)
        for col in history.columns:
            if col not in new_df.columns:
                new_df[col] = 0
        history = pd.concat([history, new_df[history.columns]], ignore_index=True)

    return predictions
