"""训练样本构造器 — 从ECDC CSV构建标准化训练数据

策略：
- 每个国家5年数据(2019-2023)，用滑动窗口生成多条样本
- 窗口组合: (context=2, target=2), (context=3, target=1), (context=3, target=2)
- 对高发国家(芬兰/德国等)和低发国家分别构造，保证数据多样性
- 增加国家类比样本：用相似发病率国家互相推断
"""

import json, os
import numpy as np
import pandas as pd

from .templates import TEMPLATES

YEARS = [2019, 2020, 2021, 2022, 2023]
MONTHLY_WEIGHTS = {
    1: 0.04, 2: 0.04, 3: 0.05, 4: 0.06, 5: 0.08, 6: 0.10,
    7: 0.14, 8: 0.14, 9: 0.09, 10: 0.08, 11: 0.06, 12: 0.05,
}


def load_ecdc_data(csv_path="data/ebola_africa.csv"):
    """加载ECDC CSV并返回国家列表。"""
    df = pd.read_csv(csv_path)
    countries = []
    for _, row in df.iterrows():
        name = str(row["country"])
        if name.startswith("EU_"):
            continue
        cases = {y: int(row[f"cases_{y}"]) for y in YEARS}
        rates = {y: float(row[f"rate_{y}"]) for y in YEARS}
        countries.append({"name": name, "cases": cases, "rates": rates})
    return countries


def _format_year_data(country, years):
    """格式化指定年份的病例数据为文本。"""
    lines = []
    for y in sorted(years):
        lines.append(f"  {y}年: {country['cases'][y]} 例 (发病率 {country['rates'][y]:.1f}/10万)")
    return "\n".join(lines)


def _classify_level(avg_rate):
    if avg_rate > 5:
        return "高发"
    elif avg_rate > 0.5:
        return "中发"
    else:
        return "低发/散发"


def _trend_direction(country, years):
    vals = [country["cases"][y] for y in years if y in country["cases"]]
    if len(vals) < 2:
        return "平稳"
    first = np.mean(list(vals)[:len(vals)//2])
    second = np.mean(list(vals)[len(vals)//2:])
    if second > first * 1.3:
        return "上升"
    elif second < first * 0.7:
        return "下降"
    return "平稳"


def build_trend_forecast_samples(countries, output_path="data/training/"):
    """生成趋势外推训练样本。

    滑动窗口: context_years=[2,3], target_years=[1,2]
    """
    samples = []
    window_configs = [
        (2, 2), (3, 1), (3, 2), (2, 1),
    ]

    for country in countries:
        name = country["name"]
        avg_rate = np.mean(list(country["rates"].values()))
        level = _classify_level(avg_rate)

        for ctx_n, tgt_n in window_configs:
            # 滑动窗口
            for start in range(len(YEARS) - ctx_n - tgt_n + 1):
                ctx_years = YEARS[start:start + ctx_n]
                tgt_years = YEARS[start + ctx_n:start + ctx_n + tgt_n]

                context_text = f"国家: {name}\n发病率等级: {level}\n"
                context_text += "历史数据:\n" + _format_year_data(country, ctx_years)

                target_text = ", ".join(str(y) for y in tgt_years)

                # 构造instruction-tuning格式
                instruction = TEMPLATES["trend_forecast"].format(
                    context_data=context_text,
                    target_period=target_text,
                )

                # 答案: 目标年份真实数据
                answer = {
                    "predictions": [
                        {"year": y, "cases": country["cases"][y]}
                        for y in tgt_years
                    ],
                    "trend": _trend_direction(country, tgt_years),
                    "confidence": "高",
                }

                samples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": json.dumps(answer, ensure_ascii=False),
                    "country": name,
                    "type": "trend_forecast",
                })

    return samples


def build_country_analogy_samples(countries, output_path="data/training/"):
    """生成国家类比样本：用相似发病率国家互相推断。"""
    samples = []

    # 按发病率分组
    high = [c for c in countries if np.mean(list(c["rates"].values())) > 1]
    medium = [c for c in countries if 0.1 < np.mean(list(c["rates"].values())) <= 1]
    low = [c for c in countries if np.mean(list(c["rates"].values())) <= 0.1]

    groups = {"高发组": high, "中发组": medium, "低发组": low}

    for group_name, group in groups.items():
        if len(group) < 2:
            continue
        for i, target in enumerate(group):
            # 用同组其他国家作为参考
            refs = [c for j, c in enumerate(group) if j != i]
            if not refs:
                continue
            # 最多取3个参考国
            refs = sorted(refs, key=lambda c: sum(c["cases"].values()), reverse=True)[:3]

            ref_text = ""
            for ref in refs:
                ref_text += f"\n{ref['name']}:\n"
                ref_text += _format_year_data(ref, YEARS[:4])  # 2019-2022作为参考
                ref_text += "\n"

            target_context = _format_year_data(target, YEARS[:3])  # 2019-2021作为已知

            instruction = TEMPLATES["country_analogy"].format(
                reference_data=ref_text,
                target_country=target["name"],
                target_context=target_context,
                target_period="2022-2023",
            )

            answer = json.dumps({
                "predictions": [
                    {"year": 2022, "cases": target["cases"][2022]},
                    {"year": 2023, "cases": target["cases"][2023]},
                ],
                "similar_to": refs[0]["name"],
            }, ensure_ascii=False)

            samples.append({
                "instruction": instruction,
                "input": "",
                "output": answer,
                "country": target["name"],
                "type": "country_analogy",
            })

    return samples


def build_seasonal_samples(countries, output_path="data/training/"):
    """生成季节性填空样本：年度总量→月度分布的预测。"""
    samples = []

    for country in countries:
        if np.mean(list(country["cases"].values())) < 1:
            continue  # 跳过极低发国家(全年不到1例，无季节模式)

        for target_year in YEARS:
            yearly_total = country["cases"][target_year]
            if yearly_total < 3:
                continue

            # 上下文: 其他年份的年度总量
            context_years = [y for y in YEARS if y != target_year]
            context_lines = [f"  {y}年: {country['cases'][y]} 例" for y in context_years]

            # 给部分月份的已知数据(模拟)
            known_months = [1, 2, 3, 7, 8, 12]  # 关键月份: 冬+夏高峰
            monthly_context = []
            for m in known_months:
                est = int(yearly_total * MONTHLY_WEIGHTS[m])
                monthly_context.append(f"  {m}月: ~{est} 例")
            monthly_context.sort()

            instruction = TEMPLATES["seasonal_fill"].format(
                context_period=", ".join(str(y) for y in context_years),
                monthly_context="\n".join(monthly_context),
                target_period=f"{target_year}年1-12月",
                yearly_estimate=yearly_total,
            )

            monthly_truth = {
                m: int(yearly_total * MONTHLY_WEIGHTS[m]) for m in range(1, 13)
            }

            answer = json.dumps({
                "monthly_cases": [
                    {"month": m, "cases": monthly_truth[m]}
                    for m in range(1, 13)
                ],
            }, ensure_ascii=False)

            samples.append({
                "instruction": instruction,
                "input": "",
                "output": answer,
                "country": country["name"],
                "type": "seasonal_fill",
            })

    return samples


def build_all_samples(csv_path="data/ebola_africa.csv", output_dir="data/training/"):
    """构建全部训练样本，保存为JSONL。"""
    os.makedirs(output_dir, exist_ok=True)
    countries = load_ecdc_data(csv_path)

    all_samples = []
    all_samples.extend(build_trend_forecast_samples(countries))
    all_samples.extend(build_country_analogy_samples(countries))
    all_samples.extend(build_seasonal_samples(countries))

    # 去重（基于instruction文本）
    seen = set()
    unique = []
    for s in all_samples:
        key = s["instruction"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(s)
    all_samples = unique

    # 保存
    output_path = os.path.join(output_dir, "samples.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 统计
    type_counts = {}
    for s in all_samples:
        t = s["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"训练样本生成完成: {len(all_samples)} 条")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c} 条")
    print(f"已保存到: {output_path}")
    return all_samples


if __name__ == "__main__":
    os.chdir("D:/Ebola")
    build_all_samples()
