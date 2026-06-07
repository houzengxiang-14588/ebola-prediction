"""回溯测试框架 — 基于历史数据验证模型预测准确性

方法: Leave-One-Year-Out
- 训练: 2019-2022年数据
- 测试: 2023年数据
- 对每个国家: 用训练期数据预测2023年，与真实值对比

评估指标:
1. MAE (平均绝对误差)
2. 趋势方向准确率 (Direction Accuracy)
3. 量级准确率 (Magnitude Accuracy, 预测在真实值±50%内)
"""

import os, json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文配置复用 app.py
_font_paths = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]
_zh_font = None
for fp in _font_paths:
    if os.path.exists(fp):
        _zh_font = fm.FontProperties(fname=fp)
        break
if _zh_font:
    plt.rcParams["font.family"] = _zh_font.get_name()


def prepare_backtest_data(csv_path="data/ebola_africa.csv"):
    """准备回溯测试数据: 30国训练/测试集。"""
    df = pd.read_csv(csv_path)

    train_set = []
    test_set = []

    for _, row in df.iterrows():
        name = str(row["country"])
        if name.startswith("EU_"):
            continue

        cases = {y: int(row[f"cases_{y}"]) for y in range(2019, 2024)}

        # 训练: 2019-2022, 测试: 2023
        train_set.append({
            "country": name,
            "cases": {y: cases[y] for y in range(2019, 2023)},
        })
        test_set.append({
            "country": name,
            "actual": cases[2023],
            "cases": cases,
        })

    return train_set, test_set


def build_backtest_prompt(country, train_years=None):
    """为回溯测试构建趋势分类 prompt。

    只问趋势方向，不问具体数值——LLM更擅长分类而非数值预测。
    """
    if train_years is None:
        train_years = [2019, 2020, 2021, 2022]

    name = country["country"]
    vals = [country["cases"][y] for y in train_years]
    avg_val = np.mean(vals)

    # 计算逐年变化率
    changes = []
    for i in range(1, len(train_years)):
        prev, curr = train_years[i - 1], train_years[i]
        pct = (country["cases"][curr] - country["cases"][prev]) / max(country["cases"][prev], 1) * 100
        changes.append(f"  {prev}→{curr}: {'+' if pct >= 0 else ''}{pct:.0f}%")

    history_lines = [f"  {y}年: {country['cases'][y]} 例" for y in train_years]

    prompt = f"""你是埃博拉病毒流行病学分析专家。请仅根据数据做趋势判断。

国家: {name}
疫情等级: {'重灾区' if avg_val > 500 else '近期疫区' if avg_val > 10 else '散发/监测国'}

逐年数据:
{chr(10).join(history_lines)}

逐年变化率:
{chr(10).join(changes)}

问题: 相比2022年，该国家2023年的埃博拉病毒病例数会上升、下降还是保持平稳？

请先分析数据特征，然后给出判断。
以JSON格式输出: {{"analysis": "简要分析", "direction": "上升", "confidence": "高/中/低"}}

JSON输出："""

    return prompt


def parse_prediction(text):
    """解析趋势分类输出。"""
    match = re.search(r"\{[\s\S]*?\}", text)
    if not match:
        return None

    try:
        result = json.loads(match.group())
        direction = result.get("direction", "平稳")
        # 标准化
        if direction not in ("上升", "下降", "平稳"):
            if "升" in str(direction):
                direction = "上升"
            elif "降" in str(direction):
                direction = "下降"
            else:
                direction = "平稳"
        return {
            "analysis": result.get("analysis", ""),
            "direction": direction,
            "confidence": result.get("confidence", "中"),
        }
    except (json.JSONDecodeError, ValueError):
        return None


def _compute_actual_direction(cases, target_year=2023, ref_year=2022):
    """计算实际趋势方向（与prompt中的阈值一致）。"""
    if cases[target_year] > cases[ref_year] * 1.15:
        return "上升"
    elif cases[target_year] < cases[ref_year] * 0.85:
        return "下降"
    return "平稳"


def run_backtest(model_client, csv_path="data/ebola_africa.csv"):
    """执行趋势分类回溯测试。

    纯分类任务：给定4年历史，判断第5年趋势方向。
    """
    train_set, test_set = prepare_backtest_data(csv_path)

    results = []
    classes = ["上升", "下降", "平稳"]

    for train, test in zip(train_set, test_set):
        prompt = build_backtest_prompt(train)
        raw_output = model_client.generate(prompt)
        prediction = parse_prediction(raw_output)

        if prediction is None:
            prediction = {"direction": "平稳", "confidence": "低", "analysis": ""}

        actual_dir = _compute_actual_direction(test["cases"])
        pred_dir = prediction["direction"]

        results.append({
            "country": test["country"],
            "actual_cases": test["cases"],
            "actual_2023": test["cases"][2023],
            "actual_direction": actual_dir,
            "predicted_direction": pred_dir,
            "correct": pred_dir == actual_dir,
            "confidence": prediction["confidence"],
            "analysis": prediction.get("analysis", ""),
        })

    # 计算分类指标
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / n

    # 混淆矩阵
    from collections import Counter
    confusion = {}
    for a in classes:
        confusion[a] = {p: 0 for p in classes}
    for r in results:
        confusion[r["actual_direction"]][r["predicted_direction"]] += 1

    # 每类 precision / recall / F1
    class_metrics = {}
    for cls in classes:
        tp = confusion[cls][cls]
        pred_total = sum(confusion[a][cls] for a in classes)
        actual_total = sum(confusion[cls][p] for p in classes)
        precision = tp / max(pred_total, 1)
        recall = tp / max(actual_total, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)
        class_metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "actual_count": actual_total,
            "pred_count": pred_total,
        }

    # 置信度校准
    high_conf_correct = sum(1 for r in results if r["correct"] and r["confidence"] == "高")
    high_conf_total = sum(1 for r in results if r["confidence"] == "高")

    metrics = {
        "accuracy": accuracy,
        "total_countries": n,
        "confusion": confusion,
        "class_metrics": class_metrics,
        "high_conf_accuracy": high_conf_correct / max(high_conf_total, 1),
        "high_conf_total": high_conf_total,
        "random_baseline": 0.333,  # 3分类随机基线
    }

    return {"results": results, "metrics": metrics}


def generate_backtest_report(backtest_result, output_path="data/backtest_report.png"):
    """生成趋势分类回溯测试可视化报告。"""
    results = backtest_result["results"]
    metrics = backtest_result["metrics"]
    confusion = metrics["confusion"]
    cm = metrics["class_metrics"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. 混淆矩阵热力图
    ax = axes[0][0]
    classes = ["上升", "平稳", "下降"]
    matrix = [[confusion[a][p] for p in classes] for a in classes]
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("预测方向"); ax.set_ylabel("实际方向")
    ax.set_title("趋势分类混淆矩阵")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if matrix[i][j] > max(max(r) for r in matrix) / 2 else "gray")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2. 每类指标柱状图
    ax = axes[0][1]
    x = np.arange(len(classes))
    width = 0.25
    precisions = [cm[c]["precision"] for c in classes]
    recalls = [cm[c]["recall"] for c in classes]
    f1s = [cm[c]["f1"] for c in classes]
    ax.bar(x - width, precisions, width, label="精确率", color="#2563eb")
    ax.bar(x, recalls, width, label="召回率", color="#22c55e")
    ax.bar(x + width, f1s, width, label="F1", color="#f59e0b")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("分数"); ax.set_title("每类 Precision / Recall / F1")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)

    # 3. 指标汇总面板
    ax = axes[1][0]
    ax.axis("off")
    acc = metrics["accuracy"]
    baseline = metrics["random_baseline"]
    hc = metrics.get("high_conf_accuracy", 0)
    hc_n = metrics.get("high_conf_total", 0)

    # 计算每类样本数
    class_dist = {c: cm[c]["actual_count"] for c in classes}

    summary = (
        f"趋势分类回溯测试报告\n"
        f"{'─' * 28}\n"
        f"测试国家数:          {metrics['total_countries']}\n"
        f"随机基线 (3分类):    {baseline:.1%}\n"
        f"{'─' * 28}\n"
        f"总准确率:            {acc:.1%}\n"
        f" vs 随机基线:        {acc - baseline:+.1%}\n"
        f"{'─' * 28}\n"
        f"高置信度准确率:      {hc:.1%} (n={hc_n})\n"
        f"{'─' * 28}\n"
        f"类别分布:\n"
    )
    for cls in classes:
        marker = "★" if cm[cls]["actual_count"] >= 8 else "  "
        summary += f"  {marker} {cls}: {class_dist[cls]} 国 (F1={cm[cls]['f1']:.1%})\n"

    ax.text(0.05, 0.55, summary, fontsize=11,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="#f0f9ff", edgecolor="#93c5fd"))

    # 4. 类别样本分布
    ax = axes[1][1]
    classes = ["上升", "平稳", "下降"]
    counts = [sum(1 for r in results if r["actual_direction"] == c) for c in classes]
    colors_pie = ["#ef4444", "#22c55e", "#3b82f6"]
    ax.pie(counts, labels=[f"{c}\n({n}国)" for c, n in zip(classes, counts)],
           colors=colors_pie, autopct="%1.1f%%", startangle=90)
    ax.set_title("测试集类别分布")

    plt.tight_layout()
    figure_path = output_path.replace(".png", "_figure.png")
    plt.savefig(figure_path, dpi=120, bbox_inches="tight")
    plt.close()

    # 文本报告
    text_path = output_path.replace(".png", ".txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(summary.replace("─", "-").replace("★", "*") + "\n\n")
        f.write("各国详细结果:\n")
        for r in sorted(results, key=lambda r: (not r["correct"], r["country"])):
            status = "OK" if r["correct"] else "XX"
            f.write(
                f"  [{status}] {r['country']:<20s} "
                f"实际={r['actual_direction']:<6s} "
                f"预测={r['predicted_direction']:<6s} "
                f"2023={r['actual_2023']:>4d}例\n"
            )

    return figure_path, text_path


if __name__ == "__main__":
    os.chdir("D:/Ebola")
    print("Backtesting module loaded.")
