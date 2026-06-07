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


def build_backtest_prompt(country, train_years=None, rag_context=""):
    """为回溯测试构建趋势分类 prompt。

    只问趋势方向，不问具体数值——LLM更擅长分类而非数值预测。
    支持可选的 RAG 知识上下文注入。
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

    rag_block = ""
    if rag_context:
        rag_block = f"\n历史相关知识（RAG检索）:\n{rag_context}\n"

    prompt = f"""你是埃博拉病毒流行病学分析专家。请仅根据数据做趋势判断。

国家: {name}
疫情等级: {'重灾区' if avg_val > 500 else '近期疫区' if avg_val > 10 else '散发/监测国'}

逐年数据:
{chr(10).join(history_lines)}

逐年变化率:
{chr(10).join(changes)}
{rag_block}
问题: 相比{train_years[-1]}年，该国家{train_years[-1]+1}年的埃博拉病毒病例数会上升、下降还是保持平稳？

请先分析数据特征，然后给出判断。
以JSON格式输出: {{"analysis": "简要分析", "direction": "上升", "confidence": "高/中/低"}}

JSON输出："""

    return prompt


def build_rag_context(retriever, country_cases, country_name):
    """从 RAG 知识库检索与目标国家疫情相关的历史知识。

    Args:
        retriever: KnowledgeRetriever 实例或 None
        country_cases: {year: count} 字典
        country_name: 国家名

    Returns:
        str: 格式化的知识文本, 或空字符串
    """
    if retriever is None:
        return ""
    try:
        daily = [country_cases.get(y, 0) for y in range(2019, 2024)]
        total = sum(daily)
        max_case = max(daily) if daily else 0
        results = retriever.retrieve_for_prediction(
            daily,
            metadata={
                "total_cases": total,
                "max_case": max_case,
                "country": country_name,
            },
        )
        lines = [
            f"  - {item['content'][:200]}"
            for item in results.get("knowledge", [])[:3]
        ]
        return "\n".join(lines) if lines else ""
    except Exception as e:
        print(f"  [WARN] RAG retrieval failed for {country_name}: {e}")
        return ""


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


def run_ablation_study(
    model_base_path="D:/llm_models/qwen/Qwen2-7B-Instruct",
    lora_path="lora_weights/final",
    csv_path="data/ebola_africa.csv",
    chroma_dir="chroma_db",
    output_path="data/ablation_results.json",
):
    """运行三组消融实验: Base / LoRA / LoRA+RAG.

    使用 WHO 周度评估集（data/eval/weekly_cases.json），
    3 个核心暴发国家 x ~64 周滑动窗口 ≈ 162 有效样本点，
    三类均有充足样本（上升~43, 下降~88, 平稳~31）。
    """
    from src.model.lora_model import LoraModelClient
    import torch, gc

    os.chdir("D:/Ebola")
    countries = load_weekly_eval_data()
    total_weeks = sum(len(c["data"]) for c in countries)
    print(f"Loaded {len(countries)} countries, {total_weeks} total weekly data points")

    def _save_progress(data, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # 断点续跑：恢复已完成组
    all_results = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"从 {output_path} 恢复了 {list(all_results.keys())} 组结果")
        except Exception:
            pass

    need_base = "base" not in all_results
    need_lora = "lora" not in all_results
    need_lora_rag = "lora_rag" not in all_results
    need_model_with_lora = need_lora or need_lora_rag
    any_need_model = need_base or need_model_with_lora

    if any_need_model:
        # 需要 LoRA 的组直接加载 LoRA 模型，不需要的从 base 开始后加载
        init_lora = lora_path if need_model_with_lora and not need_base else None
        client = LoraModelClient(
            model_path=model_base_path, lora_path=init_lora,
            temperature=0.3, max_tokens=256, load_in_4bit=True,
        )

    # ── 1. Base ──
    if need_base:
        print("\n" + "=" * 50)
        print("[1/3] Base model (no finetune, no RAG)")
        print("=" * 50)
        all_results["base"] = evaluate_sliding_window(client, countries)
        _print_ablation_summary("Base", all_results["base"]["metrics"])
        _save_progress(all_results, output_path)
        # Base 跑完后加载 LoRA 供后续组使用
        if need_model_with_lora:
            client.load_lora(lora_path)
    else:
        print("\n[1/3] Base -- 跳过 (已有结果)")

    # ── 2. LoRA ──
    if need_lora:
        print("\n" + "=" * 50)
        print("[2/3] LoRA model (finetuned, no RAG)")
        print("=" * 50)
        all_results["lora"] = evaluate_sliding_window(client, countries)
        _print_ablation_summary("LoRA", all_results["lora"]["metrics"])
        _save_progress(all_results, output_path)
    else:
        print("\n[2/3] LoRA -- 跳过 (已有结果)")

    # ── 3. LoRA + RAG ──
    if need_lora_rag:
        print("\n" + "=" * 50)
        print("[3/3] LoRA + RAG model (finetuned + ChromaDB knowledge)")
        print("=" * 50)
        retriever = None
        try:
            from src.rag.vector_store import VectorStore
            from src.rag.retriever import KnowledgeRetriever
            # 直接使用本地缓存的 embedding 模型，避免 HuggingFace 下载超时
            local_embed = "C:/Users/86136/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
            vs = VectorStore(persist_dir=chroma_dir)
            retriever = KnowledgeRetriever(vs, embed_model=local_embed, top_k=5)
            kc = vs.get_knowledge_collection().count()
            cc = vs.get_country_collection().count()
            print(f"  ChromaDB: ebola_knowledge={kc}, country_profiles={cc}")
        except Exception as e:
            print(f"  RAG initialization failed: {e}")
        all_results["lora_rag"] = evaluate_sliding_window(
            client, countries, retriever=retriever
        )
        _print_ablation_summary("LoRA+RAG", all_results["lora_rag"]["metrics"])
        _save_progress(all_results, output_path)
    else:
        print("\n[3/3] LoRA+RAG -- 跳过 (已有结果)")

    if any_need_model:
        del client
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n消融实验结果已保存至 {output_path}")
    return all_results


def load_weekly_eval_data(path="data/eval/weekly_cases.json"):
    """加载基于 WHO 数据库构建的周粒度评估集。

    Returns:
        [{"country": str, "data": [(week_label, cases), ...], "cases_dict": {idx: cases}}]
        每个国家有 ~72 周数据, 约 64 个滑动窗口。
    """
    import json as _json
    with open(path, "r", encoding="utf-8") as f:
        raw = _json.load(f)

    countries = []
    for name, d in raw.items():
        data = list(zip(d["dates"], d["cases"]))
        countries.append({
            "country": name,
            "data": data,  # [(date_str, cases_float), ...]
            "cases_dict": {i: v for i, (_, v) in enumerate(data)},
        })
    return countries


def evaluate_sliding_window(model_client, countries, retriever=None, context_window=8):
    """滑动窗口评估: 用前 N 个时间步作为上下文, 预测下一步趋势。

    同时支持年度数据 ({country, cases: {year: val}}) 和周度数据
    ({country, data: [(label, val), ...], cases_dict: {idx: val}})。

    对于周度数据, 跳过最近 8 周均值 < 5 且目标 < 5 的噪音窗口。
    """
    results = []
    is_weekly = "data" in countries[0]  # 检测数据格式

    for ci in countries:
        name = ci["country"]

        if is_weekly:
            data = ci["data"]
            cases_dict = ci["cases_dict"]
            total_weeks = len(data)

            for i in range(context_window, total_weeks):
                ctx_indices = list(range(i - context_window, i))
                tgt_idx = i
                ctx_vals = [cases_dict[j] for j in ctx_indices]
                tgt_val = cases_dict[tgt_idx]

                # 跳过全零窗口
                if max(ctx_vals) < 5 and tgt_val < 5:
                    continue

                rag_context = build_rag_context(retriever, cases_dict, name)

                prompt = _build_weekly_prompt(name, data, ctx_indices, tgt_idx, rag_context)
                raw = model_client.generate(prompt)
                pred = parse_prediction(raw)
                if pred is None:
                    pred = {"direction": "平稳", "confidence": "低", "analysis": ""}

                actual_dir = _compute_weekly_direction(ctx_vals, tgt_val)
                results.append({
                    "country": name,
                    "window_end": tgt_idx,
                    "context_len": context_window,
                    "actual_direction": actual_dir,
                    "predicted_direction": pred["direction"],
                    "correct": pred["direction"] == actual_dir,
                    "confidence": pred["confidence"],
                    "analysis": pred.get("analysis", ""),
                })

                if len(results) % 20 == 0:
                    print(f"  Progress: {len(results)} samples")
                    _save_intermediate_results(results, "data/backtest_partial.json")

        else:
            # 原有年度数据逻辑
            cases = ci["cases"]
            years = sorted(cases.keys())
            for i in range(context_window, len(years)):
                context_years = years[i - context_window:i]
                target_year = years[i]
                ctx_vals = [cases[y] for y in context_years]
                if max(ctx_vals) == 0 and cases[target_year] == 0:
                    continue
                rag_context = build_rag_context(retriever, cases, name)
                prompt = build_backtest_prompt(ci, context_years, rag_context)
                raw = model_client.generate(prompt)
                pred = parse_prediction(raw)
                if pred is None:
                    pred = {"direction": "平稳", "confidence": "低", "analysis": ""}
                actual_dir = _compute_actual_direction(cases, target_year, context_years[-1])
                results.append({
                    "country": name,
                    "target_year": target_year,
                    "context_years": context_years,
                    "actual_direction": actual_dir,
                    "predicted_direction": pred["direction"],
                    "correct": pred["direction"] == actual_dir,
                    "confidence": pred["confidence"],
                    "analysis": pred.get("analysis", ""),
                })
                if len(results) % 5 == 0:
                    _save_intermediate_results(results, "data/backtest_partial.json")

    return _compute_sliding_metrics(results)


def _build_weekly_prompt(name, data, ctx_indices, tgt_idx, rag_context=""):
    """为周度数据构建趋势分类 prompt。"""
    ctx_vals = [data[j][1] for j in ctx_indices]
    tgt_date = data[tgt_idx][0]

    history = "\n".join(
        f"  第{j+1}周 ({data[j][0]}): {data[j][1]:.0f} 例"
        for j in ctx_indices
    )

    recent_avg = np.mean(ctx_vals[-3:]) if ctx_vals else 0
    earlier_avg = np.mean(ctx_vals[:3]) if ctx_vals else 0
    if earlier_avg > 0:
        trend_desc = "上升" if recent_avg > earlier_avg * 1.3 else \
                     "下降" if recent_avg < earlier_avg * 0.7 else "波动"
    else:
        trend_desc = "初始阶段"

    rag_block = f"\n历史相关知识（RAG检索）:\n{rag_context}\n" if rag_context else ""

    prompt = (
        f"你是埃博拉病毒流行病学分析专家。请根据数据判断趋势。\n"
        f"\n国家: {name}\n"
        f"\n近{len(ctx_indices)}周病例数据:\n{history}\n"
        f"\n近期走势: {trend_desc}\n"
        f"近期均值: {recent_avg:.0f} 例/周 (前段: {earlier_avg:.0f} 例/周)\n"
        f"{rag_block}"
        f"\n问题: 在{tgt_date}这周，{name}的埃博拉病例数相比近期均值是上升、下降还是平稳？\n"
        f"\n请先分析数据，然后以JSON输出: "
        f'{{"analysis": "简要分析", "direction": "上升", "confidence": "高/中/低"}}\n'
        f"\nJSON输出:"
    )
    return prompt


def _compute_weekly_direction(ctx_vals, tgt_val):
    """计算周度数据的实际趋势方向。"""
    ref = np.mean(ctx_vals[-3:]) if sum(ctx_vals[-3:]) > 0 else 1
    if tgt_val > ref * 1.15:
        return "上升"
    elif tgt_val < ref * 0.85:
        return "下降"
    return "平稳"


def _compute_sliding_metrics(results):
    """计算滑动窗口评估的分类指标。"""
    classes = ["上升", "下降", "平稳"]
    n = len(results)
    if n == 0:
        return {"results": [], "metrics": {}}

    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / n

    confusion = {a: {p: 0 for p in classes} for a in classes}
    for r in results:
        confusion[r["actual_direction"]][r["predicted_direction"]] += 1

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

    high_conf = [r for r in results if r["confidence"] == "高"]
    hc_correct = sum(1 for r in high_conf if r["correct"])

    return {
        "results": results,
        "metrics": {
            "accuracy": accuracy,
            "total_samples": n,
            "confusion": confusion,
            "class_metrics": class_metrics,
            "high_conf_accuracy": hc_correct / max(len(high_conf), 1),
            "high_conf_total": len(high_conf),
            "random_baseline": 0.333,
        },
    }


def _save_intermediate_results(results, path):
    """保存中间结果, 防止长时间运行中丢失数据。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = _compute_sliding_metrics(results)["metrics"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "metrics": m}, f, ensure_ascii=False, indent=2)


def _print_ablation_summary(name, metrics):
    """打印消融实验单组结果摘要。"""
    print(f"\n{name} 评估结果:")
    print(f"  样本数: {metrics.get('total_samples', 0)}")
    print(f"  准确率: {metrics.get('accuracy', 0):.1%}")
    print(f"  vs 随机基线 (33.3%): {metrics.get('accuracy', 0) - 0.333:+.1%}")
    if metrics.get("high_conf_total", 0) > 0:
        print(f"  高置信度准确率: {metrics['high_conf_accuracy']:.1%} (n={metrics['high_conf_total']})")
    for cls, cm in metrics.get("class_metrics", {}).items():
        print(f"  {cls}: P={cm['precision']:.1%} R={cm['recall']:.1%} F1={cm['f1']:.1%} (实际={cm['actual_count']}, 预测={cm['pred_count']})")


def _compute_actual_direction(cases, target_year, ref_year):
    """计算实际趋势方向（与 prompt 阈值一致）。"""
    if cases[target_year] > cases[ref_year] * 1.15:
        return "上升"
    elif cases[target_year] < cases[ref_year] * 0.85:
        return "下降"
    return "平稳"


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


def generate_figures(results_path="data/ablation_results.json", output_dir="figures"):
    """Generate three comparison figures from ablation results (300 DPI).

    Fig 1: Per-class Precision/Recall/F1 comparison across three model variants
    Fig 2: High vs low incidence country prediction performance
    Fig 3: Module ablation effect (incremental contribution)
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    matplotlib.rcParams.update({"font.size": 11, "axes.unicode_minus": False})

    groups = ["base", "lora", "lora_rag"]
    group_labels = ["Base", "LoRA", "LoRA+RAG"]
    group_colors = ["#94a3b8", "#f59e0b", "#22c55e"]
    classes = ["Rise", "Fall", "Stable"]
    class_keys = ["上升", "下降", "平稳"]

    # ── Fig 1: Per-class Precision / Recall / F1 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    metrics_map = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
    ]
    for m_idx, (metric_key, metric_title) in enumerate(metrics_map):
        ax = axes[m_idx]
        x = np.arange(len(classes))
        w = 0.22
        for i, (g, gl, gc) in enumerate(zip(groups, group_labels, group_colors)):
            vals = [data[g]["metrics"]["class_metrics"][ck][metric_key] for ck in class_keys]
            bars = ax.bar(x + (i - 1) * w, vals, w, label=gl, color=gc, edgecolor="white", linewidth=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                        f"{val:.2f}", ha="center", fontsize=7.5, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=10)
        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 0.75)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if m_idx == 0:
            ax.set_ylabel("Score")
        if m_idx == 2:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("Ablation Study — Per-Class Trend Detection Performance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "fig1_prf_by_class.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[1/3] {fig_path}")

    # ── Fig 2: High vs Low Incidence Countries ──
    high_countries = {"Liberia", "Sierra Leone"}
    low_countries = {"Guinea"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax_idx, (country_set, title, c_color) in enumerate([
        (high_countries, "High Incidence\n(Liberia + Sierra Leone)", "#ef4444"),
        (low_countries, "Low Incidence\n(Guinea)", "#3b82f6"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(groups))
        w = 0.22
        n = 0
        acc = 0

        for g_idx, (g, gl, gc) in enumerate(zip(groups, group_labels, group_colors)):
            subset = [r for r in data[g]["results"] if r["country"] in country_set]
            n = len(subset)
            if n == 0:
                continue
            correct = sum(1 for r in subset if r["correct"])
            acc = correct / n
            f1s = []
            for ck in class_keys:
                actual = [r for r in subset if r["actual_direction"] == ck]
                pred_pos = sum(1 for r in subset if r["predicted_direction"] == ck and r["correct"])
                total_pred = sum(1 for r in subset if r["predicted_direction"] == ck)
                if len(actual) == 0 or total_pred == 0:
                    f1s.append(0)
                else:
                    p = pred_pos / max(total_pred, 1)
                    r = pred_pos / max(len(actual), 1)
                    f1s.append(2 * p * r / max(p + r, 0.001))

            bars = ax.bar(x + (g_idx - 1) * w, f1s, w, label=gl, color=gc, edgecolor="white", linewidth=0.8)
            for bar, val in zip(bars, f1s):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                            f"{val:.2f}", ha="center", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=10)
        ax.set_title(f"{title}\n({n} samples, acc={acc:.1%})", fontsize=11, color=c_color, fontweight="bold")
        ax.set_ylim(0, 0.85)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if ax_idx == 0:
            ax.set_ylabel("F1 Score")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("High vs Low Incidence Country Prediction Performance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "fig2_high_vs_low_incidence.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[2/3] {fig_path}")

    # ── Fig 3: Module Ablation Effect ──
    fig = plt.figure(figsize=(14, 5.5))
    ax1 = fig.add_subplot(1, 2, 1)
    baseline = 0.333
    base_acc = data["base"]["metrics"]["accuracy"]
    lora_delta = data["lora"]["metrics"]["accuracy"] - base_acc
    rag_delta = data["lora_rag"]["metrics"]["accuracy"] - data["lora"]["metrics"]["accuracy"]

    ax1.bar(0, baseline, 0.5, color="#e5e7eb", label="Random Baseline (33.3%)")
    ax1.bar(0, base_acc - baseline, 0.5, bottom=baseline, color=group_colors[0], label=group_labels[0])
    ax1.bar(1, base_acc, 0.5, color=group_colors[0])
    ax1.bar(1, lora_delta, 0.5, bottom=base_acc, color=group_colors[1], label=group_labels[1], edgecolor="white", linewidth=1.5, hatch="//")
    ax1.bar(2, data["lora"]["metrics"]["accuracy"], 0.5, color=group_colors[1])
    ax1.bar(2, rag_delta, 0.5, bottom=data["lora"]["metrics"]["accuracy"], color=group_colors[2], label=group_labels[2], edgecolor="white", linewidth=1.5, hatch="//")
    ax1.axhline(baseline, color="#ef4444", linestyle="--", linewidth=1.2)

    for i, (label, acc, top) in enumerate([
        ("Random\nBaseline", baseline, baseline),
        ("+ Base\nModel", base_acc, base_acc),
        ("+ LoRA\n+RAG", data["lora_rag"]["metrics"]["accuracy"], data["lora_rag"]["metrics"]["accuracy"]),
    ]):
        ax1.text(i, top + 0.01, f"{acc:.1%}", ha="center", fontsize=11, fontweight="bold")

    ax1.text(1, base_acc + lora_delta/2, f"{lora_delta:+.1%}", ha="center", fontsize=9, color="white", fontweight="bold")
    ax1.text(2, data["lora"]["metrics"]["accuracy"] + rag_delta/2, f"{rag_delta:+.1%}", ha="center", fontsize=9, color="white", fontweight="bold")

    ax1.set_xticks([0, 1, 2]); ax1.set_xticklabels(["Baseline", "Base", "+LoRA+RAG"])
    ax1.set_ylabel("Accuracy"); ax1.set_title("Cumulative Accuracy Gain", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 0.55)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Right panel: per-class F1 + accuracy progression line chart
    ax2 = fig.add_subplot(1, 2, 2)
    x_pos = [0, 1, 2]
    class_line_colors = ["#ef4444", "#3b82f6", "#f59e0b"]
    class_line_markers = ["o", "s", "^"]

    for cls_name, ck, lc, mk in zip(classes, class_keys, class_line_colors, class_line_markers):
        f1_vals = [data[g]["metrics"]["class_metrics"][ck]["f1"] for g in groups]
        ax2.plot(x_pos, f1_vals, color=lc, marker=mk, linewidth=2, markersize=9,
                label=f"{cls_name} (F1)", markeredgecolor="white", markeredgewidth=0.8)

    # Accuracy line (dual y-axis)
    ax2_acc = ax2.twinx()
    acc_vals = [data[g]["metrics"]["accuracy"] for g in groups]
    ax2_acc.plot(x_pos, acc_vals, color="#166534", marker="D", linewidth=2.5, markersize=10,
                 linestyle="--", label="Accuracy", markeredgecolor="white", markeredgewidth=0.8)

    ax2.axhline(0.333, color="#9ca3af", linestyle=":", linewidth=1, alpha=0.7)
    ax2.text(2.05, 0.333, "Random\nbaseline", fontsize=8, color="#9ca3af", va="center")

    ax2.set_xticks(x_pos); ax2.set_xticklabels(group_labels)
    ax2.set_ylabel("F1 Score"); ax2_acc.set_ylabel("Accuracy")
    ax2.set_title("Per-Class F1 & Accuracy Progression", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 0.62); ax2_acc.set_ylim(0, 0.55)

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_acc.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8, framealpha=0.9)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "fig3_ablation_effect.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[3/3] {fig_path}")

    print(f"\n3 figures saved to {output_dir}/ (300 DPI)")
    return [os.path.join(output_dir, f"fig{i}_") for i in range(1, 4)]


if __name__ == "__main__":
    os.chdir("D:/Ebola")
    print("Backtesting module loaded.")
