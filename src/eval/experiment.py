# experiment.py - ablation study orchestration
# Imports shared utilities from backtest.py

import os, json, sys
sys.path.insert(0, "D:/Ebola")
os.chdir("D:/Ebola")

from src.eval.backtest import (
    load_country_data, _actual_direction, _incidence_label,
    build_backtest_prompt, parse_prediction, evaluate_sliding_window,
    _compute_metrics, _print_summary
)

def build_rag_context(retriever, country_cases, country_name):
    if retriever is None: return ""
    try:
        daily_cases = [country_cases.get(y, 0) for y in range(2019, 2024)]
        results = retriever.retrieve_for_prediction(
            daily_cases,
            metadata={"total_cases": sum(daily_cases), "max_case": max(daily_cases) if daily_cases else 0}
        )
        lines = ["  - " + item["content"][:200] for item in results.get("knowledge", [])[:3]]
        return "
".join(lines) if lines else ""
    except Exception as e:
        print(f"  RAG skipped: {e}")
        return ""

print("experiment.py loaded")