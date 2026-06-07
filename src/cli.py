"""命令行入口 — 数据采集 / 预测。"""

import argparse
import sys
from pathlib import Path

import yaml
import pandas as pd

from .collector import fetch_data, list_sources
from .processor import clean_series
from .model import LLMClient
from .predictor import roll_predict


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_fetch(args):
    """拉取历史疫情数据。"""
    df = fetch_data(source=args.source, region=args.region)
    out = args.output or f"data/raw/{args.source}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"已保存 {len(df)} 条记录到 {out}")


def cmd_sources(_args):
    """列出可用数据源。"""
    for s in list_sources():
        print(f"  {s['name']:20s} — {s['description']}")


def cmd_predict(args):
    """运行填空式预测。"""
    config = load_config()
    llm_cfg = config["llm"]
    pred_cfg = config["predictor"]
    proc_cfg = config["processor"]

    # 加载数据
    df = pd.read_csv(args.data, parse_dates=["date"])
    df = clean_series(
        df,
        min_length=proc_cfg["min_sequence_length"],
        smooth_window=proc_cfg["smooth_window"],
        fill_method=proc_cfg["fill_method"],
    )
    print(f"数据加载完成，共 {len(df)} 天 ({df['date'].min()} ~ {df['date'].max()})")

    # 初始化 LLM
    client = LLMClient(
        model_path=llm_cfg["model_path"],
        device=llm_cfg.get("device", "auto"),
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
    )

    # 预测
    steps = args.steps or pred_cfg["roll_steps"]
    predictions = roll_predict(
        client=client,
        df=df,
        context_days=pred_cfg["context_days"],
        mask_days=pred_cfg["mask_days"],
        roll_steps=steps,
        samples=pred_cfg["samples"],
    )

    # 输出
    print(f"\n{'='*50}")
    print(f"  预测结果（{len(predictions)} 天）")
    print(f"{'='*50}")
    for p in predictions:
        print(p)


def main():
    parser = argparse.ArgumentParser(prog="ebola", description="疫情填空式预测工具")
    sub = parser.add_subparsers(dest="command")

    p_fetch = sub.add_parser("fetch", help="拉取历史数据")
    p_fetch.add_argument("--source", default="owid-covid")
    p_fetch.add_argument("--region", default=None)
    p_fetch.add_argument("--output", default=None)

    p_sources = sub.add_parser("sources", help="列出可用数据源")

    p_pred = sub.add_parser("predict", help="运行预测")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--steps", type=int, default=None)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    commands = {
        "fetch": cmd_fetch,
        "sources": cmd_sources,
        "predict": cmd_predict,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
