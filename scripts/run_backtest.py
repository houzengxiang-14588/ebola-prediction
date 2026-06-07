"""Run trend-classification backtest for Ebola LoRA model.

Usage: python scripts/run_backtest.py [--model base|lora]
"""

import sys, os, argparse
sys.path.insert(0, "D:/Ebola")
os.chdir("D:/Ebola")

from src.eval.backtest import run_backtest, generate_backtest_report
from src.model.lora_model import LoraModelClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "lora"], default="lora",
                       help="base=pretrained only, lora=LoRA fine-tuned")
    args = parser.parse_args()

    print(f"Backtest mode: {args.model}")
    print("=" * 50)

    lora_path = "lora_weights/final" if args.model == "lora" else None

    client = LoraModelClient(
        model_path="D:/llm_models/qwen/Qwen2-7B-Instruct",
        lora_path=lora_path,
        temperature=0.3,
        max_tokens=256,
        load_in_4bit=True,
    )

    print("\nRunning trend classification backtest (20 African countries)...")
    result = run_backtest(client)

    print("\n" + "=" * 50)
    print("Backtest Results")
    print("=" * 50)
    m = result["metrics"]
    print(f"  Accuracy:          {m['accuracy']:.1%}")
    print(f"  vs Random Baseline: +{m['accuracy'] - m['random_baseline']:.1%}")
    print(f"  High-Conf Acc:     {m['high_conf_accuracy']:.1%} (n={m['high_conf_total']})")
    for cls, cm in m["class_metrics"].items():
        print(f"  {cls}: P={cm['precision']:.1%} R={cm['recall']:.1%} F1={cm['f1']:.1%}")

    print("\nGenerating report...")
    figure_path, text_path = generate_backtest_report(
        result, f"data/backtest_{args.model}.png"
    )
    print(f"  Figure: {figure_path}")
    print(f"  Report: {text_path}")

    import json
    json_path = f"data/backtest_{args.model}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  JSON:   {json_path}")


if __name__ == "__main__":
    main()
