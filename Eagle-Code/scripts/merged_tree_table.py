"""Compute and print mean accepted length (tau) for merged-tree dual-head eval.

Reads the 4 benchmark JSONL files produced by gen_merged_tree_eval.py,
computes tau = total_new_tokens / total_decode_steps per benchmark, prints
a formatted table, and saves it to results/merged/acceptance_table.txt.

Usage:
    python scripts/merged_tree_table.py \
        --results-dir results/merged \
        --model-id merged_tree_eagle2
"""

import argparse
import json
import os
from pathlib import Path

BENCHES = ["mt_bench", "gsm8k", "math_500", "svamp"]

BASELINE = {
    "MathInstruct (single)": {"mt_bench": 2.541, "gsm8k": 4.923, "math_500": 5.284, "svamp": 4.812},
    "ShareGPT (single)": {"mt_bench": 3.606, "gsm8k": 3.835, "math_500": 3.968, "svamp": 3.778},
}


def compute_tau_and_time(jsonl_path):
    total_tokens, total_steps, total_time = 0, 0, 0
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            for choice in rec.get("choices", []):
                total_tokens += sum(choice.get("new_tokens", []))
                total_steps += sum(choice.get("idxs", []))
                total_time += sum(choice.get("wall_time", []))
    if total_steps == 0:
        return 0.0
    return round(total_tokens / total_steps, 3), round(total_time, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/merged")
    parser.add_argument("--model-id", type=str, default="merged_tree_eagle2")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    taus = {}
    wall_times = {}
    for bench in BENCHES:
        p = results_dir / f"{bench}_{args.model_id}.jsonl"
        if not p.exists():
            taus[bench] = "?"
        else:
            taus[bench], wall_times[bench] = compute_tau_and_time(p)

    col_w = 20
    label_w = 30
    header = f"{'Model':<{label_w}}" + "".join(f"{b:>{col_w}}" for b in BENCHES)
    sep = "-" * len(header)

    lines = [header, sep]
    for head, row in BASELINE.items():
        lines.append(f"{head:<{label_w}}" + "".join(f"{row.get(b, '-'):>{col_w}}" for b in BENCHES))
    lines.append(sep)
    merged_label = f"Merged-tree ({args.model_id})"
    lines.append(f"{merged_label:<{label_w}}" + "".join(f" {taus[b]:>{col_w}}" for b in BENCHES))
    lines.append(sep)
    lines.append(f"{merged_label:<{label_w}}" + "".join(f" {wall_times[b]:>{col_w}}" for b in BENCHES))

    table = "\n".join(lines)
    print(table)

    out_path = results_dir / "acceptance_table.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
