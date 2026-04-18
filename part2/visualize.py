"""
visualize.py — Member B: Result Visualization
DSS5104 Text Classification Assignment

Reads JSON files from results/ and generates:
  1. Data efficiency learning curves (macro-F1 vs training set size)
  2. Model comparison tables (for report)

Usage:
  python visualize.py --result_file results/member_b_agnews.json
  python visualize.py --result_file results/member_b_imdb.json
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")           # Save figures without GUI
import matplotlib.pyplot as plt


FRACTIONS     = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
FRACTION_PCTS = [f * 100 for f in FRACTIONS]      # x-axis labels

MODEL_COLORS = {
    "fasttext": "#e74c3c",
    "textcnn":  "#3498db",
    "bilstm":   "#2ecc71",
}
MODEL_MARKERS = {
    "fasttext": "o",
    "textcnn":  "s",
    "bilstm":   "^",
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_efficiency_curve(data: dict, dataset_name: str, save_path: str):
    """
    data: {model_name: {fraction_str: {f1_mean, f1_std, ...}}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, eff in data.items():
        y_means, y_stds = [], []
        valid_fracs = []

        for frac in FRACTIONS:
            key = str(frac)
            if key in eff and eff[key]:
                y_means.append(eff[key]["f1_mean"])
                y_stds.append(eff[key]["f1_std"])
                valid_fracs.append(frac * 100)

        y_means = np.array(y_means)
        y_stds  = np.array(y_stds)

        color  = MODEL_COLORS.get(model_name, "gray")
        marker = MODEL_MARKERS.get(model_name, "x")

        ax.plot(valid_fracs, y_means,
                color=color, marker=marker, linewidth=2, markersize=7,
                label=model_name.upper())
        ax.fill_between(valid_fracs,
                         y_means - y_stds,
                         y_means + y_stds,
                         alpha=0.15, color=color)

    ax.set_xscale("log")
    ax.set_xticks(FRACTION_PCTS)
    ax.set_xticklabels([f"{p:.0f}%" for p in FRACTION_PCTS])
    ax.set_xlabel("Training Set Size (%)", fontsize=12)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_title(f"Data Efficiency — {dataset_name.upper()}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot saved] {save_path}")


def print_full_data_table(full_data: dict):
    """Print full data results table in Markdown format"""
    print("\n### Full Data Results\n")
    print(f"| Model     | Accuracy (mean±std) | Macro-F1 (mean±std) | Train Time (s) | Infer Time (s) |")
    print(f"|-----------|---------------------|---------------------|----------------|----------------|")
    for model, s in full_data.items():
        print(f"| {model.upper():<9} | "
              f"{s['acc_mean']:.4f}±{s['acc_std']:.4f}      | "
              f"{s['f1_mean']:.4f}±{s['f1_std']:.4f}      | "
              f"{s['train_time_mean']:>14.1f} | "
              f"{s['infer_time_mean']:>14.3f} |")


def print_efficiency_markdown(efficiency: dict):
    """Print data efficiency Markdown table"""
    models = list(efficiency.keys())
    print("\n### Data Efficiency (Macro-F1)\n")
    header = "| Fraction |" + "".join(f" {m.upper()} |" for m in models)
    sep    = "|----------|" + "".join("------------|" for _ in models)
    print(header)
    print(sep)

    for frac in FRACTIONS:
        key = str(frac)
        row = f"| {frac*100:>7.0f}% |"
        for m in models:
            s = efficiency[m].get(key, {})
            if s:
                row += f" {s['f1_mean']:.4f}±{s['f1_std']:.4f} |"
            else:
                row += "    N/A    |"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", required=True)
    args = parser.parse_args()

    results = load_results(args.result_file)
    dataset = results["dataset"]

    # 1. Print full data table
    print_full_data_table(results["full_data"])

    # 2. Print data efficiency table
    print_efficiency_markdown(results["efficiency"])

    # 3. Plot learning curves
    plot_path = args.result_file.replace(".json", "_curve.png")
    plot_efficiency_curve(results["efficiency"], dataset, plot_path)


if __name__ == "__main__":
    main()
