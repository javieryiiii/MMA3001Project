"""Create static comparison tables and plots from established experiment results.

This reporting-only script does not import models, load checkpoints or datasets,
train networks, run inference, or select thresholds.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parent / "models" / "results" / "final_comparison"

VALIDATION_RESULTS = (
    {
        "method": "RGB-only U-Net + BCE",
        "input": "RGB",
        "loss_method": "BCE",
        "validation_threshold": 0.25,
        "validation_dice": 0.354316,
        "validation_iou": 0.217112,
        "validation_foreground_ratio": None,
        "test_status": "Final held-out test evaluated",
    },
    {
        "method": "RGB-only U-Net + BCE + Dice",
        "input": "RGB",
        "loss_method": "BCE + Dice",
        "validation_threshold": 0.30,
        "validation_dice": 0.365778,
        "validation_iou": 0.226002,
        "validation_foreground_ratio": 2.577,
        "test_status": "Final held-out test evaluated",
    },
    {
        "method": "RGB-D U-Net + BCE + Dice",
        "input": "RGB + depth",
        "loss_method": "BCE + Dice",
        "validation_threshold": 0.35,
        "validation_dice": 0.373455,
        "validation_iou": 0.231875,
        "validation_foreground_ratio": 2.666,
        "test_status": "Final held-out test evaluated",
    },
    {
        "method": "Average-GT spatial-prior baseline",
        "input": "None",
        "loss_method": "Mean of 393 training GT masks",
        "validation_threshold": 0.20,
        "validation_dice": 0.379240,
        "validation_iou": 0.238432,
        "validation_foreground_ratio": None,
        "test_status": "Final held-out test evaluated",
    },
    {
        "method": "RGB-only U-Net + Focal + Dice",
        "input": "RGB",
        "loss_method": "Focal + Dice (gamma=2.0)",
        "validation_threshold": 0.45,
        "validation_dice": 0.366954,
        "validation_iou": 0.227173,
        "validation_foreground_ratio": 2.576,
        "test_status": "Validation-only experiment",
    },
    {
        "method": "RGB-only U-Net + Tversky",
        "input": "RGB",
        "loss_method": "Tversky (alpha=0.7, beta=0.3)",
        "validation_threshold": 0.50,
        "validation_dice": 0.140648,
        "validation_iou": 0.075686,
        "validation_foreground_ratio": 13.278258,
        "test_status": "Validation-only experiment",
    },
)

TEST_RESULTS = (
    {
        "method": "RGB-only U-Net + BCE",
        "input": "RGB",
        "loss_method": "BCE",
        "threshold": 0.25,
        "test_dice": 0.350560,
        "test_iou": 0.214901,
    },
    {
        "method": "RGB-only U-Net + BCE + Dice",
        "input": "RGB",
        "loss_method": "BCE + Dice",
        "threshold": 0.30,
        "test_dice": 0.372018,
        "test_iou": 0.231315,
    },
    {
        "method": "RGB-D U-Net + BCE + Dice",
        "input": "RGB + depth",
        "loss_method": "BCE + Dice",
        "threshold": 0.35,
        "test_dice": 0.371957,
        "test_iou": 0.231123,
    },
    {
        "method": "Average-GT spatial-prior baseline",
        "input": "None",
        "loss_method": "Mean of 393 training GT masks",
        "threshold": 0.20,
        "test_dice": 0.379632,
        "test_iou": 0.239717,
    },
)


def write_csv(path, rows, fieldnames):
    """Write supplied comparison rows; unavailable fields remain blank."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metric_plot(rows, metric_key, title, ylabel, output_name):
    """Create a labeled report-quality bar chart with numeric value labels."""
    labels = [row["method"] for row in rows]
    values = [row[metric_key] for row in rows]
    figure_height = max(5.5, 0.75 * len(rows) + 1.5)
    figure, axis = plt.subplots(figsize=(11, figure_height))
    positions = list(range(len(rows)))
    bars = axis.barh(positions, values, color="#3976a8")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel(ylabel)
    axis.set_title(title)
    axis.set_xlim(0, max(values) * 1.18)
    axis.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(
            value + max(values) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.6f}",
            va="center",
        )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / output_name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_foreground_ratio_plot(rows):
    """Plot only validation foreground ratios that were actually reported."""
    available = [
        row for row in rows
        if row["validation_foreground_ratio"] is not None
    ]
    labels = [row["method"] for row in available]
    values = [row["validation_foreground_ratio"] for row in available]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    bars = axis.barh(range(len(values)), values, color="#c27636")
    axis.set_yticks(range(len(values)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Mean predicted / GT foreground area ratio")
    axis.set_title("Validation foreground-area ratio (reported diagnostics only)")
    axis.grid(axis="x", alpha=0.25)
    axis.set_xlim(0, max(values) * 1.2)
    for bar, value in zip(bars, values):
        axis.text(
            value + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
        )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "validation_foreground_ratio.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def print_interpretation():
    """Print the requested factual interpretation of supplied results."""
    print("\nInterpretation")
    print("- BCE + Dice improved over the original RGB-only BCE baseline.")
    print("- Adding depth produced essentially no held-out test improvement over RGB BCE + Dice.")
    print("- Average-GT achieved the highest held-out test Dice and IoU among tested methods.")
    print("- Focal + Dice gave only a very small validation improvement over BCE + Dice and did not correct foreground overprediction.")
    print("- The tested Tversky configuration (alpha=0.7, beta=0.3) failed under the current training setup, converging to a near-all-foreground solution.")
    print("- Available diagnostics show the neural networks produced masks substantially wider than the GT masks.")
    print("- Average-GT's result indicates a strong common spatial prior in this dataset.")
    print("- The networks beat the prior on some individual trees; they have not demonstrated a consistent advantage over it.")
    print("- The RGB/GT spatial transform remains undocumented; visual inspection suggested the current crop/resize assumption was broadly plausible.")


def main():
    """Write comparison tables and figures from constants above; perform no inference."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validation_csv_rows = [
        {
            **row,
            "validation_foreground_ratio": (
                "" if row["validation_foreground_ratio"] is None
                else row["validation_foreground_ratio"]
            ),
        }
        for row in VALIDATION_RESULTS
    ]
    validation_csv_path = OUTPUT_DIR / "validation_model_comparison.csv"
    write_csv(
        validation_csv_path,
        validation_csv_rows,
        (
            "method",
            "input",
            "loss_method",
            "validation_threshold",
            "validation_dice",
            "validation_iou",
            "validation_foreground_ratio",
            "test_status",
        ),
    )

    test_csv_path = OUTPUT_DIR / "test_model_comparison.csv"
    write_csv(
        test_csv_path,
        TEST_RESULTS,
        ("method", "input", "loss_method", "threshold", "test_dice", "test_iou"),
    )

    save_metric_plot(
        VALIDATION_RESULTS,
        "validation_dice",
        "Validation Dice: all six methods",
        "Binary Dice",
        "validation_dice.png",
    )
    save_metric_plot(
        VALIDATION_RESULTS,
        "validation_iou",
        "Validation IoU: all six methods",
        "Binary IoU",
        "validation_iou.png",
    )
    save_metric_plot(
        TEST_RESULTS,
        "test_dice",
        "Held-out test Dice: previously tested methods only",
        "Binary Dice",
        "test_dice.png",
    )
    save_metric_plot(
        TEST_RESULTS,
        "test_iou",
        "Held-out test IoU: previously tested methods only",
        "Binary IoU",
        "test_iou.png",
    )
    save_foreground_ratio_plot(VALIDATION_RESULTS)

    print(f"Validation comparison CSV: {validation_csv_path}")
    print(f"Test comparison CSV: {test_csv_path}")
    print(f"Comparison figures saved under: {OUTPUT_DIR}")
    print("All values are established results supplied for reporting.")
    print("No training, checkpoint access, dataset access, inference or threshold selection was performed.")
    print_interpretation()


if __name__ == "__main__":
    main()