"""Measure foreground area in fixed-threshold validation predictions."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import Tree_Project as baseline
import Tree_Project_BCE_Dice as rgb_experiment
import Tree_Project_RGBD_BCE_Dice as rgbd_experiment


RESULTS_DIR = Path(__file__).resolve().parent / "models" / "results"
CSV_PATH = RESULTS_DIR / "validation_foreground_fraction_analysis.csv"
RGB_THRESHOLD = 0.30
RGBD_THRESHOLD = 0.35
EXPECTED_VALIDATION_SAMPLES = 84


def predict_binary_masks(model_path, inputs, threshold, label):
    """Load a saved model for inference only and apply its fixed threshold."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint for {label} not found: {model_path}")
    print(f"Loading {label} checkpoint for inference only: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    probabilities = model.predict(
        inputs, batch_size=baseline.BATCH_SIZE, verbose=1
    )
    if probabilities.shape[1:] != (256, 256, 1):
        raise ValueError(
            f"Unexpected prediction shape for {label}: {probabilities.shape}"
        )
    return probabilities >= threshold


def summarize_fraction(label, values):
    """Print distribution statistics as fractions and percentages."""
    statistics = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "standard deviation": float(np.std(values)),
    }
    print(f"\n{label} foreground fraction")
    for statistic, fraction in statistics.items():
        print(
            f"{statistic.capitalize()}: {fraction:.6f} "
            f"({100 * fraction:.3f}%)"
        )


def summarize_ratios(label, ratios):
    """Print ratio location statistics and over-prediction frequencies."""
    sample_count = len(ratios)
    print(f"\n{label} foreground area ratio")
    print(f"Mean: {np.mean(ratios):.6f}")
    print(f"Median: {np.median(ratios):.6f}")
    for description, condition in (
        ("ratio > 1", ratios > 1.0),
        ("ratio < 1", ratios < 1.0),
        ("ratio > 1.5", ratios > 1.5),
        ("ratio > 2.0", ratios > 2.0),
    ):
        count = int(np.count_nonzero(condition))
        print(f"{description}: {count}/{sample_count} ({100 * count / sample_count:.1f}%)")


def save_csv(sample_keys, gt_fractions, rgb_fractions, rgbd_fractions,
             rgb_ratios, rgbd_ratios):
    """Save all per-validation-image foreground fractions and ratios."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "tree_id",
        "season",
        "sample_id",
        "gt_foreground_fraction",
        "rgb_foreground_fraction",
        "rgbd_foreground_fraction",
        "rgb_gt_foreground_ratio",
        "rgbd_gt_foreground_ratio",
    )
    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, key in enumerate(sample_keys):
            writer.writerow(
                {
                    "tree_id": key[0],
                    "season": key[1],
                    "sample_id": key[2],
                    "gt_foreground_fraction": f"{gt_fractions[index]:.8f}",
                    "rgb_foreground_fraction": f"{rgb_fractions[index]:.8f}",
                    "rgbd_foreground_fraction": f"{rgbd_fractions[index]:.8f}",
                    "rgb_gt_foreground_ratio": f"{rgb_ratios[index]:.8f}",
                    "rgbd_gt_foreground_ratio": f"{rgbd_ratios[index]:.8f}",
                }
            )
    print(f"Per-image results saved to: {CSV_PATH}")


def save_boxplot(gt_fractions, rgb_fractions, rgbd_fractions):
    """Save a box plot comparing the three validation foreground fractions."""
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.boxplot(
        (gt_fractions * 100, rgb_fractions * 100, rgbd_fractions * 100),
        tick_labels=("Ground truth", "RGB BCE + Dice", "RGB-D BCE + Dice"),
        showmeans=True,
    )
    axis.set_ylabel("Foreground pixels (%)")
    axis.set_title("Validation foreground fraction by method")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "validation_foreground_fraction_boxplot.png", dpi=160)


def save_scatter_plot(gt_fractions, prediction_fractions, model_label, filename):
    """Save a GT-versus-predicted fraction scatter plot and equality line."""
    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.scatter(
        gt_fractions * 100,
        prediction_fractions * 100,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
    )
    axis.plot((0, 100), (0, 100), color="#bd3b32", linestyle="--", label="y = x")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("GT foreground (%)")
    axis.set_ylabel("Predicted foreground (%)")
    axis.set_title(f"GT vs predicted foreground fraction: {model_label}")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / filename, dpi=160)


def save_extreme_examples(category, indices, sample_keys, rgb_inputs, rgbd_inputs,
                          gt_masks, rgb_predictions, rgbd_predictions,
                          gt_fractions, rgb_fractions, rgbd_fractions,
                          rgb_ratios, rgbd_ratios):
    """Save the four highest-ratio samples with both models' predicted masks."""
    figure, axes = plt.subplots(4, 5, figsize=(16, 13))
    column_titles = ("RGB", "Depth", "Ground truth", "RGB prediction", "RGB-D prediction")

    for row, index in enumerate(indices):
        key = sample_keys[index]
        axes[row, 0].imshow(rgb_inputs[index])
        axes[row, 1].imshow(rgbd_inputs[index, :, :, 3], cmap="viridis", vmin=0, vmax=1)
        axes[row, 2].imshow(gt_masks[index, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(rgb_predictions[index, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 4].imshow(rgbd_predictions[index, :, :, 0], cmap="gray", vmin=0, vmax=1)

        if row == 0:
            for column, title in enumerate(column_titles):
                axes[row, column].set_title(title)
        axes[row, 0].set_ylabel(
            f"{key[0]} / {key[2]}\nGT {gt_fractions[index] * 100:.2f}%",
            fontsize=8,
        )
        axes[row, 3].set_xlabel(
            f"{rgb_fractions[index] * 100:.2f}% predicted\n"
            f"ratio {rgb_ratios[index]:.2f}"
        )
        axes[row, 4].set_xlabel(
            f"{rgbd_fractions[index] * 100:.2f}% predicted\n"
            f"ratio {rgbd_ratios[index]:.2f}"
        )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(category.replace("_", " ").title())
    figure.tight_layout(rect=(0.03, 0, 1, 0.97))
    figure.savefig(RESULTS_DIR / f"{category}.png", dpi=160, bbox_inches="tight")


def main():
    """Analyze foreground area for the validation split only."""
    if not baseline.DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {baseline.DATASET_DIR}")

    images = baseline.scan_images(baseline.DATASET_DIR)
    summer_pairs = baseline.find_summer_rgb_gt_pairs(images)
    triplets, unmatched = baseline.find_summer_triplets(images)
    if unmatched:
        raise RuntimeError(f"Incomplete Summer RGB-D-GT groups: {unmatched}")
    if set(summer_pairs) != set(triplets):
        raise ValueError("Summer RGB/GT pairs and RGB-D-GT triplets do not match.")

    sample_keys = sorted(summer_pairs)
    tree_ids = np.asarray([key[0] for key in sample_keys])
    splits = baseline.split_by_tree(
        tree_ids, sample_keys, seed=baseline.RANDOM_SEED
    )
    baseline.verify_tree_split_isolation(splits)
    validation_keys = splits["validation"]["sample_keys"]
    if len(validation_keys) != EXPECTED_VALIDATION_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_VALIDATION_SAMPLES} validation samples, "
            f"got {len(validation_keys)}."
        )

    validation_pairs = {key: summer_pairs[key] for key in validation_keys}
    validation_triplets = {key: triplets[key] for key in validation_keys}
    rgb_inputs, gt_masks, _, rgb_keys = baseline.prepare_rgb_gt_dataset(validation_pairs)
    rgbd_inputs, rgbd_gt_masks, _, rgbd_keys = rgbd_experiment.prepare_rgbd_dataset(
        validation_triplets
    )
    if rgb_keys != validation_keys or rgbd_keys != validation_keys:
        raise AssertionError("Prepared validation sample order differs from the split.")
    if not np.array_equal(gt_masks, rgbd_gt_masks):
        raise AssertionError("RGB and RGB-D preprocessing produced different GT masks.")

    rgb_predictions = predict_binary_masks(
        rgb_experiment.MODEL_PATH, rgb_inputs, RGB_THRESHOLD, "RGB BCE + Dice"
    )
    rgbd_predictions = predict_binary_masks(
        rgbd_experiment.MODEL_PATH, rgbd_inputs, RGBD_THRESHOLD, "RGB-D BCE + Dice"
    )

    pixel_count = np.prod(gt_masks.shape[1:])
    gt_fractions = np.sum(gt_masks >= 0.5, axis=(1, 2, 3)) / pixel_count
    rgb_fractions = np.sum(rgb_predictions, axis=(1, 2, 3)) / pixel_count
    rgbd_fractions = np.sum(rgbd_predictions, axis=(1, 2, 3)) / pixel_count
    gt_pixel_counts = np.sum(gt_masks >= 0.5, axis=(1, 2, 3))
    if np.any(gt_pixel_counts == 0):
        raise ValueError("A validation GT mask has no foreground pixels; area ratio undefined.")
    rgb_pixel_counts = np.sum(rgb_predictions, axis=(1, 2, 3))
    rgbd_pixel_counts = np.sum(rgbd_predictions, axis=(1, 2, 3))
    rgb_ratios = rgb_pixel_counts / gt_pixel_counts
    rgbd_ratios = rgbd_pixel_counts / gt_pixel_counts

    print(f"Validation samples: {len(validation_keys)}")
    print(f"Fixed thresholds: RGB={RGB_THRESHOLD:.2f}, RGB-D={RGBD_THRESHOLD:.2f}")
    summarize_fraction("Ground truth", gt_fractions)
    summarize_fraction("RGB BCE + Dice prediction", rgb_fractions)
    summarize_fraction("RGB-D BCE + Dice prediction", rgbd_fractions)
    summarize_ratios("RGB BCE + Dice", rgb_ratios)
    summarize_ratios("RGB-D BCE + Dice", rgbd_ratios)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(
        validation_keys,
        gt_fractions,
        rgb_fractions,
        rgbd_fractions,
        rgb_ratios,
        rgbd_ratios,
    )
    save_boxplot(gt_fractions, rgb_fractions, rgbd_fractions)
    save_scatter_plot(
        gt_fractions,
        rgb_fractions,
        "RGB BCE + Dice",
        "validation_gt_vs_rgb_foreground_fraction.png",
    )
    save_scatter_plot(
        gt_fractions,
        rgbd_fractions,
        "RGB-D BCE + Dice",
        "validation_gt_vs_rgbd_foreground_fraction.png",
    )
    save_extreme_examples(
        "validation_largest_rgb_foreground_ratios",
        np.argsort(rgb_ratios)[-4:][::-1],
        validation_keys,
        rgb_inputs,
        rgbd_inputs,
        gt_masks,
        rgb_predictions,
        rgbd_predictions,
        gt_fractions,
        rgb_fractions,
        rgbd_fractions,
        rgb_ratios,
        rgbd_ratios,
    )
    save_extreme_examples(
        "validation_largest_rgbd_foreground_ratios",
        np.argsort(rgbd_ratios)[-4:][::-1],
        validation_keys,
        rgb_inputs,
        rgbd_inputs,
        gt_masks,
        rgb_predictions,
        rgbd_predictions,
        gt_fractions,
        rgb_fractions,
        rgbd_fractions,
        rgb_ratios,
        rgbd_ratios,
    )
    print(f"\nFigures saved under: {RESULTS_DIR}")
    print("No training or test-set analysis was performed.")
    plt.show()


if __name__ == "__main__":
    main()