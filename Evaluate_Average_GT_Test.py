"""Evaluate the fixed training-only average-GT prior on the held-out test set."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import Evaluate_Average_GT_Baseline as average_baseline
import Tree_Project as baseline


TEST_THRESHOLD = 0.20
RESULTS_DIR = Path(__file__).resolve().parent / "models" / "results"
TEST_EXAMPLES_PATH = RESULTS_DIR / "average_gt_test_examples.png"
NEURAL_NETWORK_RESULTS = (
    ("RGB-only BCE", "RGB", "BCE", 0.25, 0.350560, 0.214901),
    ("RGB-only BCE + Dice", "RGB", "BCE + Dice", 0.30, 0.372018, 0.231315),
    ("RGB-D BCE + Dice", "RGB + depth", "BCE + Dice", 0.35, 0.371957, 0.231123),
)


def display_test_examples(test_masks, average_gt, sample_keys):
    """Display test masks alongside the fixed prior and its binary prediction."""
    count = min(4, len(test_masks))
    if count == 0:
        raise ValueError("The held-out test split is empty.")

    selected = np.linspace(0, len(test_masks) - 1, count, dtype=int)
    binary_prediction = average_gt >= TEST_THRESHOLD
    figure, axes = plt.subplots(count, 3, figsize=(12, 3.5 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, index in enumerate(selected):
        axes[row, 0].imshow(test_masks[index, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Test GT: {sample_keys[index][0]}")
        axes[row, 1].imshow(average_gt, cmap="viridis", vmin=0, vmax=1)
        axes[row, 1].set_title("Average training-GT probability")
        axes[row, 2].imshow(binary_prediction, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Fixed prediction (threshold {TEST_THRESHOLD:.2f})")
        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle("Held-out test masks versus the training-only spatial prior")
    figure.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(TEST_EXAMPLES_PATH, dpi=160, bbox_inches="tight")
    print(f"Test examples saved to: {TEST_EXAMPLES_PATH}")


def print_comparison_table(average_dice, average_iou):
    """Print this evaluation beside the already-established network results."""
    rows = [
        (
            "Average-GT spatial prior",
            "None",
            "Training GT mean",
            TEST_THRESHOLD,
            average_dice,
            average_iou,
        ),
        *NEURAL_NETWORK_RESULTS,
    ]
    headers = ("Model", "Input", "Loss/Method", "Threshold", "Test Dice", "Test IoU")
    print("\nFinal held-out test comparison")
    print(
        f"{headers[0]:<27} {headers[1]:<13} {headers[2]:<18} "
        f"{headers[3]:>9} {headers[4]:>12} {headers[5]:>12}"
    )
    for model, model_input, method, threshold, dice, iou in rows:
        print(
            f"{model:<27} {model_input:<13} {method:<18} "
            f"{threshold:>9.2f} {dice:>12.6f} {iou:>12.6f}"
        )


def main():
    """Reconstruct the train-only prior and score it once on test GT masks."""
    if not baseline.DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {baseline.DATASET_DIR}")

    gt_paths = average_baseline.find_summer_gt_files(baseline.DATASET_DIR)
    sample_keys = sorted(gt_paths)
    if not sample_keys:
        raise RuntimeError("No Summer GT masks were found.")

    tree_ids = np.asarray([key[0] for key in sample_keys])
    splits = baseline.split_by_tree(
        tree_ids, sample_keys, seed=baseline.RANDOM_SEED
    )
    baseline.verify_tree_split_isolation(splits)
    expected_counts = {"train": 393, "validation": 84, "test": 85}
    for split_name, expected_count in expected_counts.items():
        actual_count = len(splits[split_name]["indices"])
        if actual_count != expected_count:
            raise AssertionError(
                f"Expected {expected_count} {split_name} samples, got {actual_count}."
            )

    train_keys = splits["train"]["sample_keys"]
    test_keys = splits["test"]["sample_keys"]
    print(f"Seed: {baseline.RANDOM_SEED}")
    print(f"Training GT masks used to reconstruct average: {len(train_keys)}")
    print(f"Held-out test trees: {len(splits['test']['tree_ids'])}")

    training_masks = average_baseline.load_ground_truth_masks(train_keys, gt_paths)
    average_gt = np.mean(training_masks, axis=0, dtype=np.float32)[:, :, 0]
    del training_masks

    test_masks = average_baseline.load_ground_truth_masks(test_keys, gt_paths)
    if len(test_masks) != 85:
        raise AssertionError(f"Expected 85 test masks, got {len(test_masks)}.")

    prediction_probabilities = np.broadcast_to(
        average_gt[np.newaxis, :, :, np.newaxis], test_masks.shape
    )
    binary_predictions = prediction_probabilities >= TEST_THRESHOLD
    average_dice = baseline.binary_dice_coefficient(test_masks, binary_predictions)
    average_iou = baseline.binary_intersection_over_union(test_masks, binary_predictions)

    print("\nFinal Average-GT spatial-prior test performance")
    print(f"Fixed validation-selected threshold: {TEST_THRESHOLD:.2f}")
    print(f"Binary Dice: {average_dice:.6f}")
    print(f"Binary IoU: {average_iou:.6f}")
    print_comparison_table(average_dice, average_iou)

    display_test_examples(test_masks, average_gt, test_keys)
    plt.show()


if __name__ == "__main__":
    main()