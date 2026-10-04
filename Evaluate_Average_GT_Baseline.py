"""Evaluate a training-only average-GT spatial prior on validation data."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import Tree_Project as baseline


RESULTS_DIR = Path(__file__).resolve().parent / "models" / "results"
AVERAGE_MAP_PATH = RESULTS_DIR / "average_training_gt_probability_map.png"
VALIDATION_EXAMPLES_PATH = RESULTS_DIR / "average_gt_validation_examples.png"


def find_summer_gt_files(dataset_dir):
    """Index Summer GT paths only, without scanning or loading other modalities."""
    gt_paths = {}
    for path in sorted(dataset_dir.glob("*.png")):
        match = baseline.IMAGE_PATTERN.match(path.name)
        if match is None:
            continue
        parts = match.groupdict()
        if parts["season"] != "Summer" or parts["modality"] != "GT":
            continue

        sample_key = (parts["tree_id"], parts["season"], parts["tag"])
        if sample_key in gt_paths:
            raise ValueError(f"Duplicate Summer GT sample key: {sample_key}")
        gt_paths[sample_key] = path
    return gt_paths


def load_ground_truth_masks(sample_keys, gt_paths):
    """Load only the requested GT masks and return binary float32 arrays."""
    masks = []
    for sample_key in sample_keys:
        mask = cv2.imread(str(gt_paths[sample_key]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read GT mask: {gt_paths[sample_key]}")
        if mask.shape != baseline.TARGET_SIZE:
            raise ValueError(
                f"Unexpected GT dimensions for {gt_paths[sample_key].name}: "
                f"{mask.shape}"
            )
        if not np.isin(np.unique(mask), [0, 255]).all():
            raise ValueError(f"GT mask is not binary (0/255): {gt_paths[sample_key].name}")
        masks.append((mask == 255).astype(np.float32))
    return np.stack(masks, axis=0)[..., np.newaxis]


def display_and_save_average_map(average_gt):
    """Display and save the training-only pixel-wise mean GT map."""
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(average_gt, cmap="viridis", vmin=0, vmax=1)
    axis.set_title("Mean Training GT Probability Map")
    axis.axis("off")
    figure.colorbar(image, ax=axis, label="Training-mask frequency")
    figure.tight_layout()
    figure.savefig(AVERAGE_MAP_PATH, dpi=160, bbox_inches="tight")
    print(f"Average training GT map saved to: {AVERAGE_MAP_PATH}")


def display_validation_examples(validation_masks, average_gt, threshold, sample_keys):
    """Compare validation GTs with the same thresholded training prior."""
    count = min(4, len(validation_masks))
    if count == 0:
        raise ValueError("The validation split is empty.")

    selected = np.linspace(0, len(validation_masks) - 1, count, dtype=int)
    average_prediction = average_gt >= threshold
    figure, axes = plt.subplots(count, 2, figsize=(8, 3.5 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, index in enumerate(selected):
        axes[row, 0].imshow(
            validation_masks[index, :, :, 0], cmap="gray", vmin=0, vmax=1
        )
        axes[row, 0].set_title(f"Validation GT: {sample_keys[index][0]}")
        axes[row, 1].imshow(average_prediction, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Same average prediction ({threshold:.2f})")
        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle("Validation masks versus the fixed training spatial prior")
    figure.tight_layout()
    figure.savefig(VALIDATION_EXAMPLES_PATH, dpi=160, bbox_inches="tight")
    print(f"Validation comparison saved to: {VALIDATION_EXAMPLES_PATH}")


def main():
    """Build a training-only prior and select its threshold on validation GTs."""
    if not baseline.DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {baseline.DATASET_DIR}")

    gt_paths = find_summer_gt_files(baseline.DATASET_DIR)
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
    validation_keys = splits["validation"]["sample_keys"]
    print("GT-only spatial-prior baseline; RGB and depth are not loaded.")
    print(f"Seed: {baseline.RANDOM_SEED}")
    print(
        "Tree-level split sizes: "
        f"train={len(train_keys)}, "
        f"validation={len(validation_keys)}, "
        f"test={len(splits['test']['indices'])} (not loaded or evaluated)"
    )

    training_masks = load_ground_truth_masks(train_keys, gt_paths)
    average_gt = np.mean(training_masks, axis=0, dtype=np.float32)[:, :, 0]
    del training_masks
    display_and_save_average_map(average_gt)

    validation_masks = load_ground_truth_masks(validation_keys, gt_paths)
    repeated_probabilities = np.broadcast_to(
        average_gt[np.newaxis, :, :, np.newaxis], validation_masks.shape
    )
    results = baseline.score_validation_thresholds(
        validation_masks,
        repeated_probabilities,
        baseline.VALIDATION_THRESHOLDS,
    )
    selected = max(results, key=lambda result: result["dice"])

    print("\nValidation threshold results (mean per-image binary scores)")
    for result in results:
        print(
            f"Threshold {result['threshold']:.2f}: "
            f"Dice {result['dice']:.6f}, IoU {result['iou']:.6f}"
        )
    print("\nSelected validation result")
    print(f"Selected threshold: {selected['threshold']:.2f}")
    print(f"Validation binary Dice: {selected['dice']:.6f}")
    print(f"Validation binary IoU: {selected['iou']:.6f}")
    print("Held-out test masks were not loaded or evaluated.")

    display_validation_examples(
        validation_masks, average_gt, selected["threshold"], validation_keys
    )
    plt.show()


if __name__ == "__main__":
    main()