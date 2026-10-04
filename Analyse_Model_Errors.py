"""Compare fixed-threshold neural predictions with the Average-GT prior."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import Evaluate_Average_GT_Baseline as average_baseline
import Tree_Project as baseline
import Tree_Project_BCE_Dice as rgb_experiment
import Tree_Project_RGBD_BCE_Dice as rgbd_experiment


RESULTS_DIR = Path(__file__).resolve().parent / "models" / "results"
CSV_PATH = RESULTS_DIR / "model_error_analysis_per_image.csv"
RGB_THRESHOLD = 0.30
RGBD_THRESHOLD = 0.35
PRIOR_THRESHOLD = 0.20
EXPECTED_TEST_SAMPLES = 85
METHOD_NAMES = ("Average-GT", "RGB BCE + Dice", "RGB-D BCE + Dice")


def per_image_binary_scores(y_true, y_pred):
    """Return binary Dice and IoU arrays, one score per image."""
    true_masks = np.asarray(y_true) >= 0.5
    predicted_masks = np.asarray(y_pred, dtype=bool)
    if true_masks.shape != predicted_masks.shape:
        raise ValueError("Ground-truth and prediction shapes must match.")

    image_axes = tuple(range(1, true_masks.ndim))
    intersection = np.sum(true_masks & predicted_masks, axis=image_axes)
    true_count = np.sum(true_masks, axis=image_axes)
    predicted_count = np.sum(predicted_masks, axis=image_axes)
    union = np.sum(true_masks | predicted_masks, axis=image_axes)
    dice = (2.0 * intersection + 1e-6) / (true_count + predicted_count + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return dice, iou


def evaluate_checkpoint(model_path, inputs, threshold, label):
    """Load one checkpoint for inference and apply its fixed threshold."""
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


def save_per_image_csv(records):
    """Write all per-sample accuracy and prediction-similarity values."""
    fieldnames = (
        "tree_id",
        "season",
        "sample_id",
        "prior_dice",
        "prior_iou",
        "rgb_dice",
        "rgb_iou",
        "rgb_dice_improvement_over_prior",
        "rgbd_dice",
        "rgbd_iou",
        "rgbd_dice_improvement_over_prior",
        "rgb_prediction_to_prior_similarity",
        "rgbd_prediction_to_prior_similarity",
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Per-image results saved to: {CSV_PATH}")


def save_score_distribution_plot(records):
    """Plot per-image Dice distributions for all three methods."""
    values = [
        [record["prior_dice"] for record in records],
        [record["rgb_dice"] for record in records],
        [record["rgbd_dice"] for record in records],
    ]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0.0, 1.0, 21)
    for method, scores in zip(METHOD_NAMES, values):
        axis.hist(scores, bins=bins, alpha=0.45, label=method)
    axis.set_xlabel("Per-image binary Dice")
    axis.set_ylabel("Number of test trees")
    axis.set_title("Held-out per-image Dice distributions")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "error_analysis_dice_distributions.png", dpi=160)


def save_difference_plot(records):
    """Plot per-tree neural-network Dice improvements over the prior."""
    differences = (
        [record["rgb_dice_improvement_over_prior"] for record in records],
        [record["rgbd_dice_improvement_over_prior"] for record in records],
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, values, title in zip(
        axes,
        differences,
        ("RGB BCE + Dice minus prior", "RGB-D BCE + Dice minus prior"),
    ):
        axis.hist(values, bins=21, color="#3976a8", edgecolor="white")
        axis.axvline(0.0, color="#bd3b32", linestyle="--", linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("Per-tree Dice difference")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Number of test trees")
    figure.suptitle("Positive values mean the neural network beats the prior")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "error_analysis_dice_differences.png", dpi=160)


def save_example_montage(category, indices, records, rgbd_inputs, targets,
                         prior_binary, rgb_binary, rgbd_binary):
    """Save four aligned RGB, depth, GT, prior, RGB and RGB-D examples."""
    column_names = (
        "RGB",
        "Depth",
        "Ground truth",
        "Average-GT prediction",
        "RGB BCE + Dice",
        "RGB-D BCE + Dice",
    )
    figure, axes = plt.subplots(4, 6, figsize=(19, 13))
    for row, sample_index in enumerate(indices):
        record = records[sample_index]
        row_title = (
            f"{record['tree_id']} / {record['sample_id']}\n"
            f"Dice prior/RGB/RGB-D: {record['prior_dice']:.3f} / "
            f"{record['rgb_dice']:.3f} / {record['rgbd_dice']:.3f}"
        )
        panels = (
            (rgbd_inputs[sample_index, :, :, :3], None),
            (rgbd_inputs[sample_index, :, :, 3], "viridis"),
            (targets[sample_index, :, :, 0], "gray"),
            (prior_binary[sample_index, :, :, 0], "gray"),
            (rgb_binary[sample_index, :, :, 0], "gray"),
            (rgbd_binary[sample_index, :, :, 0], "gray"),
        )
        for column, (image, color_map) in enumerate(panels):
            axis = axes[row, column]
            if color_map is None:
                axis.imshow(image)
            elif column == 1:
                axis.imshow(image, cmap=color_map, vmin=0, vmax=1)
            else:
                axis.imshow(image, cmap=color_map, vmin=0, vmax=1)
            if row == 0:
                axis.set_title(column_names[column])
            if column == 0:
                axis.set_ylabel(row_title, fontsize=8)
                axis.yaxis.set_label_coords(-0.08, 0.5)
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(category.replace("_", " ").title())
    figure.tight_layout(rect=(0.04, 0, 1, 0.97))
    path = RESULTS_DIR / f"error_analysis_{category}.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    print(f"Example montage saved to: {path}")


def print_ranked_ids(label, indices, records):
    """Print selected tree/sample IDs and their Dice differences."""
    print(label)
    for index in indices:
        record = records[index]
        delta_key = (
            "rgb_dice_improvement_over_prior"
            if "RGB" in label and "RGB-D" not in label
            else "rgbd_dice_improvement_over_prior"
        )
        print(
            f"  {record['tree_id']} / {record['sample_id']}: "
            f"difference={record[delta_key]:+.6f}, "
            f"prior={record['prior_dice']:.6f}, "
            f"RGB={record['rgb_dice']:.6f}, RGB-D={record['rgbd_dice']:.6f}"
        )


def print_summary(records):
    """Report mean accuracy, paired wins, similarities and extremes."""
    count = len(records)
    print("\nMean per-image test accuracy")
    for label, dice_key, iou_key in (
        ("Average-GT", "prior_dice", "prior_iou"),
        ("RGB BCE + Dice", "rgb_dice", "rgb_iou"),
        ("RGB-D BCE + Dice", "rgbd_dice", "rgbd_iou"),
    ):
        mean_dice = float(np.mean([record[dice_key] for record in records]))
        mean_iou = float(np.mean([record[iou_key] for record in records]))
        print(f"{label}: Dice={mean_dice:.6f}, IoU={mean_iou:.6f}")

    for label, key in (
        ("RGB BCE + Dice", "rgb_dice_improvement_over_prior"),
        ("RGB-D BCE + Dice", "rgbd_dice_improvement_over_prior"),
    ):
        differences = np.asarray([record[key] for record in records])
        wins = int(np.sum(differences > 0))
        prior_wins = int(np.sum(differences < 0))
        ties = count - wins - prior_wins
        positive_index = int(np.argmax(differences))
        negative_index = int(np.argmin(differences))
        print(
            f"\n{label} versus Average-GT: beats prior {wins}/{count} "
            f"({100 * wins / count:.1f}%), prior beats network "
            f"{prior_wins}/{count} ({100 * prior_wins / count:.1f}%), ties={ties}"
        )
        print(f"Mean Dice difference: {np.mean(differences):+.6f}")
        print(
            "Largest positive difference: "
            f"{differences[positive_index]:+.6f} "
            f"({records[positive_index]['tree_id']} / "
            f"{records[positive_index]['sample_id']})"
        )
        print(
            "Largest negative difference: "
            f"{differences[negative_index]:+.6f} "
            f"({records[negative_index]['tree_id']} / "
            f"{records[negative_index]['sample_id']})"
        )

    print("\nPrediction-to-prior similarity (not accuracy against GT)")
    for label, key in (
        ("RGB BCE + Dice", "rgb_prediction_to_prior_similarity"),
        ("RGB-D BCE + Dice", "rgbd_prediction_to_prior_similarity"),
    ):
        similarities = np.asarray([record[key] for record in records])
        print(
            f"{label}: mean={np.mean(similarities):.6f}, "
            f"min={np.min(similarities):.6f}, max={np.max(similarities):.6f}"
        )

    rgb_differences = np.asarray(
        [record["rgb_dice_improvement_over_prior"] for record in records]
    )
    rgbd_differences = np.asarray(
        [record["rgbd_dice_improvement_over_prior"] for record in records]
    )
    print("\nFour strongest RGB BCE + Dice improvements over prior")
    print_ranked_ids(
        "RGB improvements:", np.argsort(rgb_differences)[-4:][::-1], records
    )
    print("Four strongest RGB BCE + Dice failures versus prior")
    print_ranked_ids(
        "RGB failures:", np.argsort(rgb_differences)[:4], records
    )
    print("Four strongest RGB-D BCE + Dice improvements over prior")
    print_ranked_ids(
        "RGB-D improvements:", np.argsort(rgbd_differences)[-4:][::-1], records
    )
    print("Four strongest RGB-D BCE + Dice failures versus prior")
    print_ranked_ids(
        "RGB-D failures:", np.argsort(rgbd_differences)[:4], records
    )


def main():
    """Run fixed-threshold test diagnostics without model selection or training."""
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
    test_split = splits["test"]
    test_keys = test_split["sample_keys"]
    if len(test_keys) != EXPECTED_TEST_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_TEST_SAMPLES} test samples, got {len(test_keys)}."
        )

    train_keys = splits["train"]["sample_keys"]
    gt_paths = average_baseline.find_summer_gt_files(baseline.DATASET_DIR)
    training_masks = average_baseline.load_ground_truth_masks(train_keys, gt_paths)
    average_gt = np.mean(training_masks, axis=0, dtype=np.float32)[:, :, 0]
    del training_masks
    prior_binary = np.broadcast_to(
        (average_gt >= PRIOR_THRESHOLD)[np.newaxis, :, :, np.newaxis],
        (EXPECTED_TEST_SAMPLES, 256, 256, 1),
    )

    test_pairs = {key: summer_pairs[key] for key in test_keys}
    rgb_inputs, targets, _, rgb_keys = baseline.prepare_rgb_gt_dataset(test_pairs)
    test_triplets = {key: triplets[key] for key in test_keys}
    rgbd_inputs, rgbd_targets, _, rgbd_keys = rgbd_experiment.prepare_rgbd_dataset(
        test_triplets
    )
    if rgb_keys != test_keys or rgbd_keys != test_keys:
        raise AssertionError("Prepared test sample order differs from the split.")
    if not np.array_equal(targets, rgbd_targets):
        raise AssertionError("RGB and RGB-D preprocessing produced different GTs.")

    rgb_binary = evaluate_checkpoint(
        rgb_experiment.MODEL_PATH, rgb_inputs, RGB_THRESHOLD, "RGB BCE + Dice"
    )
    rgbd_binary = evaluate_checkpoint(
        rgbd_experiment.MODEL_PATH, rgbd_inputs, RGBD_THRESHOLD, "RGB-D BCE + Dice"
    )

    prior_dice, prior_iou = per_image_binary_scores(targets, prior_binary)
    rgb_dice, rgb_iou = per_image_binary_scores(targets, rgb_binary)
    rgbd_dice, rgbd_iou = per_image_binary_scores(targets, rgbd_binary)
    rgb_similarity, _ = per_image_binary_scores(rgb_binary, prior_binary)
    rgbd_similarity, _ = per_image_binary_scores(rgbd_binary, prior_binary)

    records = []
    for index, key in enumerate(test_keys):
        records.append(
            {
                "tree_id": key[0],
                "season": key[1],
                "sample_id": key[2],
                "prior_dice": float(prior_dice[index]),
                "prior_iou": float(prior_iou[index]),
                "rgb_dice": float(rgb_dice[index]),
                "rgb_iou": float(rgb_iou[index]),
                "rgb_dice_improvement_over_prior": float(
                    rgb_dice[index] - prior_dice[index]
                ),
                "rgbd_dice": float(rgbd_dice[index]),
                "rgbd_iou": float(rgbd_iou[index]),
                "rgbd_dice_improvement_over_prior": float(
                    rgbd_dice[index] - prior_dice[index]
                ),
                "rgb_prediction_to_prior_similarity": float(rgb_similarity[index]),
                "rgbd_prediction_to_prior_similarity": float(
                    rgbd_similarity[index]
                ),
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_per_image_csv(records)
    save_score_distribution_plot(records)
    save_difference_plot(records)

    rgb_delta = np.asarray(
        [record["rgb_dice_improvement_over_prior"] for record in records]
    )
    rgbd_delta = np.asarray(
        [record["rgbd_dice_improvement_over_prior"] for record in records]
    )
    rankings = (
        ("rgb_best_vs_prior", np.argsort(rgb_delta)[-4:][::-1]),
        ("prior_best_vs_rgb", np.argsort(rgb_delta)[:4]),
        ("rgbd_best_vs_prior", np.argsort(rgbd_delta)[-4:][::-1]),
        ("prior_best_vs_rgbd", np.argsort(rgbd_delta)[:4]),
    )
    for category, indices in rankings:
        save_example_montage(
            category,
            indices,
            records,
            rgbd_inputs,
            targets,
            prior_binary,
            rgb_binary,
            rgbd_binary,
        )

    print(f"\nHeld-out test samples analysed: {len(records)}")
    print(f"Fixed thresholds: prior={PRIOR_THRESHOLD:.2f}, RGB={RGB_THRESHOLD:.2f}, RGB-D={RGBD_THRESHOLD:.2f}")
    print_summary(records)
    print(f"\nAll analysis outputs saved under: {RESULTS_DIR}")
    plt.show()


if __name__ == "__main__":
    main()