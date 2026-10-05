"""Evaluate the saved RGB-D BCE + Dice checkpoint on the held-out test set."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import Tree_Project as baseline
import Tree_Project_RGBD_BCE_Dice as experiment


TEST_THRESHOLD = 0.35
RGB_ONLY_TEST_DICE = 0.372018
RGB_ONLY_TEST_IOU = 0.231315
EXAMPLES_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "results"
    / "rgbd_bce_dice_test_examples.png"
)


def display_test_examples(inputs, targets, probabilities, sample_keys):
    """Save and display four RGB-D held-out test examples."""
    if len(inputs) == 0:
        raise ValueError("The held-out test split is empty.")

    selected = np.linspace(0, len(inputs) - 1, min(4, len(inputs)), dtype=int)
    figure, axes = plt.subplots(len(selected), 5, figsize=(18, 3.5 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, index in enumerate(selected):
        axes[row, 0].imshow(inputs[index, :, :, :3])
        axes[row, 0].set_title(f"RGB: {sample_keys[index][0]}")
        axes[row, 1].imshow(inputs[index, :, :, 3], cmap="viridis", vmin=0, vmax=1)
        axes[row, 1].set_title("Depth (normalized)")
        axes[row, 2].imshow(targets[index, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("Ground Truth")
        axes[row, 3].imshow(
            probabilities[index, :, :, 0], cmap="viridis", vmin=0, vmax=1
        )
        axes[row, 3].set_title("Predicted Probability")
        axes[row, 4].imshow(
            probabilities[index, :, :, 0] >= TEST_THRESHOLD,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, 4].set_title(f"Binary Mask ({TEST_THRESHOLD:.2f})")
        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle(
        f"Held-out RGB-D test predictions at threshold {TEST_THRESHOLD:.2f}"
    )
    figure.tight_layout()
    EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(EXAMPLES_PATH, dpi=150, bbox_inches="tight")
    print(f"Test examples saved to: {EXAMPLES_PATH}")


def main():
    """Load the checkpoint and evaluate only the fixed held-out test split."""
    if not baseline.DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {baseline.DATASET_DIR}")
    if not experiment.MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained RGB-D checkpoint not found: {experiment.MODEL_PATH}"
        )

    images = baseline.scan_images(baseline.DATASET_DIR)
    summer_pairs = baseline.find_summer_rgb_gt_pairs(images)
    triplets, unmatched = baseline.find_summer_triplets(images)
    if unmatched:
        details = "; ".join(
            f"{key}: missing {', '.join(missing)}"
            for key, missing in unmatched.items()
        )
        raise RuntimeError(f"Incomplete Summer RGB-D-GT groups: {details}")
    if set(summer_pairs) != set(triplets):
        raise ValueError(
            "RGB-D triplets do not match the RGB baseline Summer RGB/GT samples."
        )

    sample_keys = sorted(summer_pairs)
    tree_ids = np.asarray([key[0] for key in sample_keys])
    splits = baseline.split_by_tree(
        tree_ids, sample_keys, seed=baseline.RANDOM_SEED
    )
    baseline.verify_tree_split_isolation(splits)
    test_split = splits["test"]
    test_triplets = {key: triplets[key] for key in test_split["sample_keys"]}

    test_inputs, test_targets, test_tree_ids, test_sample_keys = (
        experiment.prepare_rgbd_dataset(test_triplets)
    )
    if len(test_inputs) != 85 or len(test_split["tree_ids"]) != 85:
        raise AssertionError(
            "Expected exactly 85 held-out test samples and tree IDs; got "
            f"{len(test_inputs)} samples and {len(test_split['tree_ids'])} trees."
        )
    if set(test_tree_ids) != test_split["tree_ids"]:
        raise AssertionError("Prepared test tree IDs do not match the seed-42 split.")
    if test_inputs.shape[1:] != experiment.INPUT_SHAPE:
        raise AssertionError(f"Unexpected test input shape: {test_inputs.shape}")
    if test_targets.shape[1:] != (256, 256, 1):
        raise AssertionError(f"Unexpected test target shape: {test_targets.shape}")

    print(f"Held-out test trees: {len(test_split['tree_ids'])}")
    print(f"Held-out test samples loaded: {len(test_inputs)}")
    print(f"Test input and target shapes: {test_inputs.shape}, {test_targets.shape}")
    print("Loading the saved RGB-D checkpoint for inference only.")
    model = tf.keras.models.load_model(experiment.MODEL_PATH, compile=False)
    if model.input_shape[1:] != experiment.INPUT_SHAPE:
        raise AssertionError(f"Unexpected checkpoint input shape: {model.input_shape}")

    probabilities = model.predict(
        test_inputs, batch_size=baseline.BATCH_SIZE, verbose=1
    )
    binary_predictions = probabilities >= TEST_THRESHOLD
    binary_dice = baseline.binary_dice_coefficient(
        test_targets, binary_predictions
    )
    binary_iou = baseline.binary_intersection_over_union(
        test_targets, binary_predictions
    )

    print("\nFinal RGB-D BCE + Dice test performance")
    print(f"Validation-selected threshold: {TEST_THRESHOLD:.2f}")
    print(f"Binary Dice: {binary_dice:.6f}")
    print(f"Binary IoU: {binary_iou:.6f}")
    print("\nPrevious RGB-only BCE + Dice test results")
    print("Validation-selected threshold: 0.30")
    print(f"Binary Dice: {RGB_ONLY_TEST_DICE:.6f}")
    print(f"Binary IoU: {RGB_ONLY_TEST_IOU:.6f}")

    display_test_examples(
        test_inputs, test_targets, probabilities, test_sample_keys
    )
    plt.show()


if __name__ == "__main__":
    main()