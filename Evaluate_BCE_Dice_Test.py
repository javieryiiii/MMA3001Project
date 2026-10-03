"""Evaluate the saved RGB-only BCE + Dice checkpoint on the held-out test set."""

import matplotlib.pyplot as plt
import tensorflow as tf

import Tree_Project as baseline
import Tree_Project_BCE_Dice as experiment


TEST_THRESHOLD = 0.30
BASELINE_TEST_DICE = 0.350560
BASELINE_TEST_IOU = 0.214901


def main():
    """Run inference-only evaluation at the validation-selected threshold."""
    if not baseline.DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {baseline.DATASET_DIR}"
        )
    if not experiment.MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model checkpoint not found: {experiment.MODEL_PATH}"
        )

    images = baseline.scan_images(baseline.DATASET_DIR)
    summer_pairs = baseline.find_summer_rgb_gt_pairs(images)
    if not summer_pairs:
        raise RuntimeError("At least one complete Summer RGB/GT pair is required.")

    sample_keys = sorted(summer_pairs)
    inputs, targets, tree_ids, sample_keys = baseline.prepare_rgb_gt_dataset(
        summer_pairs
    )
    splits = baseline.split_by_tree(
        tree_ids, sample_keys, seed=baseline.RANDOM_SEED
    )
    baseline.verify_tree_split_isolation(splits)

    test_indices = splits["test"]["indices"]
    test_inputs = inputs[test_indices]
    test_targets = targets[test_indices]
    test_sample_keys = [sample_keys[index] for index in test_indices]
    if len(test_inputs) == 0:
        raise RuntimeError("The held-out test split is empty.")

    print(f"Held-out test trees: {len(splits['test']['tree_ids'])}")
    print(f"Held-out test images: {len(test_inputs)}")
    print("Loading the saved BCE + Dice checkpoint for inference only.")
    model = tf.keras.models.load_model(experiment.MODEL_PATH, compile=False)
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

    print("\nFinal RGB-only BCE + Dice test performance")
    print(f"Validation-selected threshold: {TEST_THRESHOLD:.2f}")
    print(f"Binary Dice: {binary_dice:.6f}")
    print(f"Binary IoU: {binary_iou:.6f}")
    print("\nComparison with the original RGB-only BCE baseline")
    print("Original BCE baseline (threshold 0.25):")
    print(f"  Binary Dice: {BASELINE_TEST_DICE:.6f}")
    print(f"  Binary IoU: {BASELINE_TEST_IOU:.6f}")
    print("BCE + Dice (threshold 0.30):")
    print(f"  Binary Dice: {binary_dice:.6f}")
    print(f"  Binary IoU: {binary_iou:.6f}")

    baseline.display_test_predictions(
        test_inputs,
        test_targets,
        probabilities,
        test_sample_keys,
        threshold=TEST_THRESHOLD,
    )
    examples_path = experiment.MODEL_PATH.with_name(
        "rgb_unet_bce_dice_test_examples.png"
    )
    plt.savefig(examples_path, dpi=150, bbox_inches="tight")
    print(f"\nRepresentative test examples: {examples_path}")
    plt.show()


if __name__ == "__main__":
    main()