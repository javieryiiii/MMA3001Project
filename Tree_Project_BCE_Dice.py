"""RGB-only U-Net experiment using combined BCE + Dice loss.

Uses the baseline's Summer data, preprocessing, tree-level split, architecture,
optimizer settings, and early-stopping procedure. The held-out test split is
not loaded or evaluated.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import Tree_Project as baseline


MODEL_PATH = Path(__file__).resolve().parent / "models" / "rgb_unet_bce_dice.keras"


def dice_loss(y_true, y_pred):
    """Return one minus the baseline's smoothed Dice coefficient."""
    return 1.0 - baseline.dice_coefficient(y_true, y_pred)


def combined_bce_dice_loss(y_true, y_pred):
    """Return mean binary cross-entropy plus Dice loss."""
    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred)
    )
    return bce + dice_loss(y_true, y_pred)


def display_validation_predictions(
    inputs, targets, probabilities, sample_keys, threshold
):
    """Display validation RGB, ground truth, probabilities, and predictions."""
    if len(inputs) == 0:
        raise ValueError("The validation split is empty.")

    count = min(4, len(inputs))
    selected = np.linspace(0, len(inputs) - 1, count, dtype=int)
    figure, axes = plt.subplots(count, 4, figsize=(14, 3.5 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, index in enumerate(selected):
        axes[row, 0].imshow(inputs[index])
        axes[row, 0].set_title(f"RGB: {sample_keys[index][0]}")

        axes[row, 1].imshow(
            targets[index, :, :, 0], cmap="gray", vmin=0, vmax=1
        )
        axes[row, 1].set_title("Ground-truth skeleton")

        axes[row, 2].imshow(
            probabilities[index, :, :, 0],
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        axes[row, 2].set_title("Predicted probability")

        axes[row, 3].imshow(
            probabilities[index, :, :, 0] >= threshold,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, 3].set_title(f"Prediction (threshold {threshold:.2f})")

        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle("Validation examples using the selected threshold")
    figure.tight_layout()


def main():
    """Train the experiment and select a threshold using validation data only."""
    if not baseline.DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {baseline.DATASET_DIR}"
        )

    images = baseline.scan_images(baseline.DATASET_DIR)
    summer_pairs = baseline.find_summer_rgb_gt_pairs(images)
    if not summer_pairs:
        raise RuntimeError("At least one complete Summer RGB/GT pair is required.")

    sample_keys = sorted(summer_pairs)
    tree_ids = np.array([key[0] for key in sample_keys])

    # Reuse the baseline's seeded tree-ID split.
    splits = baseline.split_by_tree(tree_ids, sample_keys, seed=42)
    baseline.verify_tree_split_isolation(splits)

    # Prepare only training and validation samples; do not load test images.
    experiment_indices = np.sort(
        np.concatenate((
            splits["train"]["indices"],
            splits["validation"]["indices"],
        ))
    )
    experiment_keys = [sample_keys[index] for index in experiment_indices]
    experiment_pairs = {key: summer_pairs[key] for key in experiment_keys}

    inputs, targets, prepared_tree_ids, prepared_keys = (
        baseline.prepare_rgb_gt_dataset(experiment_pairs)
    )

    train_trees = splits["train"]["tree_ids"]
    validation_trees = splits["validation"]["tree_ids"]

    train_indices = np.array(
        [
            index for index, tree_id in enumerate(prepared_tree_ids)
            if tree_id in train_trees
        ],
        dtype=np.int32,
    )
    validation_indices = np.array(
        [
            index for index, tree_id in enumerate(prepared_tree_ids)
            if tree_id in validation_trees
        ],
        dtype=np.int32,
    )

    train_inputs = inputs[train_indices]
    train_targets = targets[train_indices]
    validation_inputs = inputs[validation_indices]
    validation_targets = targets[validation_indices]
    validation_keys = [prepared_keys[index] for index in validation_indices]

    # Match the baseline seed, U-Net, Adam optimizer, and learning rate.
    tf.keras.utils.set_random_seed(42)
    model = baseline.build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_bce_dice_loss,
        metrics=[
            baseline.dice_coefficient,
            baseline.intersection_over_union,
        ],
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_inputs,
        train_targets,
        validation_data=(validation_inputs, validation_targets),
        epochs=baseline.MAX_EPOCHS,
        batch_size=baseline.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\nBest epoch (lowest validation loss): {best_epoch}")
    print(f"Experimental checkpoint: {MODEL_PATH}")

    # Use validation probabilities only for threshold selection.
    probabilities = model.predict(
        validation_inputs,
        batch_size=baseline.BATCH_SIZE,
        verbose=1,
    )
    results = baseline.score_validation_thresholds(
        validation_targets,
        probabilities,
        baseline.VALIDATION_THRESHOLDS,
    )
    selected = max(results, key=lambda result: result["dice"])

    print("\nValidation threshold results")
    for result in results:
        print(
            f"Threshold {result['threshold']:.2f}: "
            f"Dice {result['dice']:.6f}, IoU {result['iou']:.6f}"
        )

    print("\nSelected validation result")
    print(f"Best epoch: {best_epoch}")
    print(f"Selected validation threshold: {selected['threshold']:.2f}")
    print(f"Validation binary Dice: {selected['dice']:.6f}")
    print(f"Validation binary IoU: {selected['iou']:.6f}")

    display_validation_predictions(
        validation_inputs,
        validation_targets,
        probabilities,
        validation_keys,
        selected["threshold"],
    )
    plt.show()


if __name__ == "__main__":
    main()