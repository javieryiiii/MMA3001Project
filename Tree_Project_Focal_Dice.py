"""Train an RGB-only U-Net with binary focal loss plus Dice loss."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import Tree_Project as baseline


MODEL_PATH = Path(__file__).resolve().parent / "models" / "rgb_unet_focal_dice.keras"
RESULTS_DIR = Path(__file__).resolve().parent / "models" / "results"
GAMMA = 2.0


def dice_loss(y_true, y_pred):
    """Return one minus the baseline's mean soft Dice coefficient.

    Args:
        y_true: Ground-truth tensor shaped ``(N, H, W, C)``.
        y_pred: Soft prediction probabilities with the same shape.

    Returns:
        Scalar Dice loss using the baseline metric's 1e-6 smoothing.
    """
    return 1.0 - baseline.dice_coefficient(y_true, y_pred)


def binary_focal_loss(y_true, y_pred):
    """Compute mean binary focal cross-entropy for sigmoid probabilities.

    For each pixel, focal loss is BCE multiplied by (1 - p_t) ** gamma,
    where p_t is the probability assigned to the true class. No alpha/class
    balancing is added; gamma=2.0 is the only focal-loss parameter here.

    Args:
        y_true: Binary target tensor.
        y_pred: Sigmoid probabilities with the same shape as ``y_true``; values
        are clipped to the Keras epsilon bounds before logarithms are calculated.

    Returns:
        Scalar mean of the focal-weighted pixelwise binary cross-entropies.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    probabilities = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    pixel_bce = -(
        y_true * tf.math.log(probabilities)
        + (1.0 - y_true) * tf.math.log(1.0 - probabilities)
    )
    probability_for_true_class = (
        y_true * probabilities + (1.0 - y_true) * (1.0 - probabilities)
    )
    focal_weight = tf.pow(1.0 - probability_for_true_class, GAMMA)
    return tf.reduce_mean(focal_weight * pixel_bce)


def combined_focal_dice_loss(y_true, y_pred):
    """Add mean binary focal loss to the baseline's mean soft Dice loss.

    Args:
        y_true: Binary target tensor.
        y_pred: Sigmoid probabilities with the same shape as ``y_true``.

    Returns:
        Scalar sum of focal loss and Dice loss.
    """
    return binary_focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


def plot_training_history(history):
    """Save training/validation loss, Dice, and IoU curves."""
    metrics = (
        ("loss", "Focal + Dice loss"),
        ("dice_coefficient", "Dice coefficient"),
        ("intersection_over_union", "IoU"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (metric, title) in zip(axes, metrics):
        axis.plot(history.history[metric], label="Training")
        axis.plot(history.history[f"val_{metric}"], label="Validation")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.tight_layout()
    path = RESULTS_DIR / "rgb_focal_dice_training_curves.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    print(f"Training curves saved to: {path}")


def display_validation_predictions(inputs, targets, probabilities, sample_keys, threshold):
    """Save examples with RGB, GT, probability map and fixed-threshold output."""
    if len(inputs) == 0:
        raise ValueError("The validation split is empty.")
    selected = np.linspace(0, len(inputs) - 1, min(4, len(inputs)), dtype=int)
    figure, axes = plt.subplots(len(selected), 4, figsize=(14, 3.5 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, index in enumerate(selected):
        axes[row, 0].imshow(inputs[index])
        axes[row, 0].set_title(f"RGB: {sample_keys[index][0]}")
        axes[row, 1].imshow(targets[index, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground-truth skeleton")
        axes[row, 2].imshow(
            probabilities[index, :, :, 0], cmap="viridis", vmin=0, vmax=1
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

    figure.suptitle("RGB Focal + Dice validation predictions")
    figure.tight_layout()
    path = RESULTS_DIR / "rgb_focal_dice_validation_examples.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    print(f"Validation examples saved to: {path}")


def print_foreground_statistics(targets, probabilities, threshold):
    """Report foreground fractions and predicted/GT area ratios."""
    ground_truth = targets >= 0.5
    predicted = probabilities >= threshold
    pixel_count = np.prod(targets.shape[1:])
    gt_counts = np.sum(ground_truth, axis=(1, 2, 3))
    predicted_counts = np.sum(predicted, axis=(1, 2, 3))
    if np.any(gt_counts == 0):
        raise ValueError("A validation GT mask has no foreground pixels.")

    gt_fractions = gt_counts / pixel_count
    predicted_fractions = predicted_counts / pixel_count
    ratios = predicted_counts / gt_counts
    print("\nValidation foreground statistics at selected threshold")
    print(f"Mean GT foreground fraction: {np.mean(gt_fractions):.6f} ({np.mean(gt_fractions) * 100:.3f}%)")
    print(f"Mean predicted foreground fraction: {np.mean(predicted_fractions):.6f} ({np.mean(predicted_fractions) * 100:.3f}%)")
    print(f"Mean predicted/GT foreground ratio: {np.mean(ratios):.6f}")
    print(f"Median predicted/GT foreground ratio: {np.median(ratios):.6f}")
    for cutoff in (1.0, 1.5, 2.0):
        count = int(np.count_nonzero(ratios > cutoff))
        print(
            f"Samples with ratio > {cutoff:g}: {count}/{len(ratios)} "
            f"({100 * count / len(ratios):.1f}%)"
        )


def main():
    """Train on train data and select the threshold using validation only."""
    if not baseline.DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {baseline.DATASET_DIR}")

    images = baseline.scan_images(baseline.DATASET_DIR)
    summer_pairs = baseline.find_summer_rgb_gt_pairs(images)
    if not summer_pairs:
        raise RuntimeError("At least one complete Summer RGB/GT pair is required.")

    sample_keys = sorted(summer_pairs)
    tree_ids = np.asarray([key[0] for key in sample_keys])
    splits = baseline.split_by_tree(
        tree_ids, sample_keys, seed=baseline.RANDOM_SEED
    )
    baseline.verify_tree_split_isolation(splits)

    experiment_indices = np.sort(
        np.concatenate((splits["train"]["indices"], splits["validation"]["indices"]))
    )
    experiment_keys = [sample_keys[index] for index in experiment_indices]
    experiment_pairs = {key: summer_pairs[key] for key in experiment_keys}
    inputs, targets, prepared_tree_ids, prepared_keys = baseline.prepare_rgb_gt_dataset(
        experiment_pairs
    )

    train_trees = splits["train"]["tree_ids"]
    validation_trees = splits["validation"]["tree_ids"]
    train_indices = np.asarray(
        [index for index, tree_id in enumerate(prepared_tree_ids) if tree_id in train_trees],
        dtype=np.int32,
    )
    validation_indices = np.asarray(
        [index for index, tree_id in enumerate(prepared_tree_ids) if tree_id in validation_trees],
        dtype=np.int32,
    )

    train_inputs = inputs[train_indices]
    train_targets = targets[train_indices]
    validation_inputs = inputs[validation_indices]
    validation_targets = targets[validation_indices]
    validation_keys = [prepared_keys[index] for index in validation_indices]
    if len(train_inputs) != 393 or len(validation_inputs) != 84:
        raise AssertionError(
            f"Expected 393 train and 84 validation samples; got "
            f"{len(train_inputs)} and {len(validation_inputs)}."
        )
    if set(prepared_tree_ids[train_indices]) != train_trees:
        raise AssertionError("Prepared train tree IDs differ from the seed-42 split.")
    if set(prepared_tree_ids[validation_indices]) != validation_trees:
        raise AssertionError("Prepared validation tree IDs differ from the seed-42 split.")

    print("Controlled RGB-only Focal + Dice experiment")
    print(f"Random seed: {baseline.RANDOM_SEED}")
    print(f"Training samples: {len(train_inputs)}")
    print(f"Validation samples: {len(validation_inputs)}")
    print("Held-out test samples are not loaded or used.")
    print(f"Input/target shapes: {train_inputs.shape}, {train_targets.shape}")
    print(f"Focal loss gamma: {GAMMA}; alpha balancing: disabled")
    print("Loss: binary focal cross-entropy + existing smoothed Dice loss")
    print(f"Optimizer: Adam; learning rate: 1e-3; batch size: {baseline.BATCH_SIZE}")
    print(f"Maximum epochs: {baseline.MAX_EPOCHS}; early stopping patience: 5")
    print(f"Checkpoint path: {MODEL_PATH}")

    tf.keras.utils.set_random_seed(baseline.RANDOM_SEED)
    model = baseline.build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_focal_dice_loss,
        metrics=[baseline.dice_coefficient, baseline.intersection_over_union],
    )
    if model.input_shape[1:] != (256, 256, 3):
        raise AssertionError(f"Unexpected model input shape: {model.input_shape}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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

    best_epoch_index = int(np.argmin(history.history["val_loss"]))
    best_epoch = best_epoch_index + 1
    best_validation_loss = float(history.history["val_loss"][best_epoch_index])
    print(f"\nBest epoch (lowest validation loss): {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Training stopped after {len(history.history['loss'])} epochs")
    print(f"New RGB Focal + Dice checkpoint: {MODEL_PATH}")
    plot_training_history(history)

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
    print(f"Selected threshold: {selected['threshold']:.2f}")
    print(f"Validation binary Dice: {selected['dice']:.6f}")
    print(f"Validation binary IoU: {selected['iou']:.6f}")

    print_foreground_statistics(
        validation_targets, probabilities, selected["threshold"]
    )
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