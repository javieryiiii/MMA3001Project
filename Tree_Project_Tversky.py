"""Train an RGB-only U-Net with a fixed false-positive-weighted Tversky loss."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import Tree_Project as baseline


MODEL_PATH = Path(__file__).resolve().parent / "models" / "rgb_unet_tversky.keras"
RESULTS_DIR = Path(__file__).resolve().parent / "models" / "results"
ALPHA = 0.7
BETA = 0.3
SMOOTH = 1e-6


def tversky_loss(y_true, y_pred):
    """Return mean per-image Tversky loss for soft binary-segmentation masks.

    TP counts true-positive foreground pixels. FP counts background pixels
    incorrectly predicted as foreground. FN counts foreground pixels
    incorrectly predicted as background. With the fixed alpha=0.7 and beta=0.3,
    the denominator weights FP more strongly than FN; smoothing is 1e-6.

    Args:
        y_true: Target tensor shaped ``(N, H, W, C)``.
        y_pred: Soft prediction probabilities with the same shape.

    Returns:
        Scalar mean of one-minus-Tversky-index, reduced over non-batch axes and
        then averaged over the batch.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = (1, 2, 3)

    true_positive = tf.reduce_sum(y_true * y_pred, axis=axes)
    false_positive = tf.reduce_sum((1.0 - y_true) * y_pred, axis=axes)
    false_negative = tf.reduce_sum(y_true * (1.0 - y_pred), axis=axes)

    tversky_index = (true_positive + SMOOTH) / (
        true_positive
        + ALPHA * false_positive
        + BETA * false_negative
        + SMOOTH
    )
    return tf.reduce_mean(1.0 - tversky_index)


def save_training_curves(history):
    """Save separate train/validation loss, Dice and IoU plots."""
    curves = (
        ("loss", "Tversky loss", "rgb_tversky_loss_curves.png"),
        ("dice_coefficient", "Dice coefficient", "rgb_tversky_dice_curves.png"),
        (
            "intersection_over_union",
            "Intersection over union",
            "rgb_tversky_iou_curves.png",
        ),
    )
    for metric_name, title, filename in curves:
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.plot(history.history[metric_name], label="Training")
        axis.plot(history.history[f"val_{metric_name}"], label="Validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.set_title(f"RGB Tversky: {title}")
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        output_path = RESULTS_DIR / filename
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"Training curve saved to: {output_path}")


def save_validation_examples(inputs, targets, probabilities, sample_keys, threshold):
    """Save RGB, GT, probability and binary prediction examples."""
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

    figure.suptitle("RGB Tversky validation predictions")
    figure.tight_layout()
    output_path = RESULTS_DIR / "rgb_tversky_validation_examples.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Validation examples saved to: {output_path}")


def print_foreground_diagnostics(targets, probabilities, threshold):
    """Report foreground fractions and per-image predicted/GT area ratios."""
    ground_truth = targets >= 0.5
    predictions = probabilities >= threshold
    pixel_count = np.prod(targets.shape[1:])
    gt_counts = np.sum(ground_truth, axis=(1, 2, 3))
    predicted_counts = np.sum(predictions, axis=(1, 2, 3))
    if np.any(gt_counts == 0):
        raise ValueError("A validation GT mask has no foreground pixels.")

    gt_fractions = gt_counts / pixel_count
    predicted_fractions = predicted_counts / pixel_count
    ratios = predicted_counts / gt_counts
    mean_gt_fraction = float(np.mean(gt_fractions))
    mean_prediction_fraction = float(np.mean(predicted_fractions))
    mean_ratio = float(np.mean(ratios))
    median_ratio = float(np.median(ratios))

    print("\nValidation foreground diagnostics at selected threshold")
    print(f"Mean GT foreground: {mean_gt_fraction * 100:.3f}%")
    print(f"Mean predicted foreground: {mean_prediction_fraction * 100:.3f}%")
    print(f"Mean predicted/GT foreground ratio: {mean_ratio:.6f}")
    print(f"Median predicted/GT foreground ratio: {median_ratio:.6f}")
    for cutoff in (1.0, 1.5, 2.0):
        count = int(np.count_nonzero(ratios > cutoff))
        print(
            f"Ratio > {cutoff:g}: {count}/{len(ratios)} "
            f"({100 * count / len(ratios):.1f}%)"
        )
    return mean_gt_fraction, mean_prediction_fraction, mean_ratio, median_ratio, ratios


def main():
    """Train on the train split and use validation only for selection/diagnostics."""
    if not baseline.DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {baseline.DATASET_DIR}")
    if MODEL_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing Tversky checkpoint: {MODEL_PATH}"
        )

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

    print("Final controlled RGB-only Tversky experiment")
    print(f"Random seed: {baseline.RANDOM_SEED}")
    print(f"Training samples: {len(train_inputs)}")
    print(f"Validation samples: {len(validation_inputs)}")
    print("Held-out test samples are not loaded or used.")
    print(f"Input/target shapes: {train_inputs.shape}, {train_targets.shape}")
    print(f"Tversky alpha: {ALPHA}; beta: {BETA}; smooth: {SMOOTH}")
    print("Loss: Tversky loss (false-positive weight exceeds false-negative weight)")
    print(f"Optimizer: Adam; learning rate: 1e-3; batch size: {baseline.BATCH_SIZE}")
    print(f"Maximum epochs: {baseline.MAX_EPOCHS}; early stopping patience: 5")
    print(f"New checkpoint path: {MODEL_PATH}")

    tf.keras.utils.set_random_seed(baseline.RANDOM_SEED)
    model = baseline.build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tversky_loss,
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
    print(f"Saved new checkpoint: {MODEL_PATH}")
    save_training_curves(history)

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
    print(f"Alpha: {ALPHA}; beta: {BETA}")
    print(f"Selected validation threshold: {selected['threshold']:.2f}")
    print(f"Validation Dice: {selected['dice']:.6f}")
    print(f"Validation IoU: {selected['iou']:.6f}")

    _, mean_prediction_fraction, mean_ratio, median_ratio, ratios = (
        print_foreground_diagnostics(
            validation_targets, probabilities, selected["threshold"]
        )
    )
    print("\nComparison with completed RGB experiments")
    print("RGB BCE + Dice: Dice=0.365778, IoU=0.226002, FG ratio=2.577")
    print("RGB Focal + Dice: Dice=0.366954, IoU=0.227173, FG ratio=2.576")
    print(
        f"RGB Tversky: Dice={selected['dice']:.6f}, "
        f"IoU={selected['iou']:.6f}, FG ratio={mean_ratio:.6f}"
    )
    print(
        f"RGB Tversky mean predicted foreground: "
        f"{mean_prediction_fraction * 100:.3f}%"
    )

    save_validation_examples(
        validation_inputs,
        validation_targets,
        probabilities,
        validation_keys,
        selected["threshold"],
    )
    plt.show()


if __name__ == "__main__":
    main()