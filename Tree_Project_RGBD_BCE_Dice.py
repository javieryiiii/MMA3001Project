"""Train a controlled RGB-D U-Net experiment using BCE + Dice loss."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import Tree_Project as baseline
import Tree_Project_BCE_Dice as rgb_bce_dice


MODEL_PATH = Path(__file__).resolve().parent / "models" / "rgbd_unet_bce_dice.keras"
VALIDATION_EXAMPLES_PATH = Path(__file__).resolve().parent / "models" / "rgbd_unet_bce_dice_validation_examples.png"
INPUT_SHAPE = (256, 256, 4)
DEPTH_SCALE = 255.0


def prepare_rgbd_dataset(triplets):
    """Apply the baseline RGB/GT preprocessing and append normalized depth."""
    sample_keys = sorted(triplets)
    inputs = np.empty((len(sample_keys), *INPUT_SHAPE), dtype=np.float32)
    targets = np.empty((len(sample_keys), 256, 256, 1), dtype=np.float32)
    tree_ids = []

    for index, key in enumerate(sample_keys):
        modalities = triplets[key]
        rgb_bgr = baseline.load_image(modalities["RGB"], cv2.IMREAD_COLOR)
        depth = baseline.load_image(modalities["D"], cv2.IMREAD_UNCHANGED)
        gt = baseline.load_image(modalities["GT"], cv2.IMREAD_GRAYSCALE)

        if rgb_bgr.shape != (480, 640, 3):
            raise ValueError(
                f"Unexpected RGB dimensions for {modalities['RGB'].name}: "
                f"{rgb_bgr.shape}"
            )
        if depth.shape != (480, 640, 3) or depth.dtype != np.uint8:
            raise ValueError(
                f"Unexpected depth representation for {modalities['D'].name}: "
                f"{depth.shape}, {depth.dtype}"
            )
        if not (
            np.array_equal(depth[:, :, 0], depth[:, :, 1])
            and np.array_equal(depth[:, :, 1], depth[:, :, 2])
        ):
            raise ValueError(
                f"Depth channels are not identical: {modalities['D'].name}"
            )
        if gt.shape != baseline.TARGET_SIZE:
            raise ValueError(
                f"Unexpected GT dimensions for {modalities['GT'].name}: {gt.shape}"
            )
        if not np.isin(np.unique(gt), [0, 255]).all():
            raise ValueError(f"GT mask is not binary (0/255): {modalities['GT'].name}")

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb_crop = rgb[:, 80:560]
        rgb_resized = cv2.resize(
            rgb_crop, baseline.TARGET_SIZE, interpolation=cv2.INTER_AREA
        )
        inputs[index, :, :, :3] = rgb_resized.astype(np.float32) / 255.0

        depth_crop = depth[:, 80:560, 0]
        depth_resized = cv2.resize(
            depth_crop, baseline.TARGET_SIZE, interpolation=cv2.INTER_AREA
        )
        inputs[index, :, :, 3] = depth_resized.astype(np.float32) / DEPTH_SCALE

        targets[index, :, :, 0] = (gt == 255).astype(np.float32)
        tree_ids.append(key[0])

    return inputs, targets, np.asarray(tree_ids), sample_keys


def build_model():
    """Build the baseline U-Net with four input channels and matching settings."""
    tf.keras.utils.set_random_seed(baseline.RANDOM_SEED)
    model = baseline.build_unet(input_shape=INPUT_SHAPE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=rgb_bce_dice.combined_bce_dice_loss,
        metrics=[
            baseline.dice_coefficient,
            baseline.intersection_over_union,
        ],
    )
    return model


def display_validation_predictions(
    inputs, targets, probabilities, sample_keys, threshold
):
    """Display validation RGB, depth, GT, probabilities, and binary predictions."""
    if len(inputs) == 0:
        raise ValueError("The validation split is empty.")

    count = min(4, len(inputs))
    selected = np.linspace(0, len(inputs) - 1, count, dtype=int)
    figure, axes = plt.subplots(count, 5, figsize=(18, 3.5 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, index in enumerate(selected):
        axes[row, 0].imshow(inputs[index, :, :, :3])
        axes[row, 0].set_title(f"RGB: {sample_keys[index][0]}")
        axes[row, 1].imshow(
            inputs[index, :, :, 3], cmap="viridis", vmin=0, vmax=1
        )
        axes[row, 1].set_title("Depth (normalized source scale)")
        axes[row, 2].imshow(
            targets[index, :, :, 0], cmap="gray", vmin=0, vmax=1
        )
        axes[row, 2].set_title("Ground-truth skeleton")
        axes[row, 3].imshow(
            probabilities[index, :, :, 0],
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        axes[row, 3].set_title("Predicted probability")
        axes[row, 4].imshow(
            probabilities[index, :, :, 0] >= threshold,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, 4].set_title(f"Prediction (threshold {threshold:.2f})")
        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle("RGB-D validation predictions at the selected threshold")
    figure.tight_layout()
    VALIDATION_EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(VALIDATION_EXAMPLES_PATH, dpi=150, bbox_inches="tight")
    print(f"Validation examples saved to: {VALIDATION_EXAMPLES_PATH}")


def main():
    """Train on train data and select a threshold using validation data only."""
    if not baseline.DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {baseline.DATASET_DIR}"
        )

    images = baseline.scan_images(baseline.DATASET_DIR)
    summer_pairs = baseline.find_summer_rgb_gt_pairs(images)
    triplets, unmatched = baseline.find_summer_triplets(images)
    if unmatched:
        missing = "; ".join(
            f"{key}: missing {', '.join(files)}" for key, files in unmatched.items()
        )
        raise RuntimeError(f"Incomplete Summer RGB-D-GT groups: {missing}")
    if not summer_pairs:
        raise RuntimeError("At least one complete Summer RGB/GT pair is required.")

    sample_keys = sorted(summer_pairs)
    if set(sample_keys) != set(triplets):
        raise ValueError(
            "Complete RGB-D-GT triplets do not match the RGB baseline's "
            "Summer RGB/GT sample set."
        )

    tree_ids = np.array([key[0] for key in sample_keys])
    splits = baseline.split_by_tree(
        tree_ids, sample_keys, seed=baseline.RANDOM_SEED
    )
    baseline.verify_tree_split_isolation(splits)

    train_keys = splits["train"]["sample_keys"]
    validation_keys = splits["validation"]["sample_keys"]
    train_triplets = {key: triplets[key] for key in train_keys}
    validation_triplets = {key: triplets[key] for key in validation_keys}

    train_inputs, train_targets, train_tree_ids, prepared_train_keys = (
        prepare_rgbd_dataset(train_triplets)
    )
    validation_inputs, validation_targets, validation_tree_ids, validation_keys = (
        prepare_rgbd_dataset(validation_triplets)
    )

    if train_inputs.shape[1:] != INPUT_SHAPE or validation_inputs.shape[1:] != INPUT_SHAPE:
        raise AssertionError("Prepared RGB-D inputs do not match the model input shape.")
    if train_targets.shape[1:] != (256, 256, 1) or validation_targets.shape[1:] != (256, 256, 1):
        raise AssertionError("Prepared GT targets do not match the expected shape.")
    if set(train_tree_ids) != splits["train"]["tree_ids"]:
        raise AssertionError("Prepared training tree IDs do not match the baseline split.")
    if set(validation_tree_ids) != splits["validation"]["tree_ids"]:
        raise AssertionError("Prepared validation tree IDs do not match the baseline split.")

    print("Controlled RGB-D BCE + Dice experiment")
    print(f"Random seed: {baseline.RANDOM_SEED}")
    print(f"Training samples: {len(train_inputs)}")
    print(f"Validation samples: {len(validation_inputs)}")
    print("Held-out test samples are not loaded or used.")
    print(f"Train input/target shapes: {train_inputs.shape}, {train_targets.shape}")
    print(
        "Validation input/target shapes: "
        f"{validation_inputs.shape}, {validation_targets.shape}"
    )
    print(
        "RGB channel range: "
        f"{train_inputs[:, :, :, :3].min():.6f} to "
        f"{train_inputs[:, :, :, :3].max():.6f}"
    )
    print(
        "Normalized depth channel range: "
        f"{train_inputs[:, :, :, 3].min():.6f} to "
        f"{train_inputs[:, :, :, 3].max():.6f}"
    )
    print("Depth normalization is source-scale division by 255; units are unspecified.")

    model = build_model()
    if model.input_shape[1:] != INPUT_SHAPE:
        raise AssertionError(f"Unexpected model input shape: {model.input_shape}")

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

    best_epoch_index = int(np.argmin(history.history["val_loss"]))
    best_epoch = best_epoch_index + 1
    best_validation_loss = float(history.history["val_loss"][best_epoch_index])
    print(f"\nBest epoch (lowest validation loss): {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"RGB-D checkpoint: {MODEL_PATH}")

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
    print(f"Validation loss: {best_validation_loss:.6f}")
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