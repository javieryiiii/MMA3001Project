"""Visually inspect the current RGB preprocessing against Summer GT masks."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import Tree_Project as baseline


OUTPUT_DIR = Path(__file__).resolve().parent / "models" / "results" / "alignment_check"
SAMPLES_PER_SPLIT = {"train": 8, "validation": 6, "test": 6}
OVERLAY_COLOR = (1.0, 0.0, 0.85)
OVERLAY_ALPHA = 0.72


def select_representative_keys(splits):
    """Select a fixed, evenly spaced set of keys from all three splits."""
    selected = []
    for split_name, requested_count in SAMPLES_PER_SPLIT.items():
        split_keys = splits[split_name]["sample_keys"]
        if len(split_keys) < requested_count:
            raise ValueError(
                f"Split {split_name} has {len(split_keys)} samples; "
                f"need {requested_count}."
            )
        indices = np.linspace(
            0, len(split_keys) - 1, requested_count, dtype=int
        )
        selected.extend(
            (split_name, split_keys[index]) for index in indices
        )
    return selected


def add_gt_overlay(axis, processed_rgb, gt_mask):
    """Draw unchanged GT foreground pixels as a translucent color overlay."""
    axis.imshow(processed_rgb)
    overlay = np.zeros((*gt_mask.shape, 4), dtype=np.float32)
    overlay[gt_mask] = (*OVERLAY_COLOR, OVERLAY_ALPHA)
    axis.imshow(overlay, interpolation="nearest")


def save_sample_figure(split_name, sample_key, modalities, processed_rgb, gt_mask):
    """Save original, processed, GT, and overlay views for one sample."""
    original_bgr = baseline.load_image(modalities["RGB"], cv2.IMREAD_COLOR)
    if original_bgr.shape != (480, 640, 3):
        raise ValueError(
            f"Unexpected original RGB shape for {modalities['RGB'].name}: "
            f"{original_bgr.shape}"
        )
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original RGB (640x480)")
    axes[1].imshow(processed_rgb)
    axes[1].set_title("Current processed RGB (256x256)")
    axes[2].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground-truth skeleton")
    add_gt_overlay(axes[3], processed_rgb, gt_mask)
    axes[3].set_title("Processed RGB + unchanged GT overlay")
    for axis in axes:
        axis.axis("off")

    tree_id, season, tag = sample_key
    figure.suptitle(f"{split_name.upper()} | {tree_id} | {season} | {tag}")
    figure.tight_layout()
    output_path = OUTPUT_DIR / f"{split_name}_{tree_id}_{tag}.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_overlay_montage(selected, prepared_rgb, prepared_gt, key_to_index):
    """Save one labeled montage with overlays from all three splits."""
    columns = 5
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(17, 3.8 * rows))
    axes = np.asarray(axes).reshape(rows, columns)

    for axis, (split_name, sample_key) in zip(axes.flat, selected):
        index = key_to_index[sample_key]
        add_gt_overlay(axis, prepared_rgb[index], prepared_gt[index])
        tree_id, season, tag = sample_key
        axis.set_title(f"{split_name.upper()} | {tree_id}\n{season} {tag}", fontsize=8)
        axis.axis("off")

    for axis in axes.flat[len(selected):]:
        axis.axis("off")

    figure.suptitle("RGB/GT alignment check: current crop and resize assumption")
    figure.tight_layout()
    montage_path = OUTPUT_DIR / "rgb_gt_alignment_overlay_montage.png"
    figure.savefig(montage_path, dpi=160, bbox_inches="tight")
    print(f"Overlay montage saved to: {montage_path}")
    return figure


def main():
    """Create diagnostic figures only; do not calculate model performance."""
    if not baseline.DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {baseline.DATASET_DIR}")

    images = baseline.scan_images(baseline.DATASET_DIR)
    summer_pairs = baseline.find_summer_rgb_gt_pairs(images)
    if not summer_pairs:
        raise RuntimeError("No Summer RGB/GT sample pairs were found.")

    sample_keys = sorted(summer_pairs)
    tree_ids = np.asarray([key[0] for key in sample_keys])
    splits = baseline.split_by_tree(
        tree_ids, sample_keys, seed=baseline.RANDOM_SEED
    )
    baseline.verify_tree_split_isolation(splits)
    selected = select_representative_keys(splits)
    selected_keys = [sample_key for _, sample_key in selected]
    selected_pairs = {key: summer_pairs[key] for key in selected_keys}

    processed_rgb, targets, _, prepared_keys = baseline.prepare_rgb_gt_dataset(
        selected_pairs
    )
    if prepared_keys != sorted(selected_keys):
        raise AssertionError("Prepared sample order differs from the selected keys.")
    if processed_rgb.shape != (20, 256, 256, 3):
        raise AssertionError(f"Unexpected processed RGB array shape: {processed_rgb.shape}")
    if targets.shape != (20, 256, 256, 1):
        raise AssertionError(f"Unexpected GT array shape: {targets.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    key_to_index = {key: index for index, key in enumerate(prepared_keys)}
    for split_name, sample_key in selected:
        index = key_to_index[sample_key]
        save_sample_figure(
            split_name,
            sample_key,
            summer_pairs[sample_key],
            processed_rgb[index],
            targets[index, :, :, 0] >= 0.5,
        )

    montage_figure = save_overlay_montage(
        selected,
        processed_rgb,
        targets[:, :, :, 0] >= 0.5,
        key_to_index,
    )
    print(f"Samples visualized: {len(selected)}")
    for split_name, requested_count in SAMPLES_PER_SPLIT.items():
        print(f"{split_name.capitalize()} examples: {requested_count}")
    print(f"All figures saved under: {OUTPUT_DIR}")
    print("No model performance was calculated.")
    plt.show()


if __name__ == "__main__":
    main()