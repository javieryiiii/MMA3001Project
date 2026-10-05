"""Verify Summer RGB-D-GT triplets and inspect their prepared array format.

This diagnostic checks source image shapes, types, depth-channel equality,
source-scale depth statistics, and seed-42 split correspondence. Its prepared
depth channel remains on the source numeric scale; it does not infer physical
units or train a model.
"""

from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import Tree_Project as baseline


OUTPUT_IMAGE_PATH = Path(__file__).resolve().parent / "rgbd_preparation_examples.png"
DEPTH_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def new_image_summary():
    """Create empty shape/type counters and aggregate minimum/maximum fields.

    Returns:
        Mutable summary dictionary populated by :func:`record_image`.
    """
    return {
        "shapes": Counter(),
        "dtypes": Counter(),
        "minimum": None,
        "maximum": None,
    }


def record_image(summary, image):
    """Update a summary in place with one image's shape, dtype, and value range.

    Args:
        summary: Dictionary returned by :func:`new_image_summary`.
        image: NumPy image array to add to the aggregate.
    """
    summary["shapes"][image.shape] += 1
    summary["dtypes"][str(image.dtype)] += 1
    image_minimum = float(np.min(image))
    image_maximum = float(np.max(image))
    if summary["minimum"] is None or image_minimum < summary["minimum"]:
        summary["minimum"] = image_minimum
    if summary["maximum"] is None or image_maximum > summary["maximum"]:
        summary["maximum"] = image_maximum


def report_image_summary(name, summary):
    """Print the aggregated source properties stored in an image summary.

    Args:
        name: Modality label displayed in the report.
        summary: Populated summary dictionary.
    """
    print(f"{name} original shapes: {dict(summary['shapes'])}")
    print(f"{name} dtype counts: {dict(summary['dtypes'])}")
    print(
        f"{name} value range across complete triplets: "
        f"{summary['minimum']} to {summary['maximum']}"
    )


def inspect_triplets(triplets):
    """Report source properties and validate depth channels and image sizes.

    Args:
        triplets: Mapping from sample keys to RGB/D/GT paths.

    Raises:
        ValueError: If a depth image has unsupported or nonidentical channels,
            or if RGB/depth/GT spatial sizes differ from the project assumptions.
    """
    summaries = {
        modality: new_image_summary() for modality in ("RGB", "D", "GT")
    }
    depth_values = []
    depth_channel_counts = Counter()
    nonidentical_depth_files = []
    invalid_depth_files = []
    invalid_spatial_shapes = []

    for key, modalities in sorted(triplets.items()):
        rgb = baseline.load_image(modalities["RGB"], cv2.IMREAD_UNCHANGED)
        depth = baseline.load_image(modalities["D"], cv2.IMREAD_UNCHANGED)
        gt = baseline.load_image(modalities["GT"], cv2.IMREAD_UNCHANGED)
        record_image(summaries["RGB"], rgb)
        record_image(summaries["D"], depth)
        record_image(summaries["GT"], gt)

        for modality, image in (("RGB", rgb), ("D", depth)):
            if image.shape[:2] != (480, 640):
                invalid_spatial_shapes.append(
                    (modalities[modality].name, image.shape)
                )
        if gt.shape[:2] != baseline.TARGET_SIZE:
            invalid_spatial_shapes.append((modalities["GT"].name, gt.shape))

        if depth.ndim == 2:
            depth_plane = depth
            depth_channel_counts["single-channel"] += 1
        elif depth.ndim == 3 and depth.shape[2] == 3:
            if np.array_equal(depth[:, :, 0], depth[:, :, 1]) and np.array_equal(
                depth[:, :, 1], depth[:, :, 2]
            ):
                depth_plane = depth[:, :, 0]
                depth_channel_counts["identical three-channel"] += 1
            else:
                nonidentical_depth_files.append(modalities["D"].name)
                continue
        else:
            invalid_depth_files.append((modalities["D"].name, depth.shape))
            continue

        depth_values.append(depth_plane.reshape(-1))

    print("\nOriginal image properties across complete Summer triplets")
    for modality in ("RGB", "D", "GT"):
        report_image_summary(modality, summaries[modality])

    print("\nDepth channel verification")
    print(f"Depth files checked: {sum(depth_channel_counts.values()) + len(nonidentical_depth_files) + len(invalid_depth_files)}")
    print(f"Depth channel counts: {dict(depth_channel_counts)}")
    print(f"Files with nonidentical depth channels: {len(nonidentical_depth_files)}")
    for filename in nonidentical_depth_files:
        print(f"  {filename}")
    print(f"Files with unsupported depth shape: {len(invalid_depth_files)}")
    for filename, shape in invalid_depth_files:
        print(f"  {filename}: {shape}")

    if depth_values:
        all_depth_values = np.concatenate(depth_values)
        percentiles = np.percentile(all_depth_values, DEPTH_PERCENTILES)
        print("\nRaw depth distribution (one channel per pixel; source scale retained)")
        print(f"Minimum: {float(np.min(all_depth_values)):.6f}")
        print(f"Maximum: {float(np.max(all_depth_values)):.6f}")
        print(f"Mean: {float(np.mean(all_depth_values, dtype=np.float64)):.6f}")
        print(f"Median: {float(np.median(all_depth_values)):.6f}")
        for percentile, value in zip(DEPTH_PERCENTILES, percentiles):
            print(f"{percentile}th percentile: {float(value):.6f}")

    if nonidentical_depth_files:
        raise ValueError(
            "Depth channels are not identical in every file; refusing to choose "
            "a channel silently. See the files listed above."
        )
    if invalid_depth_files:
        raise ValueError("Depth images must be single-channel or three-channel PNGs.")
    if invalid_spatial_shapes:
        print("\nUnexpected original spatial shapes")
        for filename, shape in invalid_spatial_shapes:
            print(f"  {filename}: {shape}")
        raise ValueError("Images do not match the baseline spatial preprocessing assumption.")


def prepare_rgbd_inputs(triplets):
    """Append one source-scale depth channel to baseline-preprocessed RGB.

    RGB is processed by ``baseline.prepare_rgb_gt_dataset``. Depth is reduced
    to its first channel when stored as a three-channel image, cropped using
    columns 80:560, and resized to 256x256 with ``INTER_AREA``. This helper does
    not divide depth by 255; returned depth values remain on the source scale.
    The main diagnostic calls :func:`inspect_triplets` to check source channel
    equality before preparing the arrays.

    Args:
        triplets: Mapping from sample keys to RGB/D/GT paths.

    Returns:
        A tuple ``(rgbd_inputs, targets, tree_ids, sample_keys)``. RGB-D inputs
        have shape ``(N, 256, 256, 4)`` with RGB in channels 0:3 and source-scale
        depth in channel 3; targets have shape ``(N, 256, 256, 1)``.
    """
    rgb_inputs, targets, tree_ids, sample_keys = baseline.prepare_rgb_gt_dataset(
        triplets
    )
    rgbd_inputs = np.empty(
        (len(sample_keys), 256, 256, 4), dtype=np.float32
    )
    rgbd_inputs[:, :, :, :3] = rgb_inputs

    for index, key in enumerate(sample_keys):
        depth = baseline.load_image(triplets[key]["D"], cv2.IMREAD_UNCHANGED)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth_crop = depth[:, 80:560]
        depth_resized = cv2.resize(
            depth_crop, baseline.TARGET_SIZE, interpolation=cv2.INTER_AREA
        )
        rgbd_inputs[index, :, :, 3] = depth_resized.astype(np.float32)

    return rgbd_inputs, targets, tree_ids, sample_keys


def report_split_comparison(reference_splits, rgbd_sample_keys):
    """Print RGB-D sample and tree counts against the reference tree split.

    Args:
        reference_splits: Split mapping returned by ``baseline.split_by_tree``.
        rgbd_sample_keys: Sample keys prepared for RGB-D data.

    The report includes per-split counts, tree-set matches, and exact sample-key
    coverage. Results are printed; this helper does not raise on a mismatch.
    """
    rgbd_tree_ids = np.array([key[0] for key in rgbd_sample_keys])
    all_counts_match = True

    print("\nRGB baseline versus RGB-D split counts")
    print("Split counts use the baseline's original seed-42 tree assignments.")
    for split_name, reference_split in reference_splits.items():
        expected_tree_ids = reference_split["tree_ids"]
        rgbd_indices = np.array(
            [index for index, tree_id in enumerate(rgbd_tree_ids) if tree_id in expected_tree_ids],
            dtype=np.int32,
        )
        rgbd_trees = set(rgbd_tree_ids[rgbd_indices])
        expected_samples = len(reference_split["sample_keys"])
        prepared_samples = len(rgbd_indices)
        trees_match = rgbd_trees == expected_tree_ids
        counts_match = prepared_samples == expected_samples and trees_match
        all_counts_match &= counts_match
        print(
            f"{split_name.capitalize()}: baseline {expected_samples} samples / "
            f"{len(expected_tree_ids)} trees; RGB-D {prepared_samples} samples / "
            f"{len(rgbd_trees)} trees; match: {counts_match}"
        )

    reference_keys = {
        key for split in reference_splits.values() for key in split["sample_keys"]
    }
    exact_sample_match = set(rgbd_sample_keys) == reference_keys
    print(f"All RGB-D sample keys match the baseline RGB/GT set: {exact_sample_match}")
    print(f"All split counts and tree IDs match the baseline: {all_counts_match}")


def display_examples(rgbd_inputs, targets, sample_keys, train_tree_ids):
    """Save RGB, source-scale depth, and GT examples from training trees only.

    Args:
        rgbd_inputs: Prepared ``(N, 256, 256, 4)`` RGB-D array.
        targets: Prepared ``(N, 256, 256, 1)`` binary GT array.
        sample_keys: Key corresponding to each prepared sample.
        train_tree_ids: Tree IDs whose samples may be displayed.

    Raises:
        ValueError: If no samples belong to the requested training trees.
    """
    indices = [
        index for index, key in enumerate(sample_keys)
        if key[0] in train_tree_ids
    ]
    if not indices:
        raise ValueError("No complete RGB-D training examples are available to display.")

    selected = np.linspace(0, len(indices) - 1, min(4, len(indices)), dtype=int)
    selected_indices = [indices[index] for index in selected]
    figure, axes = plt.subplots(
        len(selected_indices), 3, figsize=(11, 3.5 * len(selected_indices))
    )
    if len(selected_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, index in enumerate(selected_indices):
        axes[row, 0].imshow(rgbd_inputs[index, :, :, :3])
        axes[row, 0].set_title(f"RGB: {sample_keys[index][0]}")
        axes[row, 1].imshow(rgbd_inputs[index, :, :, 3], cmap="viridis")
        axes[row, 1].set_title("Depth (source scale)")
        axes[row, 2].imshow(targets[index, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("Ground-truth skeleton")
        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle("RGB-D preparation examples from the training split")
    figure.tight_layout()
    figure.savefig(OUTPUT_IMAGE_PATH, dpi=150, bbox_inches="tight")
    print(f"\nExample figure saved to: {OUTPUT_IMAGE_PATH}")


def main():
    """Run the RGB-D preparation checks and display training-split examples."""
    if not baseline.DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {baseline.DATASET_DIR}")

    images = baseline.scan_images(baseline.DATASET_DIR)
    triplets, unmatched = baseline.find_summer_triplets(images)
    print(f"Dataset folder: {baseline.DATASET_DIR}")
    print(f"Complete Summer RGB-D-GT triplets: {len(triplets)}")
    missing_file_count = sum(len(missing) for missing in unmatched.values())
    print(f"Incomplete Summer groups: {len(unmatched)}")
    print(f"Missing/unmatched files: {missing_file_count}")
    for key, missing in unmatched.items():
        print(f"  {key}: missing {', '.join(missing)}")
    if not triplets:
        raise RuntimeError("No complete Summer RGB-D-GT triplets were found.")

    reference_pairs = baseline.find_summer_rgb_gt_pairs(images)
    reference_keys = sorted(reference_pairs)
    reference_tree_ids = np.array([key[0] for key in reference_keys])
    reference_splits = baseline.split_by_tree(
        reference_tree_ids, reference_keys, seed=42
    )
    baseline.verify_tree_split_isolation(reference_splits)

    inspect_triplets(triplets)
    rgbd_inputs, targets, tree_ids, sample_keys = prepare_rgbd_inputs(triplets)
    print("\nPrepared RGB-D arrays")
    print(f"RGB-D input shape/dtype: {rgbd_inputs.shape}, {rgbd_inputs.dtype}")
    print(f"GT target shape/dtype: {targets.shape}, {targets.dtype}")
    print(f"RGB normalized range: {rgbd_inputs[:, :, :, :3].min():.6f} to {rgbd_inputs[:, :, :, :3].max():.6f}")
    print(
        "Depth channel range after spatial resize (not normalized): "
        f"{rgbd_inputs[:, :, :, 3].min():.6f} to "
        f"{rgbd_inputs[:, :, :, 3].max():.6f}"
    )
    print(f"GT unique values: {np.unique(targets).tolist()}")
    report_split_comparison(reference_splits, sample_keys)

    train_tree_ids = reference_splits["train"]["tree_ids"]
    display_examples(rgbd_inputs, targets, sample_keys, train_tree_ids)
    plt.show()


if __name__ == "__main__":
    main()