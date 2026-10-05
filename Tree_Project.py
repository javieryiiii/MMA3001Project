"""Shared data, preprocessing, split, model, and metric utilities.

This module provides Summer image pairing, RGB/GT preprocessing, deterministic
tree-level splitting, the RGB U-Net, and segmentation metrics. Its command-line
workflow evaluates the original RGB+BCE baseline checkpoint on the held-out
test split; reusable training helpers are also available below.
"""

from pathlib import Path
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


DATASET_DIR = Path(__file__).resolve().parent / "Dataset (With Summer GT)"
RANDOM_SEED = 42
TARGET_SIZE = (256, 256)
MAX_EPOCHS = 30
BATCH_SIZE = 8
LOSS_FUNCTION = "binary_crossentropy"
PREDICTION_THRESHOLD = 0.5
TEST_PREDICTION_THRESHOLD = 0.25
VALIDATION_THRESHOLDS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "rgb_unet_baseline.keras"
IMAGE_PATTERN = re.compile(
	r"^(?P<tree_id>R\d+[UN]\d+)_(?P<season>Summer|Winter)_"
	r"(?P<modality>RGB|D|GT)-(?P<tag>.+)\.png$"
)
SEARCH_TERMS = (
	"crop", "resize", "256", "ground truth", "ground_truth", "GT", "mask",
	"skeleton", "ROI", "bounding box", "annotation",
)
TEXT_SUFFIXES = {".md", ".py", ".ipynb", ".txt", ".json"}


def scan_images(dataset_dir):
	"""Index recognized PNG files in the flat dataset directory.

	Args:
		dataset_dir: Directory containing modality-tagged PNG files.

	Returns:
		A mapping from ``(tree_id, season, capture_tag)`` to modality/path
		mappings. Filenames that do not match ``IMAGE_PATTERN`` are ignored.
	"""
	images = {}
	for path in sorted(dataset_dir.glob("*.png")):
		match = IMAGE_PATTERN.match(path.name)
		if match is None:
			continue

		parts = match.groupdict()
		key = (parts["tree_id"], parts["season"], parts["tag"])
		images.setdefault(key, {})[parts["modality"]] = path
	return images


def find_summer_triplets(images):
	"""Separate complete Summer RGB/depth/GT groups from incomplete groups.

	Args:
		images: Mapping returned by :func:`scan_images`.

	Returns:
		A pair ``(complete, unmatched)``. ``complete`` maps sample keys to
		modalities containing RGB, D, and GT. ``unmatched`` maps each incomplete
		Summer key to a sorted list of missing modality names.
	"""
	summer = {
		key: modalities for key, modalities in images.items()
		if key[1] == "Summer"
	}
	required = {"RGB", "D", "GT"}
	complete = {
		key: modalities for key, modalities in summer.items()
		if required <= modalities.keys()
	}
	unmatched = {
		key: sorted(required - modalities.keys())
		for key, modalities in summer.items()
		if required - modalities.keys()
	}
	return complete, unmatched


def find_summer_rgb_gt_pairs(images):
	"""Select Summer samples that have both RGB and GT files.

	Args:
		images: Mapping returned by :func:`scan_images`.

	Returns:
		A mapping from matching sample keys to their original modality/path
		mappings. Other modalities, such as D, remain present in each mapping.
	"""
	return {
		key: modalities for key, modalities in images.items()
		if key[1] == "Summer" and {"RGB", "GT"} <= modalities.keys()
	}


def find_winter_pairs(images):
	"""Separate complete Winter RGB/depth pairs from incomplete groups.

	Args:
		images: Mapping returned by :func:`scan_images`.

	Returns:
		A pair ``(complete, unmatched)``. Complete groups contain RGB and D;
		unmatched groups map sample keys to sorted missing-modality names.
	"""
	winter = {
		key: modalities for key, modalities in images.items()
		if key[1] == "Winter"
	}
	required = {"RGB", "D"}
	complete = {
		key: modalities for key, modalities in winter.items()
		if required <= modalities.keys()
	}
	unmatched = {
		key: sorted(required - modalities.keys())
		for key, modalities in winter.items()
		if required - modalities.keys()
	}
	return complete, unmatched


def load_image(path, flags):
	"""Read an image with OpenCV.

	Args:
		path: Image path accepted by :func:`cv2.imread` after string conversion.
		flags: OpenCV read mode, such as ``cv2.IMREAD_COLOR``.

	Returns:
		The decoded image array in OpenCV channel order.

	Raises:
		RuntimeError: If OpenCV cannot decode the file.
	"""
	image = cv2.imread(str(path), flags)
	if image is None:
		raise RuntimeError(f"OpenCV could not read: {path}")
	return image


def describe_image(label, path, image):
	"""Print the filename, array properties, and value range of an image."""
	print(f"{label} filename: {path.name}")
	print(f"{label} shape: {image.shape}")
	print(f"{label} dtype: {image.dtype}")
	print(f"{label} value range: {image.min()} to {image.max()}")


def describe_gt(gt):
	"""Report the GT representation and its pixel values without converting it."""
	if gt.ndim == 2:
		representation = "grayscale"
	elif gt.ndim == 3 and gt.shape[2] == 3:
		representation = "RGB/BGR three-channel"
	else:
		representation = "another representation"

	unique_values = np.unique(gt)
	if unique_values.size <= 20:
		values = unique_values.tolist()
	else:
		values = {
			"number_of_unique_values": int(unique_values.size),
			"first_values": unique_values[:10].tolist(),
			"last_values": unique_values[-10:].tolist(),
		}
	print(f"GT representation: {representation}")
	print(f"GT unique/representative pixel values: {values}")


def search_repository(root):
	"""Search text files for evidence about GT generation or spatial transforms."""
	print("\nRepository provenance search")
	found = []
	for path in sorted(root.rglob("*")):
		if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
			continue
		if "Dataset" in path.parts:
			continue
		try:
			text = path.read_text(encoding="utf-8", errors="ignore")
		except OSError:
			continue
		matches = [term for term in SEARCH_TERMS if term.lower() in text.lower()]
		if matches:
			found.append((path, matches))

	if not found:
		print("No matching documentation or source-code evidence was found.")
	else:
		for path, matches in found:
			print(f"{path.relative_to(root)}: {', '.join(matches)}")


def get_shape_counts(file_groups, modality):
	"""Count image spatial shapes for one modality across all sample groups."""
	shape_counts = {}
	for modalities in file_groups.values():
		image = load_image(modalities[modality], cv2.IMREAD_UNCHANGED)
		shape = image.shape[:2]
		shape_counts[shape] = shape_counts.get(shape, 0) + 1
	return shape_counts


def depth_channels_are_identical(file_groups):
	"""Check whether every three-channel depth image has identical channels."""
	different_files = []
	for key, modalities in file_groups.items():
		depth = load_image(modalities["D"], cv2.IMREAD_UNCHANGED)
		if depth.ndim != 3 or depth.shape[2] != 3:
			different_files.append((key, "not three-channel"))
			continue
		if not (np.array_equal(depth[:, :, 0], depth[:, :, 1]) and
				np.array_equal(depth[:, :, 1], depth[:, :, 2])):
			different_files.append((key, "channels differ"))
	return different_files


def gt_white_percentages(summer_triplets):
	"""Calculate the percentage of 255-valued pixels in every Summer GT mask."""
	percentages = []
	for modalities in summer_triplets.values():
		gt = load_image(modalities["GT"], cv2.IMREAD_UNCHANGED)
		percentages.append(100.0 * np.count_nonzero(gt == 255) / gt.size)
	return np.array(percentages)


def print_shape_check(summer_triplets, winter_pairs):
	"""Print shape distributions for all required Summer and Winter images."""
	print("\nAll-image shape checks")
	print(f"Summer RGB shapes: {get_shape_counts(summer_triplets, 'RGB')}")
	print(f"Summer depth shapes: {get_shape_counts(summer_triplets, 'D')}")
	print(f"Summer GT shapes: {get_shape_counts(summer_triplets, 'GT')}")
	print(f"Winter RGB shapes: {get_shape_counts(winter_pairs, 'RGB')}")
	print(f"Winter depth shapes: {get_shape_counts(winter_pairs, 'D')}")


def display_alignment_hypothesis(summer_triplets, number_of_examples=6):
	"""Compare a centre-crop hypothesis with GT without changing original files."""
	keys = sorted(summer_triplets)
	indices = np.linspace(0, len(keys) - 1, number_of_examples, dtype=int)
	selected_keys = [keys[index] for index in indices]
	figure, axes = plt.subplots(number_of_examples, 5, figsize=(22, 4 * number_of_examples))
	if number_of_examples == 1:
		axes = np.expand_dims(axes, axis=0)

	for row, key in enumerate(selected_keys):
		modalities = summer_triplets[key]
		rgb = load_image(modalities["RGB"], cv2.IMREAD_COLOR)
		gt = load_image(modalities["GT"], cv2.IMREAD_UNCHANGED)
		if rgb.shape[:2] != (480, 640) or gt.shape[:2] != (256, 256):
			raise ValueError(f"Unexpected image dimensions for sample {key}: RGB {rgb.shape}, GT {gt.shape}")

		centre_crop = rgb[:, 80:560]
		resized_crop = cv2.resize(centre_crop, (256, 256), interpolation=cv2.INTER_AREA)
		original_rgb_display = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
		crop_display = cv2.cvtColor(centre_crop, cv2.COLOR_BGR2RGB)
		resized_display = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)

		axes[row, 0].imshow(original_rgb_display)
		axes[row, 0].set_title(f"Original RGB\n{modalities['RGB'].name}")
		axes[row, 1].imshow(crop_display)
		axes[row, 1].set_title("480x480 centre crop\nremove 80 px per side")
		axes[row, 2].imshow(resized_display)
		axes[row, 2].set_title("Crop resized to 256x256")
		axes[row, 3].imshow(gt, cmap="gray", vmin=0, vmax=255)
		axes[row, 3].set_title(f"Original GT\n{modalities['GT'].name}")
		axes[row, 4].imshow(resized_display)
		axes[row, 4].imshow(np.ma.masked_where(gt == 0, gt), cmap="autumn", alpha=0.55, vmin=0, vmax=255)
		axes[row, 4].set_title("GT overlay on resized crop")
		for axis in axes[row]:
			axis.axis("off")

	figure.suptitle(
		"Centre-crop alignment hypothesis only; visual comparison does not verify the transformation"
	)
	figure.tight_layout()


def prepare_rgb_gt_dataset(summer_pairs):
	"""Prepare matched Summer RGB/GT samples using the established RGB transform.

	RGB files must be 480x640 and GT masks must be binary 256x256 PNGs with
	values 0 or 255. RGB is converted from BGR to RGB, cropped with columns
	80:560, resized to 256x256 using ``INTER_AREA``, and divided by 255. GT
	geometry is unchanged and values are converted to 0/1.

	Args:
		summer_pairs: Mapping from sample keys to modality/path mappings with RGB
			and GT entries.

	Returns:
		A tuple ``(inputs, targets, tree_ids, sample_keys)``. Inputs have shape
		``(N, 256, 256, 3)`` and targets ``(N, 256, 256, 1)``, both float32.
		RGB values are in [0, 1]; targets are binary. IDs and keys follow sorted
		sample-key order.

	Raises:
		ValueError: If source RGB/GT dimensions or GT values violate the expected
			input format.
	"""
	sample_keys = sorted(summer_pairs)
	inputs = np.empty((len(sample_keys), 256, 256, 3), dtype=np.float32)
	targets = np.empty((len(sample_keys), 256, 256, 1), dtype=np.float32)
	tree_ids = []

	for index, key in enumerate(sample_keys):
		modalities = summer_pairs[key]
		rgb_bgr = load_image(modalities["RGB"], cv2.IMREAD_COLOR)
		gt = load_image(modalities["GT"], cv2.IMREAD_GRAYSCALE)
		if rgb_bgr.shape[:2] != (480, 640):
			raise ValueError(f"Unexpected RGB dimensions for {modalities['RGB'].name}: {rgb_bgr.shape}")
		if gt.shape != TARGET_SIZE:
			raise ValueError(f"Unexpected GT dimensions for {modalities['GT'].name}: {gt.shape}")
		if not np.isin(np.unique(gt), [0, 255]).all():
			raise ValueError(f"GT mask is not binary (0/255): {modalities['GT'].name}")

		rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
		centre_crop = rgb[:, 80:560]
		resized_rgb = cv2.resize(centre_crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
		inputs[index] = resized_rgb.astype(np.float32) / 255.0
		targets[index, :, :, 0] = (gt == 255).astype(np.float32)
		tree_ids.append(key[0])

	return inputs, targets, np.array(tree_ids), sample_keys


def split_by_tree(tree_ids, sample_keys, seed=RANDOM_SEED):
	"""Assign samples to deterministic splits while keeping each tree intact.

	Unique tree IDs are sorted by ``np.unique`` and permuted by a NumPy generator
	initialized with ``seed``. The first floor(70%) of trees are training, the
	next floor(15%) are validation, and the remainder are test. Sample order
	within each returned index/key list follows the input order.

	Args:
		tree_ids: One tree ID per sample.
		sample_keys: One corresponding sample key per sample.
		seed: Seed used to permute the unique tree IDs.

	Returns:
		A mapping for ``train``, ``validation``, and ``test``. Each entry contains
		an int32 index array, a set of assigned tree IDs, and the corresponding
		sample keys.
	"""
	unique_tree_ids = np.unique(tree_ids)
	shuffled_tree_ids = np.random.default_rng(seed).permutation(unique_tree_ids)
	train_tree_count = int(len(unique_tree_ids) * 0.70)
	validation_tree_count = int(len(unique_tree_ids) * 0.15)

	train_trees = set(shuffled_tree_ids[:train_tree_count])
	validation_trees = set(
		shuffled_tree_ids[train_tree_count:train_tree_count + validation_tree_count]
	)
	test_trees = set(shuffled_tree_ids[train_tree_count + validation_tree_count:])
	assignments = {
		"train": train_trees,
		"validation": validation_trees,
		"test": test_trees,
	}

	splits = {}
	for split_name, split_trees in assignments.items():
		indices = np.array(
			[index for index, tree_id in enumerate(tree_ids) if tree_id in split_trees],
			dtype=np.int32,
		)
		splits[split_name] = {
			"indices": indices,
			"tree_ids": split_trees,
			"sample_keys": [sample_keys[index] for index in indices],
		}

	return splits


def verify_tree_split_isolation(splits):
	"""Check that tree IDs are not assigned to multiple split groups.

	Args:
		splits: Mapping with ``train``, ``validation``, and ``test`` entries, each
		containing a ``tree_ids`` set.

	Raises:
		AssertionError: If any pair of split tree-ID sets overlaps or if the
			union size differs from the sum of their sizes, indicating duplicate
			assignment. This check does not verify expected split sizes or missing IDs.
	"""
	train_trees = splits["train"]["tree_ids"]
	validation_trees = splits["validation"]["tree_ids"]
	test_trees = splits["test"]["tree_ids"]
	if train_trees & validation_trees or train_trees & test_trees or validation_trees & test_trees:
		raise AssertionError("Tree identity overlap detected between dataset splits.")
	all_split_trees = train_trees | validation_trees | test_trees
	if len(all_split_trees) != sum(len(splits[name]["tree_ids"]) for name in splits):
		raise AssertionError("Tree IDs are duplicated across dataset splits.")


def report_prepared_dataset(inputs, targets, splits, tree_ids):
	"""Print dataset array properties, split sizes, ranges, and leakage check."""
	print("\nPrepared RGB-only dataset")
	print(f"Unique trees: {len(np.unique(tree_ids))}")
	print(f"Images: {len(inputs)}")
	for split_name, split in splits.items():
		print(
			f"{split_name.capitalize()}: {len(split['tree_ids'])} trees, "
			f"{len(split['indices'])} images"
		)
	print(f"RGB input shape/dtype: {inputs.shape[1:]}, {inputs.dtype}")
	print(f"GT target shape/dtype: {targets.shape[1:]}, {targets.dtype}")
	print(f"RGB normalized value range: {inputs.min():.6f} to {inputs.max():.6f}")
	print(f"GT unique values: {np.unique(targets).tolist()}")
	verify_tree_split_isolation(splits)
	print("Tree ID overlap between train/validation/test: 0")


def display_processed_split_examples(inputs, targets, splits, examples_per_split=3):
	"""Display processed RGB and target examples from each reproducible split."""
	for split_name, split in splits.items():
		indices = split["indices"][:examples_per_split]
		figure, axes = plt.subplots(len(indices), 2, figsize=(8, 3 * len(indices)))
		if len(indices) == 1:
			axes = np.expand_dims(axes, axis=0)
		for row, index in enumerate(indices):
			axes[row, 0].imshow(inputs[index])
			axes[row, 0].set_title(f"{split_name.capitalize()} RGB: {split['sample_keys'][row][0]}")
			axes[row, 1].imshow(targets[index, :, :, 0], cmap="gray", vmin=0, vmax=1)
			axes[row, 1].set_title("Binary GT")
			for axis in axes[row]:
				axis.axis("off")
		figure.suptitle(f"Processed samples from {split_name} split")
		figure.tight_layout()


def convolution_block(input_tensor, filters):
	"""Apply two same-padded 3x3 ReLU convolutions.

	Args:
		input_tensor: Keras feature tensor.
		filters: Number of output channels for each convolution.

	Returns:
		Feature tensor with the same spatial dimensions and ``filters`` channels.
	"""
	output = tf.keras.layers.Conv2D(
		filters, 3, activation="relu", padding="same", kernel_initializer="he_normal"
	)(input_tensor)
	output = tf.keras.layers.Conv2D(
		filters, 3, activation="relu", padding="same", kernel_initializer="he_normal"
	)(output)
	return output


def build_unet(input_shape=(256, 256, 3), base_filters=16):
	"""Build and compile the project U-Net with skip-connected encoder/decoder stages.

	Args:
		input_shape: Height, width, and channel count. The project RGB models use
			``(256, 256, 3)``; the RGB-D model supplies four channels.
		base_filters: Width of the first encoder stage; subsequent stages scale
			this value by powers of two.
		Input height and width must be divisible by 16 for the four pooling stages
		and skip connections to align.

	Returns:
		A compiled Keras model with a one-channel sigmoid probability output of
		the same spatial size as the input.
	"""
	inputs = tf.keras.Input(shape=input_shape)
	filters = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8]
	encoder_features = []
	output = inputs

	# The encoder pools spatial dimensions while increasing feature channels.
	for filter_count in filters:
		output = convolution_block(output, filter_count)
		encoder_features.append(output)
		output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(output)

	# The bottleneck captures the most compressed, high-level representation.
	output = convolution_block(output, base_filters * 16)
	# Upsampling plus skip connections recover detail lost during pooling.
	for filter_count, skip_features in zip(reversed(filters), reversed(encoder_features)):
		output = tf.keras.layers.Conv2DTranspose(
			filter_count, 2, strides=2, padding="same"
		)(output)
		output = tf.keras.layers.Concatenate()([output, skip_features])
		output = convolution_block(output, filter_count)

	outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="skeleton_probability")(output)
	model = tf.keras.Model(inputs=inputs, outputs=outputs, name="rgb_baseline_unet")
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
		# Keep BCE easy to replace with Dice or a combined loss in a later stage.
		loss=LOSS_FUNCTION,
		metrics=[dice_coefficient, intersection_over_union],
	)
	return model


def dice_coefficient(y_true, y_pred):
	"""Compute mean soft Dice over a batch of segmentation masks.

	Args:
		y_true: Ground-truth tensor shaped ``(N, H, W, C)``.
		y_pred: Soft prediction probabilities with the same shape.

	Returns:
		Scalar batch mean of per-image Dice values, using ``1e-6`` smoothing in
		the numerator and denominator.
	"""
	y_true = tf.cast(y_true, tf.float32)
	y_pred = tf.cast(y_pred, tf.float32)
	intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
	mask_sum = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
	dice = (2.0 * intersection + 1e-6) / (mask_sum + 1e-6)
	return tf.reduce_mean(dice)


def intersection_over_union(y_true, y_pred):
	"""Compute mean soft intersection over union (IoU) over a batch.

	Args:
		y_true: Ground-truth tensor shaped ``(N, H, W, C)``.
		y_pred: Soft prediction probabilities with the same shape.

	Returns:
		Scalar batch mean of per-image IoU values, using ``1e-6`` smoothing in
		the numerator and denominator.
	"""
	y_true = tf.cast(y_true, tf.float32)
	y_pred = tf.cast(y_pred, tf.float32)
	intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
	union = tf.reduce_sum(y_true + y_pred - y_true * y_pred, axis=(1, 2, 3))
	iou = (intersection + 1e-6) / (union + 1e-6)
	return tf.reduce_mean(iou)


def binary_dice_coefficient(y_true, y_pred_binary):
	"""Calculate mean per-image binary Dice for a batch of masks.

	Ground truth is binarized with ``>= 0.5`` and predictions are converted to
	boolean. All non-batch axes are reduced independently per image.

	Args:
		y_true: Ground-truth array with shape ``(N, ...)``.
		y_pred_binary: Binary prediction array with the same shape.

	Returns:
		Python float containing the mean per-image Dice; ``1e-6`` smoothing keeps
		empty-mask comparisons defined.

	Raises:
		ValueError: If the input shapes differ.
	"""
	y_true_binary = np.asarray(y_true) >= 0.5
	y_pred_binary = np.asarray(y_pred_binary, dtype=bool)
	if y_true_binary.shape != y_pred_binary.shape:
		raise ValueError("Ground-truth and binary prediction shapes must match.")
	image_axes = tuple(range(1, y_true_binary.ndim))
	intersection = np.sum(y_true_binary & y_pred_binary, axis=image_axes)
	foreground_sum = np.sum(y_true_binary, axis=image_axes) + np.sum(y_pred_binary, axis=image_axes)
	return float(np.mean((2.0 * intersection + 1e-6) / (foreground_sum + 1e-6)))


def binary_intersection_over_union(y_true, y_pred_binary):
	"""Calculate mean per-image binary IoU for a batch of masks.

	Ground truth is binarized with ``>= 0.5`` and predictions are converted to
	boolean. All non-batch axes are reduced independently per image.

	Args:
		y_true: Ground-truth array with shape ``(N, ...)``.
		y_pred_binary: Binary prediction array with the same shape.

	Returns:
		Python float containing the mean per-image IoU, with ``1e-6`` smoothing.

	Raises:
		ValueError: If the input shapes differ.
	"""
	y_true_binary = np.asarray(y_true) >= 0.5
	y_pred_binary = np.asarray(y_pred_binary, dtype=bool)
	if y_true_binary.shape != y_pred_binary.shape:
		raise ValueError("Ground-truth and binary prediction shapes must match.")
	image_axes = tuple(range(1, y_true_binary.ndim))
	intersection = np.sum(y_true_binary & y_pred_binary, axis=image_axes)
	union = np.sum(y_true_binary | y_pred_binary, axis=image_axes)
	return float(np.mean((intersection + 1e-6) / (union + 1e-6)))


def report_validation_probability_stats(probabilities, sample_keys, number_of_examples=5):
	"""Report overall and representative per-image probability-map statistics."""
	flat_probabilities = probabilities.ravel()
	threshold_percentage = 100.0 * np.mean(flat_probabilities >= PREDICTION_THRESHOLD)
	print("\nValidation probability-map summary")
	print(f"Minimum: {flat_probabilities.min():.6f}")
	print(f"Maximum: {flat_probabilities.max():.6f}")
	print(f"Mean: {flat_probabilities.mean():.6f}")
	print(f"Median: {np.median(flat_probabilities):.6f}")
	print(f"Pixels >= {PREDICTION_THRESHOLD}: {threshold_percentage:.4f}%")

	selected_indices = np.linspace(
		0, len(probabilities) - 1, min(number_of_examples, len(probabilities)), dtype=int
	)
	print(f"\nPer-image probability statistics ({len(selected_indices)} validation images)")
	for index in selected_indices:
		image_probabilities = probabilities[index].ravel()
		image_key = sample_keys[index]
		image_threshold_percentage = 100.0 * np.mean(
			image_probabilities >= PREDICTION_THRESHOLD
		)
		print(
			f"{image_key[0]} tag={image_key[2]}: min={image_probabilities.min():.6f}, "
			f"max={image_probabilities.max():.6f}, mean={image_probabilities.mean():.6f}, "
			f"median={np.median(image_probabilities):.6f}, "
			f">={PREDICTION_THRESHOLD}={image_threshold_percentage:.4f}%"
		)


def score_validation_thresholds(y_true, probabilities, thresholds=VALIDATION_THRESHOLDS):
	"""Score binary masks at each supplied probability threshold.

	Predictions use ``probabilities >= threshold`` and the binary metric helpers
	return mean per-image Dice and IoU. No threshold is selected by this function.

	Args:
		y_true: Ground-truth array shaped ``(N, H, W, C)``.
		probabilities: Soft predictions with the same shape.
		thresholds: Iterable of probability cutoffs to evaluate.

	Returns:
		List of dictionaries, one per threshold, with ``threshold``, ``dice``,
		and ``iou`` keys.
	"""
	results = []
	for threshold in thresholds:
		predicted_binary = probabilities >= threshold
		results.append({
			"threshold": threshold,
			"dice": binary_dice_coefficient(y_true, predicted_binary),
			"iou": binary_intersection_over_union(y_true, predicted_binary),
		})
	return results


def plot_validation_threshold_metrics(results):
	"""Plot validation binary Dice and IoU across the tested thresholds."""
	thresholds = [result["threshold"] for result in results]
	figure, axis = plt.subplots(figsize=(8, 5))
	axis.plot(thresholds, [result["dice"] for result in results], marker="o", label="Binary Dice")
	axis.plot(thresholds, [result["iou"] for result in results], marker="s", label="Binary IoU")
	axis.set_xlabel("Probability threshold")
	axis.set_ylabel("Mean per-image score")
	axis.set_title("Validation segmentation metrics by threshold")
	axis.set_xticks(thresholds)
	axis.grid(True, alpha=0.3)
	axis.legend()
	figure.tight_layout()


def train_model(model, train_inputs, train_targets, validation_inputs, validation_targets):
	"""Fit on training data, monitor validation loss, and save the best checkpoint."""
	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	callbacks = [
		tf.keras.callbacks.EarlyStopping(
			monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
		),
		tf.keras.callbacks.ModelCheckpoint(
			filepath=MODEL_PATH,
			monitor="val_loss",
			save_best_only=True,
			verbose=1,
		),
	]
	return model.fit(
		train_inputs,
		train_targets,
		validation_data=(validation_inputs, validation_targets),
		epochs=MAX_EPOCHS,
		batch_size=BATCH_SIZE,
		callbacks=callbacks,
		verbose=1,
	)


def plot_training_history(history):
	"""Plot training and validation BCE loss, Dice, and IoU by epoch."""
	metric_names = ["loss", "dice_coefficient", "intersection_over_union"]
	figure, axes = plt.subplots(1, len(metric_names), figsize=(15, 4))
	for axis, metric_name in zip(axes, metric_names):
		axis.plot(history.history[metric_name], label="Training")
		axis.plot(history.history[f"val_{metric_name}"], label="Validation")
		axis.set_title(metric_name.replace("_", " ").title())
		axis.set_xlabel("Epoch")
		axis.set_ylabel(metric_name.replace("_", " ").title())
		axis.legend()
		axis.grid(True, alpha=0.3)
	figure.tight_layout()


def evaluate_test_set(model, test_inputs, test_targets):
	"""Evaluate the restored best model once on the held-out test samples."""
	return model.evaluate(test_inputs, test_targets, batch_size=BATCH_SIZE, return_dict=True, verbose=1)


def display_test_predictions(test_inputs, test_targets, predicted_probabilities, sample_keys, threshold=TEST_PREDICTION_THRESHOLD, number_of_examples=4):
	"""Show test RGB, GT, probabilities, and validation-threshold predictions."""
	if len(test_inputs) == 0:
		raise ValueError("The test split is empty.")
	selected_indices = np.linspace(
		0, len(test_inputs) - 1, min(number_of_examples, len(test_inputs)), dtype=int
	)
	figure, axes = plt.subplots(len(selected_indices), 4, figsize=(14, 3.5 * len(selected_indices)))
	if len(selected_indices) == 1:
		axes = np.expand_dims(axes, axis=0)

	for row, sample_index in enumerate(selected_indices):
		probability_map = predicted_probabilities[sample_index, :, :, 0]
		predicted_mask = probability_map >= threshold
		axes[row, 0].imshow(test_inputs[sample_index])
		axes[row, 0].set_title(f"RGB: {sample_keys[sample_index][0]}")
		axes[row, 1].imshow(test_targets[sample_index, :, :, 0], cmap="gray", vmin=0, vmax=1)
		axes[row, 1].set_title("Ground truth")
		axes[row, 2].imshow(probability_map, cmap="viridis", vmin=0, vmax=1)
		axes[row, 2].set_title("Predicted probability")
		axes[row, 3].imshow(predicted_mask, cmap="gray", vmin=0, vmax=1)
		axes[row, 3].set_title(f"Predicted mask (threshold {threshold})")
		for axis in axes[row]:
			axis.axis("off")
	figure.suptitle(f"Held-out test predictions; validation-selected threshold {threshold}")
	figure.tight_layout()


def display_winter(rgb, depth, title):
	"""Display one Winter RGB-depth pair side-by-side."""
	figure, axes = plt.subplots(1, 2, figsize=(10, 5))
	axes[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
	axes[0].set_title("Winter RGB")
	axes[1].imshow(depth, cmap="viridis")
	axes[1].set_title("Winter depth")
	for axis in axes:
		axis.axis("off")
	figure.suptitle(title)
	figure.tight_layout()


def print_unmatched(label, unmatched):
	"""Print every incomplete group and identify the missing modalities."""
	print(f"{label} unmatched groups: {len(unmatched)}")
	for key, missing in unmatched.items():
		print(f"  {key}: missing {', '.join(missing)}")


def main():
	"""Evaluate the saved baseline once on test data at the selected threshold."""
	if not DATASET_DIR.is_dir():
		raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")
	if not MODEL_PATH.is_file():
		raise FileNotFoundError(f"Trained model checkpoint not found: {MODEL_PATH}")

	print(f"Dataset folder: {DATASET_DIR}")
	images = scan_images(DATASET_DIR)
	summer_pairs = find_summer_rgb_gt_pairs(images)
	print(f"Complete Summer RGB/GT pairs: {len(summer_pairs)}")
	if not summer_pairs:
		raise RuntimeError("At least one complete Summer RGB/GT pair is required.")

	print("RGB preprocessing uses the assumed 480x480 centre crop, resized to 256x256.")
	print("GT masks remain at their original dimensions and are converted from 0/255 to 0/1.")
	inputs, targets, tree_ids, sample_keys = prepare_rgb_gt_dataset(summer_pairs)
	splits = split_by_tree(tree_ids, sample_keys)
	test_indices = splits["test"]["indices"]
	test_inputs = inputs[test_indices]
	test_targets = targets[test_indices]
	test_sample_keys = [sample_keys[index] for index in test_indices]
	print(f"Held-out test images: {len(test_inputs)}")
	print(
		"Final RGB-only BCE baseline test performance using a validation-selected "
		f"threshold of {TEST_PREDICTION_THRESHOLD:.2f}"
	)
	print("Loading the existing best checkpoint for inference only; no training or threshold sweep.")
	model = tf.keras.models.load_model(MODEL_PATH, compile=False)
	probabilities = model.predict(test_inputs, batch_size=BATCH_SIZE, verbose=1)
	binary_predictions = probabilities >= TEST_PREDICTION_THRESHOLD
	binary_dice = binary_dice_coefficient(test_targets, binary_predictions)
	binary_iou = binary_intersection_over_union(test_targets, binary_predictions)
	print("\nFinal test metrics (mean per-image binary segmentation scores)")
	print(f"Binary Dice at threshold {TEST_PREDICTION_THRESHOLD:.2f}: {binary_dice:.6f}")
	print(f"Binary IoU at threshold {TEST_PREDICTION_THRESHOLD:.2f}: {binary_iou:.6f}")
	display_test_predictions(
		test_inputs,
		test_targets,
		probabilities,
		test_sample_keys,
		threshold=TEST_PREDICTION_THRESHOLD,
	)
	plt.show()


if __name__ == "__main__":
	main()
