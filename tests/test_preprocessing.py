import cv2
import numpy as np
import pytest

import Tree_Project as baseline
import Tree_Project_RGBD_BCE_Dice as rgbd_experiment


SAMPLE_KEY = ("R01N01", "Summer", "synthetic-sample")


def write_rgb_gt_pair(directory, rgb_shape=(480, 640, 3)):
    rgb_path = directory / "R01N01_Summer_RGB-synthetic-sample.png"
    gt_path = directory / "R01N01_Summer_GT-synthetic-sample.png"
    rgb_bgr = np.zeros(rgb_shape, dtype=np.uint8)
    if rgb_shape[:2] == (480, 640):
        rgb_bgr[:, 80:560, :] = (10, 20, 30)
    gt = np.zeros((256, 256), dtype=np.uint8)
    gt[::2, ::2] = 255
    assert cv2.imwrite(str(rgb_path), rgb_bgr)
    assert cv2.imwrite(str(gt_path), gt)
    return {SAMPLE_KEY: {"RGB": rgb_path, "GT": gt_path}}, gt


def test_rgb_gt_preprocessing_shape_crop_normalization_and_binary_mask(tmp_path):
    pairs, original_gt = write_rgb_gt_pair(tmp_path)

    inputs, targets, tree_ids, sample_keys = baseline.prepare_rgb_gt_dataset(pairs)

    assert inputs.shape == (1, 256, 256, 3)
    assert targets.shape == (1, 256, 256, 1)
    assert inputs.dtype == np.float32
    assert targets.dtype == np.float32
    assert np.all((inputs >= 0.0) & (inputs <= 1.0))
    expected_rgb = np.empty((256, 256, 3), dtype=np.float32)
    expected_rgb[:] = np.asarray((30, 20, 10), dtype=np.float32) / 255.0
    np.testing.assert_allclose(inputs[0], expected_rgb)
    np.testing.assert_array_equal(np.unique(targets), np.asarray((0.0, 1.0)))
    np.testing.assert_array_equal(targets[0, :, :, 0], original_gt == 255)
    np.testing.assert_array_equal(tree_ids, np.asarray((SAMPLE_KEY[0],)))
    assert sample_keys == [SAMPLE_KEY]


def test_rgbd_preprocessing_appends_one_normalized_depth_channel(tmp_path):
    pairs, _ = write_rgb_gt_pair(tmp_path)
    key = SAMPLE_KEY
    depth_path = tmp_path / "R01N01_Summer_D-synthetic-sample.png"
    depth = np.full((480, 640, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(depth_path), depth)
    triplets = {key: {**pairs[key], "D": depth_path}}

    inputs, targets, tree_ids, sample_keys = (
        rgbd_experiment.prepare_rgbd_dataset(triplets)
    )

    assert inputs.shape == (1, 256, 256, 4)
    assert targets.shape == (1, 256, 256, 1)
    expected_rgb = np.broadcast_to(
        np.asarray((30, 20, 10), dtype=np.float32) / 255.0,
        inputs[0, :, :, :3].shape,
    )
    np.testing.assert_allclose(inputs[0, :, :, :3], expected_rgb)
    np.testing.assert_allclose(inputs[0, :, :, 3], 128 / 255.0)
    np.testing.assert_array_equal(np.unique(targets), np.asarray((0.0, 1.0)))
    np.testing.assert_array_equal(tree_ids, np.asarray((SAMPLE_KEY[0],)))
    assert sample_keys == [SAMPLE_KEY]


def test_rgb_preprocessing_rejects_unexpected_original_dimensions(tmp_path):
    pairs, _ = write_rgb_gt_pair(tmp_path, rgb_shape=(479, 640, 3))

    with pytest.raises(ValueError, match="Unexpected RGB dimensions"):
        baseline.prepare_rgb_gt_dataset(pairs)