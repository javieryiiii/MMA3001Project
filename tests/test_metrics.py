import numpy as np
import tensorflow as tf

import Tree_Project as baseline


def test_binary_dice_perfect_overlap():
    mask = np.asarray([1, 1, 0, 0], dtype=bool).reshape(1, 1, 4, 1)

    assert np.isclose(baseline.binary_dice_coefficient(mask, mask), 1.0)


def test_binary_dice_partial_overlap_and_no_overlap():
    ground_truth = np.asarray([1, 1, 0, 0], dtype=bool).reshape(1, 1, 4, 1)
    partial_prediction = np.asarray([1, 0, 1, 0], dtype=bool).reshape(1, 1, 4, 1)
    no_overlap_prediction = np.asarray([0, 0, 1, 1], dtype=bool).reshape(1, 1, 4, 1)

    assert np.isclose(
        baseline.binary_dice_coefficient(ground_truth, partial_prediction), 0.5
    )
    expected_no_overlap = 1e-6 / (4.0 + 1e-6)
    assert np.isclose(
        baseline.binary_dice_coefficient(ground_truth, no_overlap_prediction),
        expected_no_overlap,
    )


def test_binary_iou_perfect_partial_and_no_overlap():
    ground_truth = np.asarray([1, 1, 0, 0], dtype=bool).reshape(1, 1, 4, 1)
    partial_prediction = np.asarray([1, 0, 1, 0], dtype=bool).reshape(1, 1, 4, 1)
    no_overlap_prediction = np.asarray([0, 0, 1, 1], dtype=bool).reshape(1, 1, 4, 1)

    assert np.isclose(baseline.binary_intersection_over_union(ground_truth, ground_truth), 1.0)
    assert np.isclose(
        baseline.binary_intersection_over_union(ground_truth, partial_prediction),
        1.0 / 3.0,
    )
    expected_no_overlap = 1e-6 / (4.0 + 1e-6)
    assert np.isclose(
        baseline.binary_intersection_over_union(ground_truth, no_overlap_prediction),
        expected_no_overlap,
    )


def test_tensor_dice_and_iou_match_known_binary_masks():
    ground_truth = tf.constant([[[[1.0], [1.0], [0.0], [0.0]]]])
    prediction = tf.constant([[[[1.0], [0.0], [1.0], [0.0]]]])

    assert np.isclose(float(baseline.dice_coefficient(ground_truth, prediction)), 0.5)
    assert np.isclose(
        float(baseline.intersection_over_union(ground_truth, prediction)), 1.0 / 3.0
    )


def test_binary_metrics_reject_different_shapes():
    ground_truth = np.zeros((1, 2, 2, 1), dtype=bool)
    prediction = np.zeros((1, 2, 2), dtype=bool)

    with np.testing.assert_raises(ValueError):
        baseline.binary_dice_coefficient(ground_truth, prediction)
    with np.testing.assert_raises(ValueError):
        baseline.binary_intersection_over_union(ground_truth, prediction)