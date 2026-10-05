import numpy as np
import tensorflow as tf

import Tree_Project as baseline
import Tree_Project_BCE_Dice as bce_dice_experiment
import Tree_Project_Focal_Dice as focal_experiment
import Tree_Project_Tversky as tversky_experiment


def test_dice_loss_is_one_minus_existing_smoothed_dice():
    ground_truth = tf.constant([[[[1.0], [1.0], [0.0], [0.0]]]])
    prediction = tf.constant([[[[1.0], [0.0], [1.0], [0.0]]]])

    expected = 1.0 - float(baseline.dice_coefficient(ground_truth, prediction))
    actual = float(bce_dice_experiment.dice_loss(ground_truth, prediction))
    assert np.isclose(actual, expected)


def test_combined_bce_dice_loss_is_sum_of_its_components():
    ground_truth = tf.constant([[[[1.0], [0.0], [1.0], [0.0]]]])
    prediction = tf.constant([[[[0.8], [0.2], [0.4], [0.1]]]])
    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(ground_truth, prediction)
    )
    expected = float(bce) + float(
        bce_dice_experiment.dice_loss(ground_truth, prediction)
    )

    actual = float(
        bce_dice_experiment.combined_bce_dice_loss(ground_truth, prediction)
    )
    assert np.isclose(actual, expected)


def test_binary_focal_loss_matches_numpy_reference():
    ground_truth = np.asarray([1.0, 0.0], dtype=np.float32).reshape(1, 1, 2, 1)
    prediction = np.asarray([0.8, 0.2], dtype=np.float32).reshape(1, 1, 2, 1)
    epsilon = tf.keras.backend.epsilon()
    probabilities = np.clip(prediction, epsilon, 1.0 - epsilon)
    pixel_bce = -(
        ground_truth * np.log(probabilities)
        + (1.0 - ground_truth) * np.log(1.0 - probabilities)
    )
    probability_for_true_class = (
        ground_truth * probabilities
        + (1.0 - ground_truth) * (1.0 - probabilities)
    )
    expected = np.mean(
        np.power(1.0 - probability_for_true_class, focal_experiment.GAMMA)
        * pixel_bce
    )

    actual = float(
        focal_experiment.binary_focal_loss(ground_truth, prediction)
    )
    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_combined_focal_dice_loss_is_sum_of_its_components():
    ground_truth = tf.constant([[[[1.0], [0.0]]]])
    prediction = tf.constant([[[[0.7], [0.3]]]])
    expected = float(focal_experiment.binary_focal_loss(ground_truth, prediction))
    expected += float(focal_experiment.dice_loss(ground_truth, prediction))

    actual = float(
        focal_experiment.combined_focal_dice_loss(ground_truth, prediction)
    )
    assert np.isclose(actual, expected)


def test_tversky_loss_matches_fixed_alpha_beta_reference():
    ground_truth = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32).reshape(
        1, 1, 4, 1
    )
    prediction = np.asarray([0.8, 0.3, 0.6, 0.1], dtype=np.float32).reshape(
        1, 1, 4, 1
    )
    true_positive = 1.1
    false_positive = 0.7
    false_negative = 0.9
    smooth = tversky_experiment.SMOOTH
    expected = 1.0 - (
        (true_positive + smooth)
        / (
            true_positive
            + tversky_experiment.ALPHA * false_positive
            + tversky_experiment.BETA * false_negative
            + smooth
        )
    )

    actual = float(tversky_experiment.tversky_loss(ground_truth, prediction))
    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-7)
    assert tversky_experiment.ALPHA == 0.7
    assert tversky_experiment.BETA == 0.3