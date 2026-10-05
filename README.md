# MMA3001 Tree Skeleton Segmentation

## Project Overview

This MMA3001 Numerical Methods and Machine Learning project investigates supervised 2D segmentation of tree-skeleton masks from Summer RGB images and, optionally, RGB-D input. It compares U-Net models with a simple spatial-prior baseline built from training ground-truth masks. The task is 2D mask prediction; it is not 3D reconstruction.

## Computational Problem

Given a processed RGB image, or RGB concatenated with one depth channel, predict a binary 256 x 256 ground-truth skeleton mask. Model performance is reported using mean per-image binary Dice and intersection over union (IoU). Prediction thresholds are selected on validation data and fixed before held-out test evaluation.

## Dataset

The supervised scripts use the flat directory `Dataset (With Summer GT)/`. It contains 562 complete Summer RGB/depth/GT triplets, plus Winter RGB/depth images. Filenames associate each modality using a tree ID, season, and capture tag; for example:

```text
R01N01_Summer_RGB-12-40-50.png
R01N01_Summer_D-12-40-50.png
R01N01_Summer_GT-12-40-50.png
```

Summer source RGB and depth images are 480 x 640 pixels with three channels. GT masks are 256 x 256 binary masks. All depth PNGs were found to have three identical channels; the RGB-D model uses one channel. The separate `Dataset/` directory contains seasonal RGB/depth files, but its relationship to the labelled data and its provenance are not documented. Dataset provenance and annotation-generation details are not available in this repository.

## Preprocessing

- RGB is read, converted from OpenCV BGR to RGB, centre-cropped by removing 80 columns from each horizontal side (480 x 480 crop), resized to 256 x 256, and normalized by 255 to [0, 1].
- GT remains at its existing 256 x 256 dimensions and is converted from 0/255 to binary 0/1.
- RGB-D preparation applies the same crop and resize to one depth channel, then divides its uint8 source values by 255. This is numerical normalization only; physical depth units are undocumented.

The RGB/GT transform is not supported by calibration or transformation documentation. A qualitative overlay inspection found the centre-crop-and-resize assumption broadly plausible, but does not establish exact spatial alignment. Six held-out test examples were viewed as part of that visual inspection; no test metrics were calculated for that inspection, and preprocessing was not changed as a result.

## Data Split

All experiments use the same seed-42 tree-level split:

| Split | Samples |
|---|---:|
| Training | 393 |
| Validation | 84 |
| Test | 85 |

Whole tree IDs are assigned to one split only. The split helper checks isolation, and the automated tests cover determinism and overlap rejection. Threshold and model selection use validation data; the held-out test results below are reported only for methods that were evaluated there.

## Methods and Results

Dice and IoU values are mean per-image binary scores. Prediction thresholds were selected using validation data. For methods evaluated on the held-out test set, the selected threshold was then fixed before test evaluation.

### Validation

| Method | Input | Threshold | Dice | IoU | Test evaluation |
|---|---|---:|---:|---:|---|
| U-Net + BCE | RGB | 0.25 | 0.354316 | 0.217112 | Yes |
| U-Net + BCE + Dice | RGB | 0.30 | 0.365778 | 0.226002 | Yes |
| U-Net + BCE + Dice | RGB + depth | 0.35 | 0.373455 | 0.231875 | Yes |
| Average-GT spatial prior | None | 0.20 | 0.379240 | 0.238432 | Yes |
| U-Net + Focal + Dice | RGB | 0.45 | 0.366954 | 0.227173 | No; validation-only |
| U-Net + Tversky (alpha=0.7, beta=0.3) | RGB | 0.50 | 0.140648 | 0.075686 | No; validation-only |

### Held-Out Test

Focal + Dice and Tversky were not evaluated on the held-out test set and are intentionally omitted here.

| Method | Input | Fixed threshold | Dice | IoU |
|---|---|---:|---:|---:|
| U-Net + BCE | RGB | 0.25 | 0.350560 | 0.214901 |
| U-Net + BCE + Dice | RGB | 0.30 | 0.372018 | 0.231315 |
| U-Net + BCE + Dice | RGB + depth | 0.35 | 0.371957 | 0.231123 |
| Average-GT spatial prior | None | 0.20 | 0.379632 | 0.239717 |

The Average-GT baseline is the pixel-wise mean of only the 393 training GT masks. It uses no RGB or depth and produces the same thresholded prediction for every sample. Its threshold was chosen on validation data and fixed for test evaluation. It achieved the highest test Dice and IoU among the tested methods. This indicates that a common spatial prior is strong; it does not show that the neural networks learned only that prior. Per-tree analysis found RGB BCE + Dice beat the prior on 37/85 test trees and RGB-D BCE + Dice beat it on 42/85, but neither model showed a consistent overall advantage.

### Foreground Diagnostics

On validation, the mean GT foreground fraction was 7.549%. Mean predicted foreground fractions and predicted/GT area ratios were:

| Method | Mean predicted foreground | Mean predicted/GT ratio |
|---|---:|---:|
| RGB BCE + Dice | 19.314% | 2.577 |
| RGB-D BCE + Dice | 19.984% | 2.666 |
| RGB Focal + Dice | 19.313% | 2.576 |
| RGB Tversky | 99.465% | 13.278258 |

The successful U-Net runs systematically predicted substantially more foreground area than the GT masks. Focal + Dice changed this little. The tested Tversky configuration (alpha=0.7, beta=0.3) failed under the current training setup, converging to a near-all-foreground solution; this result applies only to that configuration.

## Repository Guide

| File or directory | Purpose |
|---|---|
| `Tree_Project.py` | Shared pairing, RGB/GT preprocessing, split, U-Net, metrics, and original BCE baseline evaluator. |
| `Tree_Project_BCE_Dice.py` | RGB BCE + Dice training and validation threshold selection. |
| `Tree_Project_RGBD_BCE_Dice.py` | RGB-D BCE + Dice training and validation threshold selection. |
| `Tree_Project_Focal_Dice.py` | RGB Focal + Dice validation-only experiment. |
| `Tree_Project_Tversky.py` | RGB Tversky validation-only experiment. |
| `Prepare_RGBD_Data.py` | RGB-D triplet, channel, distribution, preprocessing, and split verification. |
| `Check_RGB_GT_Alignment.py` | Saves visual RGB/GT alignment examples across train, validation, and test. |
| `Evaluate_Average_GT_Baseline.py` | Builds the prior from training GT and selects its threshold on validation. |
| `Evaluate_BCE_Dice_Test.py` | Final test evaluation for RGB BCE + Dice. |
| `Evaluate_RGBD_BCE_Dice_Test.py` | Final test evaluation for RGB-D BCE + Dice. |
| `Evaluate_Average_GT_Test.py` | Final test evaluation for the Average-GT prior. |
| `Analyse_Foreground_Fraction.py` | Validation foreground fraction and area-ratio diagnostics. |
| `Analyse_Model_Errors.py` | Per-tree comparison, prior similarity, CSV, plots, and example montages for tested models. |
| `Final_Model_Comparison.py` | Generates comparison CSVs and figures from established results. |
| `tests/` | Synthetic-data pytest coverage for split, pairing, preprocessing, metrics, and losses. |
| `models/` | Saved checkpoints and generated result files. |
| `models/results/` | Diagnostic, evaluation, and comparison outputs. |

## Installation

The project was tested with Python 3.12.4. From the repository root, install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Running the Project

Run commands from the repository root. Training scripts start training and write their corresponding checkpoints; they are not required to inspect the existing result tables. These scripts are independent experiments, not a single sequence.

Examples of actual script entry points:

```powershell
python Prepare_RGBD_Data.py
python Check_RGB_GT_Alignment.py
python Tree_Project.py
python Tree_Project_BCE_Dice.py
python Tree_Project_RGBD_BCE_Dice.py
python Tree_Project_Focal_Dice.py
python Tree_Project_Tversky.py
python Evaluate_Average_GT_Baseline.py
python Evaluate_BCE_Dice_Test.py
python Evaluate_RGBD_BCE_Dice_Test.py
python Evaluate_Average_GT_Test.py
python Analyse_Foreground_Fraction.py
python Analyse_Model_Errors.py
python Final_Model_Comparison.py
```

The final comparison script reports supplied results; it does not load models or run inference. The original RGB+BCE checkpoint, `models/rgb_unet_baseline.keras`, is included in the repository, and `python Tree_Project.py` evaluates it on the held-out test split. The other test evaluators require their corresponding saved checkpoint and the expected dataset directory.

## Automated Tests

The tests use synthetic data and temporary files; they do not train or run inference. Run them from the repository root:

```powershell
python -m pytest tests -v
```

The completed suite result was **17 passed**. Coverage includes seed-based split determinism and isolation, modality pairing, preprocessing and normalization, Dice/IoU, and BCE + Dice, Focal + Dice, and Tversky loss calculations.

## Limitations and Reproducibility

- Dataset provenance, GT generation details, and a calibrated RGB-to-GT transformation are not documented.
- Visual alignment inspection is qualitative only. Six test samples were viewed in that diagnostic, but no test metrics were used for selection and no transform was tuned from it.
- The Average-GT baseline is a strong comparator because it exploits the common spatial distribution of the masks; the current networks did not show a consistent advantage, though they beat it on some individual test trees.
- The original RGB+BCE checkpoint is present; reproducing its evaluation still requires the expected dataset directory and pinned environment.
- Focal + Dice and the tested Tversky configuration have validation results only. No test results are available or reported for them.
- Reproduction assumes the data is placed in the exact active folder name `Dataset (With Summer GT)/` and retains the expected filename pairing pattern.

## AI-Use Acknowledgement

ChatGPT and GitHub Copilot were used to assist with explanation, debugging, code review, documentation, and test development. Generated suggestions were reviewed and tested before acceptance. The detailed AI-use reflection is provided in the project report.