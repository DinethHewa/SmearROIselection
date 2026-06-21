# Inference-Time ROI Policy for Domain-Robust Deep Autofocus

This repository contains the source code, manuscript documents, summary data, tables, and figures for the study:

**Inference-time region-of-interest policy as a deployment variable for domain-robust deep autofocus in smear microscopy**

The study evaluates 17 ROI-selection policies under a frozen sign-aware autofocus estimator. Model weights, preprocessing, sign-confidence threshold, test support, and FOV-level aggregation are held constant. Only the inference-time ROI subset is changed.

## Main Result

Adaptive hybrid ROI selection reduced the worst-to-best smear-domain MAE gap from approximately 0.1702 um for center-top1 to 0.1202 um, a 29.39% relative reduction. Weighted pooled MAE remained nearly unchanged because WBC contributes most FOVs and is predominantly represented by one cached ROI per FOV.

The result supports a domain-consistency and efficiency claim, not a large pooled-MAE superiority claim.

## Repository Layout

```text
manuscript/       Main BSPC draft and supplementary document
src/              ROI selection and fixed-model evaluation code
config/           Frozen ROI calibration CSVs
results/data/     Policy-level summary data and statistical outputs
results/tables/   Manuscript-facing CSV and LaTeX tables
results/figures/  Main ROI-policy figures
```

Large per-ROI predictions, model checkpoints, score diagnostics, and image panels are excluded from GitHub. They belong in the associated Zenodo artifact package.

## Core Scripts

```text
src/Regression/scripts/final phase - regression/evaluation/run_roi_ablation_suite.py
src/Regression/scripts/final phase - regression/evaluation/roi_policy_utils.py
src/Regression/scripts/final phase - regression/evaluation/sanity_check_roi_ablation.py
src/Regression/scripts/phase 2 - roi selection/roi_selection_v2.py
src/Regression/scripts/phase 2 - roi selection/roi_selection_cnn_v1.py
```

## Policy Families

```text
center_top1
all_rois
random_k, k = 1, 3, 5, 7
focus_only_topk, k = 1, 3, 5, 7
occupancy_only_topk, k = 1, 3, 5, 7
legacy_adaptive
cnn_adaptive
hybrid_proposed
```

## Environment

Install dependencies using:

```bash
python -m pip install -r requirements.txt
```

The original research scripts retain parts of the parent-project directory structure. Use the Zenodo package for the frozen checkpoints, fixed-model predictions, and complete reproducibility artifacts.

