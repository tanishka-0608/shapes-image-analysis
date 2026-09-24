# Shapes Image Dataset: Data Audit, EDA & Classification

**Author:** Tanishka | **Role:** Data Analyst Intern, IBM

An end-to-end data analysis project on a synthetic image dataset of 5,400 geometric shapes.
The project covers the full analyst workflow: **data quality audit → feature engineering → exploratory analysis → statistical bias testing → machine-learning classification → reporting**.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Setup & Usage](#setup--usage)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Key Insights](#key-insights)
8. [Limitations](#limitations)
9. [Future Work](#future-work)
10. [Tech Stack](#tech-stack)

---

## Project Overview

**Business-style question:** *Is this image dataset clean, balanced and free of shortcuts, and can the shape in each image be identified from a small set of interpretable features?*

**Objectives**
- Audit the dataset for missing, corrupt, duplicate or leaked images.
- Convert raw pixels into human-interpretable numeric features (colour + geometry).
- Explore how those features differ across the six classes.
- Test statistically whether colour (a nuisance variable) leaks information about the label.
- Train and compare classical ML models, and explain which features drive the predictions.

Classical ML on hand-crafted features was chosen deliberately (instead of a deep network) because every feature is explainable, which is what a stakeholder-facing analysis needs.

## Dataset

| Property | Value |
|---|---|
| Images | 5,400 PNG files, 64 × 64 RGB |
| Classes | circle, square, triangle, star, pentagon, cross (900 each) |
| Splits | train 3,780 / val 810 / test 810 (stratified 70/15/15) |
| Variation | random colours (enforced contrast), position, size, rotation, blur, Gaussian noise |
| Labels | `train_labels.csv`, `val_labels.csv`, `test_labels.csv` with columns `filepath, label, label_id` |

The data is **synthetic** and fully reproducible (`generate_dataset.py`, seed = 42).

```
shapes_dataset/
├── train/<class>/<class>_0000.png ...
├── val/<class>/...
├── test/<class>/...
├── train_labels.csv
├── val_labels.csv
├── test_labels.csv
└── README.txt
```

## Project Structure

```
ibm_shapes_project/
├── README.md                 # this file
├── requirements.txt          # Python dependencies
├── generate_dataset.py       # (optional) regenerates the dataset from scratch
├── shapes_analysis.py        # main analysis pipeline
└── results/                  # created by the pipeline
    ├── summary.txt           # one-page text summary
    ├── metrics.json          # audit results + all model metrics
    ├── figures/              # 9 PNG charts
    └── tables/               # features.csv, ANOVA, classification report, ...
```

## Setup & Usage

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. get the dataset: unzip shapes_dataset.zip here, OR regenerate it
python generate_dataset.py ./shapes_dataset

# 4. run the analysis
python shapes_analysis.py --data-dir ./shapes_dataset --out-dir ./results
```

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `./shapes_dataset` | Folder containing the split folders and label CSVs |
| `--out-dir` | `./results` | Where figures, tables and metrics are saved |
| `--sample N` | `0` (all) | Use only N images per class per split, for a quick test run |
| `--seed` | `42` | Random seed for models and cross-validation |

Full run time is about 30 seconds on a standard laptop.

## Methodology

### 1. Data audit
Checks performed on all 5,400 records: missing files, corrupt files, image size and colour mode consistency, MD5-based duplicate detection, folder-name vs label consistency, null values, class-imbalance ratio, and **cross-split leakage** (the same image appearing in more than one split).

### 2. Feature engineering (`extract_features`)
Each image is segmented into foreground and background (background colour = median of the border pixels; foreground = pixels far from it, cleaned with morphological opening and largest-component selection). Then:

| Group | Features | Purpose |
|---|---|---|
| Colour / quality | foreground & background RGB, brightness, luminance contrast, noise level, sharpness | Nuisance variables, used to test for bias |
| Geometry | `area_frac`, `extent`, `bbox_aspect`, `circularity`, `solidity`, `eccentricity` | Overall form |
| Radial signature | `radial_mean`, `radial_cv`, `radial_min_max`, `n_vertices` | Distance from centroid to boundary as a function of angle; the number of peaks estimates the number of corners |

The geometric features are invariant to rotation, scale and colour, which is what makes them suitable for this dataset.

### 3. Exploratory analysis
Class and split balance, sample grid, per-class feature boxplots, colour distributions, correlation heatmap and a PCA projection of the shape features.

### 4. Statistical bias test
One-way ANOVA of each colour/quality feature across the six classes. A colour-only Random Forest is also trained as a sanity check: if it beats random guessing (16.7 %), colour would be leaking the label.

### 5. Modelling
Logistic Regression (scaled) and Random Forest (300 trees) trained on the shape features, evaluated on validation and held-out test sets, plus 5-fold stratified cross-validation on the training split. The best model is chosen on validation accuracy, with cross-validation accuracy as the tie-breaker.

## Results

### Data audit

| Check | Result |
|---|---|
| Missing / corrupt files | 0 / 0 |
| Image size / mode | all 64 × 64 / RGB |
| Duplicate images | 0 |
| Cross-split leakage | 0 |
| Label-folder mismatches / nulls | 0 / 0 |
| Class imbalance ratio (max/min) | 1.0 (perfectly balanced) |

### Model performance

| Model | Features | Val acc. | Test acc. | Test macro-F1 | 5-fold CV acc. |
|---|---|---|---|---|---|
| Colour-only Random Forest (bias check) | colour + quality | 0.185 | 0.169 | 0.168 | n/a |
| Logistic Regression | shape | 1.000 | 0.999 | 0.999 | 0.9997 ± 0.0005 |
| **Random Forest** | shape | **1.000** | **1.000** | **1.000** | **1.000 ± 0.000** |

Random guessing would give 0.167.

### Most important features (Random Forest)
`n_vertices` (0.25) > `radial_cv` (0.20) > `radial_min_max` (0.19) > `solidity` (0.14) > `extent` (0.13)

### Figures (in `results/figures/`)
| File | Content |
|---|---|
| `01_class_distribution.png` | Class balance overall and per split |
| `02_sample_images.png` | Sample images for every class |
| `03_shape_features_by_class.png` | Geometric feature boxplots by class |
| `04_colour_independence.png` | Colour statistics by class |
| `05_correlation_heatmap.png` | Feature correlation matrix |
| `06_pca_shape_features.png` | 2-D PCA projection (PC1 + PC2 explain ~70 % of variance) |
| `07_confusion_matrix.png` | Test-set confusion matrix (counts and row-normalised) |
| `08_feature_importance.png` | Random Forest feature importance |
| `09_model_comparison.png` | Validation vs test accuracy by model |

## Key Insights

1. **The dataset is clean and well-designed.** No missing, corrupt, duplicate or leaked images, and all classes are perfectly balanced in every split.
2. **Colour carries no information about the class.** A colour-only model scores at chance level (0.169 test accuracy vs 0.167 random), and ANOVA finds no significant class differences for any colour feature (all p > 0.05).
3. **`sharpness` does differ by class (p < 0.001), but this is expected, not a bias.** It is computed from the image Laplacian, which depends on the length of the shape's edge, so it reflects geometry rather than a colour or background shortcut.
4. **Corner count and radial variation separate the classes.** The classes fall into clear groups: circles have no corners and a near-constant radius; triangles, squares and pentagons have 3, 4 and 5 corners; stars and crosses have low `solidity` (concave outlines). `n_vertices` alone splits most classes.
5. **Simple, interpretable models are enough.** A linear model reaches 99.9 % test accuracy, so a deep network would add complexity without benefit on this data.

## Limitations

- **The near-perfect accuracy reflects an easy, clean synthetic dataset**, not real-world performance. Real images (occlusion, lighting, texture, clutter) would be far harder.
- Shapes are drawn with high contrast against a flat background, which makes segmentation trivial.
- Pixel-based `circularity` and `solidity` are approximations (values slightly above 1 occur because perimeter and hull area are measured on a discrete pixel grid). They are used as relative measures between classes, not as exact geometric quantities.
- Single dataset, single random seed for the main split.

## Future Work

- Add harder variations (overlapping shapes, textured backgrounds, low contrast, partial occlusion) and re-test the feature approach.
- Compare against a small CNN and report where interpretable features stop being enough.
- Add a Streamlit or IBM Cognos dashboard for interactive exploration of the feature table.
- Package the pipeline as a scheduled job with automated data-quality alerts.

## Tech Stack

Python 3 · NumPy · pandas · Pillow · SciPy · scikit-learn · Matplotlib · Seaborn
#   s h a p e s - i m a g e - a n a l y s i s  
 