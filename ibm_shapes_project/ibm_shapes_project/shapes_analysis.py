#!/usr/bin/env python3
"""
Shapes Image Dataset - Data Audit, EDA & Classification
========================================================

End-to-end analysis of a synthetic image dataset (6 shape classes, 64x64 RGB).

Pipeline
--------
1. Load      : read train/val/test label files into pandas
2. Audit     : missing/corrupt files, image size & mode, duplicates, label consistency
3. Features  : extract interpretable colour + geometric features from every image
4. EDA       : class balance, feature distributions, correlations, PCA projection
5. Bias check: are colour features independent of the class label? (ANOVA + colour-only model)
6. Modelling : Logistic Regression & Random Forest on shape features, evaluated on val/test
7. Report    : figures, tables, metrics.json and summary.txt saved to the output folder

Usage
-----
    python shapes_analysis.py --data-dir ./shapes_dataset --out-dir ./results
    python shapes_analysis.py --data-dir ./shapes_dataset --sample 300   # quick run
"""

import argparse
import hashlib
import json
import logging
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
log = logging.getLogger("shapes")

SPLITS = ["train", "val", "test"]
COLOUR_FEATURES = ["fg_r", "fg_g", "fg_b", "bg_r", "bg_g", "bg_b",
                   "fg_brightness", "bg_brightness", "lum_contrast"]
QUALITY_FEATURES = ["noise_std", "sharpness"]
SHAPE_FEATURES = ["area_frac", "extent", "bbox_aspect", "circularity", "solidity",
                  "eccentricity", "radial_mean", "radial_cv", "radial_min_max",
                  "n_vertices"]


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #
def load_labels(data_dir: str) -> pd.DataFrame:
    """Read <split>_labels.csv for every split and stack them into one table."""
    frames = []
    for split in SPLITS:
        path = os.path.join(data_dir, f"{split}_labels.csv")
        df = pd.read_csv(path)
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["abs_path"] = df["filepath"].apply(lambda p: os.path.join(data_dir, p))
    log.info("Loaded %d records from %d splits", len(df), len(SPLITS))
    return df


# --------------------------------------------------------------------------- #
# 2. Data audit
# --------------------------------------------------------------------------- #
def audit_dataset(df: pd.DataFrame) -> dict:
    """Run data-quality checks and return them as a dictionary."""
    report = {"n_records": int(len(df))}

    report["missing_files"] = int((~df["abs_path"].apply(os.path.exists)).sum())

    sizes, modes, hashes = [], [], []
    corrupt = 0
    for path in df["abs_path"]:
        try:
            with Image.open(path) as im:
                sizes.append(im.size)
                modes.append(im.mode)
            with open(path, "rb") as f:
                hashes.append(hashlib.md5(f.read()).hexdigest())
        except Exception:
            corrupt += 1
            sizes.append(None)
            modes.append(None)
            hashes.append(None)
    df["img_size"] = sizes
    df["img_mode"] = modes
    df["md5"] = hashes

    report["corrupt_files"] = corrupt
    report["unique_image_sizes"] = sorted({str(s) for s in sizes if s})
    report["unique_image_modes"] = sorted({m for m in modes if m})
    report["duplicate_images"] = int(df["md5"].dropna().duplicated().sum())

    # folder name in the path must agree with the label column
    folder_label = df["filepath"].str.split("/").str[1]
    report["label_folder_mismatch"] = int((folder_label != df["label"]).sum())

    report["null_values"] = int(df[["filepath", "label", "label_id"]].isna().sum().sum())
    report["class_counts"] = df["label"].value_counts().sort_index().to_dict()
    report["split_counts"] = df["split"].value_counts().reindex(SPLITS).to_dict()

    counts = df["label"].value_counts()
    report["imbalance_ratio_max_min"] = round(counts.max() / counts.min(), 3)

    # No image should appear in more than one split (data leakage check)
    leak = df.dropna(subset=["md5"]).groupby("md5")["split"].nunique()
    report["cross_split_leaks"] = int((leak > 1).sum())
    return report


# --------------------------------------------------------------------------- #
# 3. Feature engineering
# --------------------------------------------------------------------------- #
def extract_features(path: str) -> dict:
    """
    Turn one image into a row of interpretable features.

    Colour / quality features describe *how the image looks* (nuisance variables).
    Shape features describe *what the object is* (geometry) and are rotation-,
    scale- and colour-invariant, which is why they can separate the classes.
    """
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    h, w, _ = img.shape

    # Background colour = median of the border pixels (shapes never touch the border)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    bg = np.median(border, axis=0)

    # Foreground mask = pixels far from the background colour, then clean-up
    dist = np.linalg.norm(img - bg, axis=2)
    mask = dist > 60
    mask = ndi.binary_opening(mask, iterations=1)
    labelled, n = ndi.label(mask)
    if n > 1:  # keep the largest connected blob only
        sizes = ndi.sum(mask, labelled, range(1, n + 1))
        mask = labelled == (np.argmax(sizes) + 1)
    if mask.sum() < 10:  # safety net: degenerate image
        return {}

    fg = img[mask].mean(axis=0)
    lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    gray = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    feats = {
        "fg_r": fg[0], "fg_g": fg[1], "fg_b": fg[2],
        "bg_r": bg[0], "bg_g": bg[1], "bg_b": bg[2],
        "fg_brightness": lum(fg), "bg_brightness": lum(bg),
        "lum_contrast": abs(lum(fg) - lum(bg)),
        "noise_std": float(img[~ndi.binary_dilation(mask, iterations=3)].std(axis=0).mean()),
        "sharpness": float(ndi.laplace(gray).var()),
    }

    # ---- geometry ---------------------------------------------------------
    ys, xs = np.nonzero(mask)
    area = float(mask.sum())
    feats["area_frac"] = area / (h * w)

    bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    feats["extent"] = area / (bh * bw)                 # fill ratio of bounding box
    feats["bbox_aspect"] = min(bh, bw) / max(bh, bw)

    boundary = mask & ~ndi.binary_erosion(mask)
    perimeter = float(boundary.sum())
    feats["circularity"] = 4 * np.pi * area / (perimeter ** 2 + 1e-9)

    pts = np.column_stack([xs, ys])
    try:
        feats["solidity"] = area / ConvexHull(pts).volume   # area / convex-hull area
    except Exception:
        feats["solidity"] = 1.0

    cx, cy = xs.mean(), ys.mean()
    mu20, mu02 = ((xs - cx) ** 2).mean(), ((ys - cy) ** 2).mean()
    mu11 = ((xs - cx) * (ys - cy)).mean()
    common = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    l1, l2 = (mu20 + mu02 + common) / 2, (mu20 + mu02 - common) / 2
    feats["eccentricity"] = float(np.sqrt(max(0.0, 1 - l2 / (l1 + 1e-9))))

    # ---- radial signature: distance from centroid to boundary vs. angle ------
    by, bx = np.nonzero(boundary)
    r = np.hypot(bx - cx, by - cy)
    theta = np.arctan2(by - cy, bx - cx)
    n_bins = 36
    bins = ((theta + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    sig = np.full(n_bins, np.nan)
    for b in range(n_bins):
        vals = r[bins == b]
        if len(vals):
            sig[b] = vals.max()
    good = ~np.isnan(sig)
    centers = np.arange(n_bins)
    sig = np.interp(centers, centers[good], sig[good], period=n_bins)
    sig = ndi.uniform_filter1d(sig, size=3, mode="wrap")

    sig_n = sig / sig.mean()
    feats["radial_mean"] = float(sig.mean() / np.sqrt(area))
    feats["radial_cv"] = float(sig_n.std())
    feats["radial_min_max"] = float(sig.min() / sig.max())

    # number of "corners" = peaks of the radial signature (circular -> tile 3x)
    tiled = np.tile(sig_n, 3)
    peaks, _ = find_peaks(tiled, prominence=0.04)
    feats["n_vertices"] = int(((peaks >= n_bins) & (peaks < 2 * n_bins)).sum())
    return feats


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features for every image and join them to the metadata table."""
    t0 = time.time()
    rows = []
    for i, path in enumerate(df["abs_path"], 1):
        rows.append(extract_features(path))
        if i % 1000 == 0:
            log.info("  features: %d/%d images", i, len(df))
    feat = pd.DataFrame(rows, index=df.index)
    out = pd.concat([df.drop(columns=["img_size"], errors="ignore"), feat], axis=1)
    out = out.dropna(subset=SHAPE_FEATURES).reset_index(drop=True)
    log.info("Feature extraction done in %.1fs (%d rows kept)", time.time() - t0, len(out))
    return out


# --------------------------------------------------------------------------- #
# 4. Plots
# --------------------------------------------------------------------------- #
def save(fig, out_dir, name):
    path = os.path.join(out_dir, "figures", name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  saved %s", path)


def plot_class_distribution(df, out_dir, classes):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.countplot(data=df, x="label", order=classes, ax=axes[0], palette="viridis")
    axes[0].set_title("Images per class (all splits)")
    axes[0].set_xlabel("")
    for c in axes[0].containers:
        axes[0].bar_label(c)
    sns.countplot(data=df, x="split", hue="label", order=SPLITS, hue_order=classes,
                  ax=axes[1], palette="viridis")
    axes[1].set_title("Class balance within each split")
    axes[1].legend(ncol=2, fontsize=8)
    save(fig, out_dir, "01_class_distribution.png")


def plot_sample_grid(df, out_dir, classes, per_class=8):
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 1.3, len(classes) * 1.4))
    train = df[df["split"] == "train"]
    for r, cls in enumerate(classes):
        sub = train[train["label"] == cls].head(per_class)
        for c in range(per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(sub):
                ax.imshow(Image.open(sub.iloc[c]["abs_path"]))
            if c == 0:
                ax.set_title(cls, fontsize=9, loc="left")
    fig.suptitle("Sample images per class", y=1.0)
    save(fig, out_dir, "02_sample_images.png")


def plot_feature_distributions(df, out_dir, classes):
    feats = ["area_frac", "extent", "circularity", "solidity",
             "radial_cv", "radial_min_max", "eccentricity", "n_vertices"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, f in zip(axes.ravel(), feats):
        sns.boxplot(data=df, x="label", y=f, order=classes, ax=ax, palette="Set2", fliersize=1)
        ax.set_title(f)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40)
    fig.suptitle("Geometric features by class", y=1.01, fontsize=14)
    fig.tight_layout()
    save(fig, out_dir, "03_shape_features_by_class.png")


def plot_colour_independence(df, out_dir, classes):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, f in zip(axes, ["fg_brightness", "bg_brightness", "lum_contrast"]):
        sns.violinplot(data=df, x="label", y=f, order=classes, ax=ax, palette="pastel", cut=0)
        ax.set_title(f)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40)
    fig.suptitle("Colour statistics look the same for every class (no colour shortcut)", y=1.03)
    save(fig, out_dir, "04_colour_independence.png")


def plot_correlation(df, out_dir):
    cols = SHAPE_FEATURES + ["noise_std", "sharpness", "lum_contrast"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature correlation matrix")
    save(fig, out_dir, "05_correlation_heatmap.png")


def plot_pca(df, out_dir, classes):
    X = StandardScaler().fit_transform(df[SHAPE_FEATURES])
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for cls in classes:
        m = (df["label"] == cls).values
        ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.5, label=cls)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA of shape features")
    ax.legend(markerscale=3)
    save(fig, out_dir, "06_pca_shape_features.png")
    return [float(v) for v in pca.explained_variance_ratio_]


def plot_confusion(y_true, y_pred, classes, title, out_dir, name):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=axes[0])
    axes[0].set_title(f"{title} - counts")
    cmn = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title(f"{title} - row-normalised")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    save(fig, out_dir, name)


def plot_importance(model, features, out_dir):
    imp = pd.Series(model.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot.barh(ax=ax, color="teal")
    ax.set_title("Random Forest feature importance")
    ax.set_xlabel("Importance")
    save(fig, out_dir, "08_feature_importance.png")
    return imp.sort_values(ascending=False)


def plot_model_comparison(results, out_dir):
    df = pd.DataFrame(results).T[["val_accuracy", "test_accuracy"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df.plot.bar(ax=ax, rot=15, color=["#4c72b0", "#dd8452"])
    ax.axhline(1 / 6, ls="--", color="grey", label="random guess (1/6)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model comparison")
    ax.legend()
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    save(fig, out_dir, "09_model_comparison.png")


# --------------------------------------------------------------------------- #
# 5/6. Statistical bias check + modelling
# --------------------------------------------------------------------------- #
def colour_bias_test(df, classes):
    """ANOVA per nuisance feature: a large p-value = feature is independent of class."""
    rows = []
    for f in COLOUR_FEATURES + QUALITY_FEATURES:
        groups = [df.loc[df["label"] == c, f].values for c in classes]
        F, p = stats.f_oneway(*groups)
        rows.append({"feature": f, "F_statistic": F, "p_value": p,
                     "independent_of_class(p>0.01)": bool(p > 0.01)})
    return pd.DataFrame(rows)


def evaluate(name, model, feats, tr, va, te, classes):
    model.fit(tr[feats], tr["label"])
    out = {}
    for split_name, part in [("val", va), ("test", te)]:
        pred = model.predict(part[feats])
        out[f"{split_name}_accuracy"] = accuracy_score(part["label"], pred)
        out[f"{split_name}_macro_f1"] = f1_score(part["label"], pred, average="macro")
        out[f"_{split_name}_pred"] = pred
    log.info("  %-28s val acc %.4f | test acc %.4f | test macro-F1 %.4f",
             name, out["val_accuracy"], out["test_accuracy"], out["test_macro_f1"])
    return out


def run_models(df, classes, out_dir, seed):
    tr, va, te = (df[df["split"] == s] for s in SPLITS)
    results, preds = {}, {}

    specs = {
        "Colour-only RF (bias check)": (
            RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
            COLOUR_FEATURES + QUALITY_FEATURES),
        "Logistic Regression": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
            SHAPE_FEATURES),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
            SHAPE_FEATURES),
    }
    fitted = {}
    for name, (model, feats) in specs.items():
        r = evaluate(name, model, feats, tr, va, te, classes)
        preds[name] = r.pop("_test_pred")
        r.pop("_val_pred")
        results[name] = r
        fitted[name] = (model, feats)

    # 5-fold cross-validation on the training split for the two shape models
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for name in ["Logistic Regression", "Random Forest"]:
        model, feats = fitted[name]
        scores = cross_val_score(model, tr[feats], tr["label"], cv=skf, n_jobs=-1)
        results[name]["cv_accuracy_mean"] = float(scores.mean())
        results[name]["cv_accuracy_std"] = float(scores.std())

    best = max(["Logistic Regression", "Random Forest"],
               key=lambda n: (results[n]["val_accuracy"], results[n]["cv_accuracy_mean"]))  # tie-break on CV
    plot_confusion(te["label"], preds[best], classes, f"{best} (test set)", out_dir,
                   "07_confusion_matrix.png")
    imp = plot_importance(fitted["Random Forest"][0], SHAPE_FEATURES, out_dir)
    plot_model_comparison(results, out_dir)

    report = pd.DataFrame(classification_report(te["label"], preds[best], output_dict=True)).T
    return results, best, report, imp


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./shapes_dataset", help="folder that contains the dataset")
    ap.add_argument("--out-dir", default="./results", help="where figures/tables are written")
    ap.add_argument("--sample", type=int, default=0, help="use only N images per class (quick test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    np.random.seed(args.seed)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)

    # 1. load
    df = load_labels(args.data_dir)
    if args.sample:
        df = df.groupby(["split", "label"], group_keys=False).head(args.sample).reset_index(drop=True)
        log.info("Sampling enabled -> %d records", len(df))
    classes = sorted(df["label"].unique())

    # 2. audit
    log.info("Auditing dataset ...")
    audit = audit_dataset(df)
    for k, v in audit.items():
        log.info("  %-26s %s", k, v)

    # 3. features
    log.info("Extracting features ...")
    feat = build_feature_table(df, )
    feat.drop(columns=["abs_path"]).to_csv(os.path.join(args.out_dir, "tables", "features.csv"), index=False)

    # 4. EDA
    log.info("Creating EDA figures ...")
    plot_class_distribution(feat, args.out_dir, classes)
    plot_sample_grid(feat, args.out_dir, classes)
    plot_feature_distributions(feat, args.out_dir, classes)
    plot_colour_independence(feat, args.out_dir, classes)
    plot_correlation(feat, args.out_dir)
    pca_var = plot_pca(feat, args.out_dir, classes)

    summary_stats = feat.groupby("label")[SHAPE_FEATURES].mean().round(3)
    summary_stats.to_csv(os.path.join(args.out_dir, "tables", "shape_feature_means_by_class.csv"))

    # 5. bias check
    log.info("Testing colour independence ...")
    bias = colour_bias_test(feat, classes)
    bias.to_csv(os.path.join(args.out_dir, "tables", "colour_bias_anova.csv"), index=False)

    # 6. models
    log.info("Training models ...")
    results, best, report, imp = run_models(feat, classes, args.out_dir, args.seed)
    report.to_csv(os.path.join(args.out_dir, "tables", "classification_report_test.csv"))
    imp.to_csv(os.path.join(args.out_dir, "tables", "feature_importance.csv"), header=["importance"])

    # 7. report
    metrics = {"audit": audit, "pca_explained_variance_pc1_pc2": pca_var,
               "models": results, "best_model": best,
               "top_features": imp.head(5).round(4).to_dict()}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    lines = ["SHAPES DATASET - ANALYSIS SUMMARY", "=" * 40,
             f"Images analysed        : {audit['n_records']}",
             f"Missing / corrupt files: {audit['missing_files']} / {audit['corrupt_files']}",
             f"Duplicate images       : {audit['duplicate_images']}",
             f"Cross-split leakage    : {audit['cross_split_leaks']}",
             f"Class imbalance ratio  : {audit['imbalance_ratio_max_min']}", "",
             "Model results (accuracy):"]
    for name, r in results.items():
        lines.append(f"  {name:<30} val {r['val_accuracy']:.4f} | test {r['test_accuracy']:.4f}")
    lines += ["", f"Best shape model: {best}",
              "Top features: " + ", ".join(imp.head(5).index)]
    summary = "\n".join(lines)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
