# -*- coding: utf-8 -*-
import io
import json
import math
import shutil
import tempfile
import warnings
from contextlib import ExitStack
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shap
import streamlit as st
from rasterio.io import MemoryFile
from rasterio.windows import Window
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV, RepeatedKFold

try:
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
except Exception:
    WarpedVRT = None
    Resampling = None

st.set_page_config(
    page_title="MicroFragment Atlas Pro",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- style ----------

def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 12%, rgba(28, 93, 142, 0.10), transparent 24%),
                radial-gradient(circle at 88% 14%, rgba(11, 108, 104, 0.08), transparent 24%),
                radial-gradient(circle at 80% 76%, rgba(74, 108, 152, 0.08), transparent 28%),
                linear-gradient(180deg, #f4f8fb 0%, #edf3f7 46%, #f8fbfd 100%);
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 1.7rem 1.9rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(19,39,63,0.96), rgba(27,78,104,0.92));
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 24px 68px rgba(15,23,42,0.16);
            margin-bottom: 1rem;
        }
        .hero::after {
            content: "";
            position: absolute;
            right: -40px;
            bottom: -40px;
            width: 240px;
            height: 240px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 64%);
            pointer-events: none;
        }
        .hero h1 {
            margin: 0 0 .35rem 0;
            color: #f8fbff;
            font-size: 2.4rem;
            line-height: 1.02;
            letter-spacing: -.025em;
        }
        .hero p {
            margin: 0;
            color: rgba(248,251,255,0.86);
            font-size: 1rem;
            line-height: 1.58;
            max-width: 920px;
        }
        .kicker {
            display: inline-block;
            padding: .34rem .72rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.10);
            color: #dbeeff;
            font-size: .8rem;
            font-weight: 700;
            letter-spacing: .06em;
            text-transform: uppercase;
            margin-bottom: .75rem;
        }
        .glass {
            border-radius: 22px;
            border: 1px solid rgba(15,23,42,.08);
            background: rgba(255,255,255,.86);
            box-shadow: 0 16px 42px rgba(15,23,42,.05);
            padding: 1rem 1.05rem .9rem 1.05rem;
        }
        .tiny {
            color: #556476;
            font-size: .93rem;
            line-height: 1.56;
        }
        .section-title {
            margin-top: .2rem;
            margin-bottom: .52rem;
            color: #102033;
            font-weight: 800;
            letter-spacing: -.02em;
        }
        .soft-note {
            border-radius: 16px;
            border: 1px solid rgba(15,23,42,.08);
            background: rgba(255,255,255,.78);
            padding: .85rem 1rem;
            color: #4f6073;
            line-height: 1.55;
            font-size: .93rem;
            margin-bottom: .8rem;
        }
        .stMetric {
            background: rgba(255,255,255,.9);
            border: 1px solid rgba(15,23,42,.06);
            padding: .7rem .9rem;
            border-radius: 18px;
            box-shadow: 0 10px 26px rgba(15,23,42,.04);
        }
        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(15,23,42,.08);
        }
        .stButton>button, .stDownloadButton>button {
            border-radius: 14px;
            font-weight: 700;
        }
        .sidebar-note {
            color: #546579;
            font-size: .92rem;
            line-height: 1.56;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

# ---------- helpers ----------
def metric_dict(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")

def _shap_values_tree_explainer(explainer, X):
    try:
        out = explainer(X)
        sv = out.values if hasattr(out, "values") else out
    except Exception:
        sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]
    return sv

def sanitize_feature_name(name: str):
    return Path(name).stem.lower().replace("1000", "")

def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file, na_values=["#VALUE!", "NaN", "nan", "Inf", "inf"])

def prepare_simple_training_data(df, target, x_coord=None, y_coord=None, drop_cols=None):
    drop_cols = drop_cols or []
    df2 = df.copy()

    required = [target]
    if x_coord:
        required.append(x_coord)
    if y_coord:
        required.append(y_coord)

    keep_cols = [c for c in df2.columns if c not in set(drop_cols)]
    df2 = df2[keep_cols].dropna()

    if target not in df2.columns:
        raise ValueError(f"Target column '{target}' is not present after exclusions.")

    predictors = [c for c in df2.columns if c != target and c not in {x_coord, y_coord}]
    if not predictors:
        raise ValueError("No predictors remain. Please keep at least one predictor column.")

    X_df = df2[predictors].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df2[target], errors="coerce").values.astype(float)

    coord_df = None
    if x_coord and y_coord and x_coord in df2.columns and y_coord in df2.columns:
        coord_df = df2[[x_coord, y_coord]].apply(pd.to_numeric, errors="coerce")

    good = np.isfinite(X_df.values).all(axis=1) & np.isfinite(y)
    if coord_df is not None:
        good &= np.isfinite(coord_df.values).all(axis=1)

    X_df = X_df.loc[good].copy()
    y = y[good]
    if coord_df is not None:
        coord_df = coord_df.loc[good].copy()

    if len(X_df) < 5:
        raise ValueError(f"Only {len(X_df)} valid rows remain after dropna()/numeric conversion. Please check the CSV.")
    if X_df.shape[1] < 1:
        raise ValueError("No usable predictors remain after preprocessing.")
    if np.unique(y).size < 2:
        raise ValueError("Target variable has fewer than 2 unique values after preprocessing.")

    return X_df, y, coord_df

def fit_simple_rf(X, y, n_estimators=300, random_state=42):
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X, y)
    return model

def cross_val_summary_for_fixed_model(X, y, n_estimators=300, random_state=42, cv_splits=5):
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold_id, (tr_idx, te_idx) in enumerate(cv.split(X), start=1):
        m = fit_simple_rf(X[tr_idx], y[tr_idx], n_estimators=n_estimators, random_state=random_state)
        pred = m.predict(X[te_idx])
        rows.append({"fold": fold_id, **metric_dict(y[te_idx], pred)})
    df_folds = pd.DataFrame(rows)
    return df_folds, float(df_folds["R2"].mean())

def compute_shap_sample(model, X_df, sample_size=200, random_state=42):
    if len(X_df) > sample_size:
        X_use = X_df.sample(sample_size, random_state=random_state).copy()
    else:
        X_use = X_df.copy()
    explainer = shap.TreeExplainer(model)
    sv = _shap_values_tree_explainer(explainer, X_use.values.astype("float32", copy=False))
    shap_df = pd.DataFrame(sv, columns=X_use.columns, index=X_use.index)
    imp = pd.DataFrame({
        "feature": X_use.columns,
        "mean_abs_shap": np.abs(shap_df.values).mean(axis=0),
        "mean_shap": shap_df.values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return X_use, shap_df, imp

def prepare_predictors(df, target, x_coord=None, y_coord=None, drop_cols=None):
    drop_cols = drop_cols or []
    exclude = {target}
    if x_coord:
        exclude.add(x_coord)
    if y_coord:
        exclude.add(y_coord)

    predictors = [c for c in df.columns if c not in exclude and c not in drop_cols]
    X_df = df[predictors].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce").values.astype(float)

    coord_df = None
    if x_coord and y_coord and x_coord in df.columns and y_coord in df.columns:
        coord_df = df[[x_coord, y_coord]].apply(pd.to_numeric, errors="coerce")

    good = np.isfinite(X_df.values).all(axis=1) & np.isfinite(y)
    if coord_df is not None:
        good &= np.isfinite(coord_df.values).all(axis=1)

    return X_df.loc[good].copy(), y[good], None if coord_df is None else coord_df.loc[good].copy()

def collinearity_filter(X_df, threshold=0.85, method="spearman", always_keep=None):
    always_keep = set(always_keep or [])
    feats0 = list(X_df.columns)
    if len(feats0) <= 1:
        return X_df.copy(), pd.DataFrame({"feature": feats0, "kept": True, "reason": "not_filtered"}), pd.DataFrame()

    corr = X_df.corr(method=method).abs()
    remaining = list(feats0)
    removed = {}
    removed_pairs = []

    while True:
        pairs = []
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                a, b = remaining[i], remaining[j]
                v = corr.loc[a, b]
                if pd.notna(v) and v >= threshold:
                    pairs.append((a, b, float(v)))
        if not pairs:
            break

        a, b, v = sorted(pairs, key=lambda x: x[2], reverse=True)[0]
        if a in always_keep and b in always_keep:
            corr.loc[a, b] = -np.inf
            corr.loc[b, a] = -np.inf
            continue
        elif a in always_keep:
            drop, keep = b, a
        elif b in always_keep:
            drop, keep = a, b
        else:
            a_score = corr.loc[a, remaining].drop(a).mean()
            b_score = corr.loc[b, remaining].drop(b).mean()
            drop, keep = (a, b) if a_score >= b_score else (b, a)

        remaining.remove(drop)
        removed[drop] = f"removed_due_to_collinearity_with_{keep}"
        removed_pairs.append({"feature_a": a, "feature_b": b, "abs_corr": v, "dropped": drop, "kept": keep})

    report_rows = [{"feature": f, "kept": f in remaining, "reason": "kept" if f in remaining else removed.get(f, "removed")} for f in feats0]
    return X_df[remaining].copy(), pd.DataFrame(report_rows), pd.DataFrame(removed_pairs)

def get_param_distributions():
    return {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [None, 5, 10, 15, 20, 30],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "max_features": [1.0, "sqrt", 0.5, 0.7],
        "bootstrap": [True],
    }

def fit_best_rf(X, y, random_state=42, search_iter=20, cv_splits=5):
    base_model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=1,
    )
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=get_param_distributions(),
        n_iter=search_iter,
        scoring="r2",
        cv=cv,
        n_jobs=1,
        random_state=random_state,
        refit=True,
        verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)

def evaluate_repeated_cv(model, X, y, random_state=42, cv_splits=5, cv_repeats=3):
    rkf = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=random_state)
    rows = []
    oof = np.full(y.shape[0], np.nan, dtype=float)
    for fold_id, (tr_idx, te_idx) in enumerate(rkf.split(X), start=1):
        m = RandomForestRegressor(**model.get_params())
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[te_idx])
        first_time = np.isnan(oof[te_idx])
        oof[te_idx[first_time]] = pred[first_time]
        rows.append({"fold": fold_id, **metric_dict(y[te_idx], pred)})
    return pd.DataFrame(rows), pd.DataFrame({"observed": y, "predicted_oof": oof, "residual": y - oof})

def evaluate_spatial_cv(model, X, y, coord_df, random_state=42, spatial_blocks=5):
    if coord_df is None or coord_df.empty:
        return None, None
    km = KMeans(n_clusters=spatial_blocks, random_state=random_state, n_init=20)
    groups = km.fit_predict(coord_df.values)
    n_splits = min(spatial_blocks, len(np.unique(groups)))
    if n_splits < 2:
        return None, None
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    oof = np.full(y.shape[0], np.nan, dtype=float)
    for fold_id, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        m = RandomForestRegressor(**model.get_params())
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[te_idx])
        oof[te_idx] = pred
        rows.append({"fold": fold_id, **metric_dict(y[te_idx], pred)})
    return pd.DataFrame(rows), pd.DataFrame({"observed": y, "predicted_spatial_oof": oof, "residual": y - oof, "group": groups})

def compute_permutation_importance(model, X, y, feature_names, random_state=42):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi = permutation_importance(model, X, y, n_repeats=15, random_state=random_state, n_jobs=1, scoring="r2")
    return pd.DataFrame({
        "feature": feature_names,
        "perm_importance_mean": pi.importances_mean,
        "perm_importance_std": pi.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

def compute_shap_summary(model, X_df):
    explainer = shap.TreeExplainer(model)
    sv = _shap_values_tree_explainer(explainer, X_df.values.astype("float32", copy=False))
    shap_df = pd.DataFrame(sv, columns=X_df.columns)
    imp = pd.DataFrame({
        "feature": X_df.columns,
        "mean_abs_shap": np.abs(shap_df.values).mean(axis=0),
        "mean_shap": shap_df.values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return shap_df, imp

def fig_observed_pred(df_oof, pred_col="predicted_oof", title="Observed vs predicted"):
    fig, ax = plt.subplots(figsize=(5.6, 5.1))
    good = np.isfinite(df_oof["observed"]) & np.isfinite(df_oof[pred_col])
    x = df_oof.loc[good, "observed"].values
    y = df_oof.loc[good, pred_col].values
    ax.scatter(x, y, s=28, alpha=0.72)
    if len(x) > 0:
        mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([mn, mx], [mn, mx], linewidth=1.2)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    return fig

def fig_barh(df, value_col, label_col="feature", title="", top_n=15, xlabel=None):
    d = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, max(4, 0.35 * len(d))))
    ax.barh(d[label_col], d[value_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel or value_col)
    ax.set_ylabel("")
    return fi

def fig_value_shap_driver(x, y, feature_name, bins=60):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    good = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[good]
    y = np.asarray(y)[good]
    ax.scatter(x, y, s=18, alpha=0.35, label="Samples")

    if len(x) >= 8:
        order = np.argsort(x)
        xs = x[order]
        ys = y[order]
        n = len(xs)
        window = max(5, min(max(7, n // 12), 51))
        if window % 2 == 0:
            window += 1
        smooth = pd.Series(ys).rolling(window=window, center=True, min_periods=max(3, window // 3)).mean().to_numpy()
        valid = np.isfinite(smooth)
        if valid.any():
            ax.plot(xs[valid], smooth[valid], linewidth=2.2, label="Smoothed driver")

    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("SHAP value")
    ax.set_title(f"Driver process: {feature_name}")
    ax.legend(frameon=False)
    return fig

def fig_feature_distribution(x, feature_name):
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) > 0:
        ax.hist(x, bins=min(40, max(10, len(x) // 8)))
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Count")
    ax.set_title(f"Observed distribution: {feature_name}")
    return fig
g

# ---------- raster helpers ----------
def save_uploaded_rasters_to_temp(uploaded_rasters):
    tempdir = Path(tempfile.mkdtemp(prefix="microfragment_rasters_"))
    saved = []
    for up in uploaded_rasters:
        p = tempdir / up.name
        p.write_bytes(up.getbuffer())
        saved.append(p)
    return tempdir, saved

def find_raster_for_feature(feature, raster_paths):
    feature_norm = feature.lower().strip()
    for p in raster_paths:
        stem = sanitize_feature_name(p.name)
        if stem == feature_norm:
            return str(p)
    for p in raster_paths:
        stem = sanitize_feature_name(p.name)
        if feature_norm in stem or stem in feature_norm:
            return str(p)
    raise FileNotFoundError(f"No uploaded TIFF matched feature '{feature}'.")

def open_and_align_datasets(feature_names, raster_paths, resampling_name="bilinear"):
    path_map = {f: find_raster_for_feature(f, raster_paths) for f in feature_names}
    stack = ExitStack()
    raw = {}
    try:
        for feat, p in path_map.items():
            raw[feat] = stack.enter_context(rasterio.open(p))
        ref_key = next(iter(raw.keys()))
        ref_ds = raw[ref_key]
        datasets = {}
        if WarpedVRT is None or Resampling is None:
            for feat, ds in raw.items():
                datasets[feat] = ds
            return stack, datasets

        resampling = getattr(Resampling, resampling_name, Resampling.bilinear)
        for feat, ds in raw.items():
            same = (
                ds.crs == ref_ds.crs and
                ds.transform == ref_ds.transform and
                ds.width == ref_ds.width and
                ds.height == ref_ds.height
            )
            if same:
                datasets[feat] = ds
            else:
                vrt = WarpedVRT(
                    ds,
                    crs=ref_ds.crs,
                    transform=ref_ds.transform,
                    width=ref_ds.width,
                    height=ref_ds.height,
                    resampling=resampling,
                )
                datasets[feat] = stack.enter_context(vrt)
        return stack, datasets
    except Exception:
        stack.close()
        raise

def _make_profile(tmpl_ds, nodata=-9999.0):
    profile = tmpl_ds.profile.copy()
    profile["driver"] = "GTiff"
    profile["count"] = 1
    profile["dtype"] = "float32"
    profile["nodata"] = float(nodata)
    profile["compress"] = "lzw"
    return profile

def read_predictor_block(datasets, feature_names, win):
    h = int(win.height)
    w = int(win.width)
    p = len(feature_names)
    stack_x = np.empty((h, w, p), dtype="float32")
    invalid = None
    for j, fn in enumerate(feature_names):
        ds = datasets[fn]
        arr = ds.read(1, window=win).astype("float32", copy=False)
        inv = ~np.isfinite(arr)
        if ds.nodata is not None:
            inv |= (arr == ds.nodata)
        invalid = inv if invalid is None else (invalid | inv)
        stack_x[:, :, j] = arr
    return stack_x, invalid

def build_kfold_models(best_model, X, y, cv_splits=5, random_state=42):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    models = []
    for tr_idx, te_idx in kf.split(X):
        m = RandomForestRegressor(**best_model.get_params())
        m.fit(X[tr_idx], y[tr_idx])
        models.append(m)
    return models

def run_raster_prediction(best_model, feature_names, raster_paths, X_train_df, y, cv_splits=5, random_state=42, block=512):
    tempdir = Path(tempfile.mkdtemp(prefix="microfragment_outputs_"))
    stack, datasets = open_and_align_datasets(feature_names, raster_paths)
    try:
        tmpl = datasets[next(iter(datasets.keys()))]
        profile = _make_profile(tmpl, nodata=-9999.0)
        pred_path = tempdir / "prediction.tif"
        mean_unc_path = tempdir / "prediction_cv_mean.tif"
        std_unc_path = tempdir / "prediction_cv_std.tif"

        X_train = X_train_df.values.astype("float32", copy=False)
        kfold_models = build_kfold_models(best_model, X_train, y, cv_splits=cv_splits, random_state=random_state)

        with rasterio.open(pred_path, "w", **profile) as dst_pred, \
             rasterio.open(mean_unc_path, "w", **profile) as dst_mean, \
             rasterio.open(std_unc_path, "w", **profile) as dst_std:
            for row0 in range(0, tmpl.height, block):
                for col0 in range(0, tmpl.width, block):
                    h = min(block, tmpl.height - row0)
                    w = min(block, tmpl.width - col0)
                    win = Window(col0, row0, w, h)

                    stack_x, invalid = read_predictor_block(datasets, feature_names, win)
                    Xw = stack_x.reshape(-1, len(feature_names))
                    valid = ~invalid.reshape(-1)

                    pred = np.full(h * w, -9999.0, dtype="float32")
                    pred_mean = np.full(h * w, -9999.0, dtype="float32")
                    pred_std = np.full(h * w, -9999.0, dtype="float32")

                    if np.any(valid):
                        Xv = Xw[valid]
                        pred[valid] = best_model.predict(Xv).astype("float32", copy=False)
                        preds = np.column_stack([m.predict(Xv).astype("float32", copy=False) for m in kfold_models])
                        pred_mean[valid] = preds.mean(axis=1)
                        pred_std[valid] = preds.std(axis=1, ddof=1 if preds.shape[1] > 1 else 0)

                    dst_pred.write(pred.reshape(h, w), 1, window=win)
                    dst_mean.write(pred_mean.reshape(h, w), 1, window=win)
                    dst_std.write(pred_std.reshape(h, w), 1, window=win)
        return tempdir, pred_path, mean_unc_path, std_unc_path
    finally:
        stack.close()


def run_single_feature_shap_raster(best_model, feature_names, raster_paths, selected_feature, block=512):
    if selected_feature not in feature_names:
        raise ValueError(f"Selected feature '{selected_feature}' is not in model predictors.")

    tempdir = Path(tempfile.mkdtemp(prefix="microfragment_shap_outputs_"))
    stack, datasets = open_and_align_datasets(feature_names, raster_paths)
    try:
        tmpl = datasets[next(iter(datasets.keys()))]
        profile = _make_profile(tmpl, nodata=-9999.0)
        shap_path = tempdir / f"shap_{selected_feature}.tif"
        selected_idx = feature_names.index(selected_feature)
        explainer = shap.TreeExplainer(best_model)

        with rasterio.open(shap_path, "w", **profile) as dst_shap:
            for row0 in range(0, tmpl.height, block):
                for col0 in range(0, tmpl.width, block):
                    h = min(block, tmpl.height - row0)
                    w = min(block, tmpl.width - col0)
                    win = Window(col0, row0, w, h)

                    stack_x, invalid = read_predictor_block(datasets, feature_names, win)
                    Xw = stack_x.reshape(-1, len(feature_names))
                    valid = ~invalid.reshape(-1)

                    out = np.full(h * w, -9999.0, dtype="float32")
                    if np.any(valid):
                        Xv = Xw[valid].astype("float32", copy=False)
                        sv = _shap_values_tree_explainer(explainer, Xv)
                        out[valid] = sv[:, selected_idx].astype("float32", copy=False)

                    dst_shap.write(out.reshape(h, w), 1, window=win)
        return tempdir, shap_path
    finally:
        stack.close()

def preview_raster_png(raster_path):
    with rasterio.open(raster_path) as ds:
        arr = ds.read(1).astype("float32")
        nodata = ds.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        im = ax.imshow(arr, cmap="viridis")
        ax.set_title(Path(raster_path).name)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return fig

def export_bundle(model, feature_names, config, tables):
    bundle = io.BytesIO()
    with io.BytesIO() as m:
        joblib.dump({"model": model, "features": feature_names, "config": config}, m)
        model_bytes = m.getvalue()

    import zipfile
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.joblib", model_bytes)
        z.writestr("config.json", json.dumps(config, ensure_ascii=False, indent=2))
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame):
                z.writestr(f"{name}.csv", df.to_csv(index=False))
    bundle.seek(0)
    return bundle

# ---------- sidebar ----------
with st.sidebar:
    st.title("MicroFragment Atlas Pro")
    st.markdown('<div class="sidebar-note">An elegant scientific workspace for microplastic fragmentation modelling, sample-level interpretation, and raster-based projection.</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-note">Recommended workflow: upload the sample CSV, configure the response and coordinates, train the model, inspect sample-level SHAP, then run raster prediction and create a SHAP map for one selected predictor.</div>', unsafe_allow_html=True)
    csv_file = st.file_uploader("Upload sampling CSV", type=["csv"])
    raster_files = st.file_uploader("Upload predictor TIFF files", type=["tif", "tiff"], accept_multiple_files=True)

    st.markdown("---")
    st.markdown("**Model controls**")
    random_state = st.number_input("Random state", 1, 9999, 42, 1)
    n_estimators = st.slider("Number of trees", 100, 1000, 300, 50)
    cv_splits = st.slider("CV splits", 3, 10, 5)
    cv_repeats = st.slider("CV repeats", 1, 5, 3)
    spatial_blocks = st.slider("Spatial blocks", 3, 10, 5)
    compute_perm = st.checkbox("Compute permutation importance", value=True)
    compute_shap = st.checkbox("Enable sample-level SHAP analysis", value=False, help="Disabled by default to reduce memory usage during deployment.")

# ---------- hero ----------
st.markdown(
    """
    <div class="hero">
      <div class="kicker">Scientific modelling workspace</div>
      <h1>MicroFragment Atlas</h1>
      <p>Train a robust random forest model from sampling data, compare repeated and spatial cross-validation, interpret drivers with sample-level SHAP, generate regional GeoTIFF prediction surfaces, and produce a single-variable SHAP raster after prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_data, tab_model, tab_validation, tab_interpret, tab_raster, tab_export = st.tabs(
    ["Overview", "Data setup", "Model", "Validation", "Sample-level SHAP", "Raster prediction", "Export"]
)

if csv_file is None:
    with tab_overview:
        st.info("Upload a CSV from the sidebar to begin.")
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="glass"><h4 class="section-title">Research-grade interface</h4><p class="tiny">A clean visual language designed for model building, interpretation, and publication-ready inspection.</p></div>', unsafe_allow_html=True)
        c2.markdown('<div class="glass"><h4 class="section-title">Sample-to-region workflow</h4><p class="tiny">Move from sample-table modelling and validation to regional raster prediction in one controlled workflow.</p></div>', unsafe_allow_html=True)
        c3.markdown('<div class="glass"><h4 class="section-title">Two-stage SHAP logic</h4><p class="tiny">First inspect SHAP at sample points, then optionally generate one spatial SHAP map after raster prediction.</p></div>', unsafe_allow_html=True)
    st.stop()

df = load_csv(csv_file)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isna().sum().sum()))
    c4.metric("Uploaded TIFFs", len(raster_files) if raster_files else 0)
    st.subheader("Data preview")
    st.dataframe(df.head(12), width="stretch", height=420)

with tab_data:
    st.subheader("Dataset configuration")
    st.caption("Define the response variable, optional coordinate fields, and any columns to exclude from modelling.")
    all_cols = list(df.columns)
    col1, col2, col3 = st.columns(3)
    default_target = all_cols.index("MPs") if "MPs" in all_cols else 0
    target = col1.selectbox("Response variable", all_cols, index=default_target)
    x_default = ([""] + all_cols).index("Longitude") if "Longitude" in all_cols else 0
    y_default = ([""] + all_cols).index("Latitude") if "Latitude" in all_cols else 0
    x_coord = col2.selectbox("X coordinate (optional)", [""] + all_cols, index=x_default)
    y_coord = col3.selectbox("Y coordinate (optional)", [""] + all_cols, index=y_default)

    possible_drop = [c for c in all_cols if c not in {target, x_coord, y_coord}]
    drop_cols = st.multiselect("Exclude columns from predictors", possible_drop, default=[])

    try:
        X_filt, y, coord_df = prepare_simple_training_data(df, target, x_coord or None, y_coord or None, drop_cols)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Rows used for modelling", len(X_filt))
        mc2.metric("Predictor count", X_filt.shape[1])
        mc3.metric("Target", target)
        st.markdown("**Predictors used**")
        st.write(", ".join(list(X_filt.columns)))
    except Exception as e:
        X_filt, y, coord_df = None, None, None
        st.error(f"Dataset configuration error: {e}")

run_model = tab_model.button("Run modelling workflow", type="primary", width="stretch")

if "results" not in st.session_state:
    st.session_state["results"] = None

if run_model:
    if X_filt is None:
        st.error("Please fix the dataset configuration first.")
    else:
        X = X_filt.values.astype("float32", copy=False)

        with st.spinner("Training fixed RandomForest model, validating performance and computing interpretation outputs..."):
            best_model = fit_simple_rf(X, y, n_estimators=n_estimators, random_state=random_state)
            cv_summary, mean_cv_r2 = cross_val_summary_for_fixed_model(X, y, n_estimators=n_estimators, random_state=random_state, cv_splits=cv_splits)
            rep_folds, rep_oof = evaluate_repeated_cv(best_model, X, y, random_state=random_state, cv_splits=cv_splits, cv_repeats=cv_repeats)
            spatial_folds, spatial_oof = evaluate_spatial_cv(best_model, X, y, coord_df, random_state=random_state, spatial_blocks=spatial_blocks)

            rf_imp = pd.DataFrame({
                "feature": X_filt.columns,
                "rf_importance": best_model.feature_importances_,
            }).sort_values("rf_importance", ascending=False).reset_index(drop=True)

            if compute_perm:
                perm_imp = compute_permutation_importance(best_model, X, y, list(X_filt.columns), random_state=random_state)
            else:
                perm_imp = pd.DataFrame(columns=["feature", "perm_importance_mean", "perm_importance_std"])

            if compute_shap:
                shap_X, shap_df, shap_imp = compute_shap_sample(best_model, X_filt, sample_size=min(250, len(X_filt)), random_state=random_state)
            else:
                shap_X = X_filt.head(0).copy()
                shap_df = pd.DataFrame(columns=X_filt.columns)
                shap_imp = pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_shap"])

        st.session_state["results"] = {
            "target": target,
            "x_coord": x_coord,
            "y_coord": y_coord,
            "drop_cols": drop_cols,
            "X_filt": X_filt,
            "y": y,
            "coord_df": coord_df,
            "best_model": best_model,
            "best_params": {"model_type": "Fixed RandomForestRegressor", "n_estimators": int(n_estimators), "random_state": int(random_state)},
            "best_cv_score": mean_cv_r2,
            "rep_folds": rep_folds,
            "rep_oof": rep_oof,
            "spatial_folds": spatial_folds,
            "spatial_oof": spatial_oof,
            "rf_imp": rf_imp,
            "perm_imp": perm_imp,
            "shap_X": shap_X,
            "shap_df": shap_df,
            "shap_imp": shap_imp,
            "col_report": pd.DataFrame({"feature": list(X_filt.columns), "kept": True, "reason": "used_in_model"}),
            "col_pairs": pd.DataFrame(),
            "raster_outputs": None,
            "shap_raster_outputs": None,
        }

res = st.session_state["results"]

with tab_model:
    if res is None:
        st.info("Configure the data in the Data audit tab, then run the modelling workflow.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Response", res["target"])
        c2.metric("Mean CV R²", f'{res["best_cv_score"]:.3f}')
        c3.metric("Retained predictors", res["X_filt"].shape[1])
        st.markdown("**Model settings**")
        st.json(res["best_params"])

with tab_validation:
    if res is None:
        st.info("Run the modelling workflow first.")
    else:
        st.subheader("Repeated cross-validation")
        c1, c2 = st.columns([1.05, 0.95])
        rep_summary = res["rep_folds"][["R2", "RMSE", "MAE"]].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
        c1.dataframe(rep_summary, width="stretch")
        c2.pyplot(fig_observed_pred(res["rep_oof"], "predicted_oof", "Repeated-CV prediction"))
        st.dataframe(res["rep_folds"], width="stretch", height=260)

        st.subheader("Spatial cross-validation")
        if res["spatial_folds"] is None:
            st.warning("Spatial validation was not run because valid coordinate columns were not provided.")
        else:
            s1, s2 = st.columns([1, 1])
            spatial_summary = res["spatial_folds"][["R2", "RMSE", "MAE"]].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
            s1.dataframe(spatial_summary, width="stretch")
            s2.pyplot(fig_observed_pred(res["spatial_oof"], "predicted_spatial_oof", "Spatial-CV prediction"))

with tab_interpret:
    if res is None:
        st.info("Run the modelling workflow first.")
    else:
        st.subheader("Sample-level SHAP analysis")
        st.caption("This section focuses on driver mechanisms at the sample-point level before any raster-based projection is generated.")

        c1, c2, c3 = st.columns(3)
        c1.pyplot(fig_barh(res["rf_imp"], "rf_importance", "feature", "Random forest importance", 12, "Importance"))
        if res["perm_imp"].empty:
            c2.info("Permutation importance was not computed.")
        else:
            c2.pyplot(fig_barh(res["perm_imp"], "perm_importance_mean", "feature", "Permutation importance", 12, "Mean permutation importance"))
        if res["shap_imp"].empty:
            c3.info("Sample-level SHAP is disabled.")
        else:
            c3.pyplot(fig_barh(res["shap_imp"], "mean_abs_shap", "feature", "Global SHAP importance", 12, "Mean |SHAP|"))

        if res["shap_imp"].empty:
            st.info("Enable sample-level SHAP analysis from the sidebar to inspect variable-level driving processes.")
        else:
            st.markdown("### Variable driving process")
            driver_feature = st.selectbox(
                "Select one predictor to inspect the sample-level driving process",
                list(res["shap_X"].columns),
                key="sample_shap_feature",
            )

            left, right = st.columns([1.4, 1.0])
            with left:
                st.pyplot(
                    fig_value_shap_driver(
                        res["shap_X"][driver_feature].values,
                        res["shap_df"][driver_feature].values,
                        driver_feature,
                    )
                )
            with right:
                st.pyplot(
                    fig_feature_distribution(
                        res["shap_X"][driver_feature].values,
                        driver_feature,
                    )
                )

            st.markdown("### Value–SHAP data view")
            driver_df = pd.DataFrame({
                "feature_value": res["shap_X"][driver_feature].values,
                "shap_value": res["shap_df"][driver_feature].values,
            }).sort_values("feature_value").reset_index(drop=True)
            st.dataframe(driver_df, width="stretch", height=260)

            with st.expander("Show interpretation tables"):
                t1, t2 = st.columns(2)
                t1.dataframe(res["perm_imp"], width="stretch", height=280)
                t2.dataframe(res["shap_imp"], width="stretch", height=280)

with tab_raster:
    if res is None:
        st.info("Run the modelling workflow first.")
    else:
        st.subheader("Regional raster prediction")
        st.caption("Upload one TIFF per retained predictor. File names should match predictor names, optionally with the suffix 1000, for example sand1000.tif.")
        retained = list(res["X_filt"].columns)
        st.write("Retained predictors:", ", ".join(retained))

        if not raster_files:
            st.warning("No TIFF files have been uploaded yet.")
        else:
            raster_names = [f.name for f in raster_files]
            st.write("Uploaded TIFF files:", ", ".join(raster_names))

            if st.button("Run raster prediction and uncertainty export", width="stretch"):
                with st.spinner("Matching TIFF predictors, aligning rasters and generating outputs..."):
                    raster_tempdir, saved_rasters = save_uploaded_rasters_to_temp(raster_files)
                    try:
                        out_dir, pred_path, mean_path, std_path = run_raster_prediction(
                            res["best_model"],
                            retained,
                            saved_rasters,
                            res["X_filt"],
                            res["y"],
                            cv_splits=cv_splits,
                            random_state=random_state,
                        )
                        res["raster_outputs"] = {
                            "temp_rasters": str(raster_tempdir),
                            "out_dir": str(out_dir),
                            "prediction": str(pred_path),
                            "mean": str(mean_path),
                            "std": str(std_path),
                        }
                        st.success("Raster prediction completed.")
                    except Exception as e:
                        st.error(f"Raster prediction failed: {e}")

        if res.get("raster_outputs"):
            outputs = res["raster_outputs"]
            p1, p2 = st.columns(2)
            with p1:
                st.pyplot(preview_raster_png(outputs["prediction"]))
                st.download_button(
                    "Download prediction.tif",
                    data=Path(outputs["prediction"]).read_bytes(),
                    file_name="prediction.tif",
                    mime="application/octet-stream",
                )
            with p2:
                st.pyplot(preview_raster_png(outputs["std"]))
                st.download_button(
                    "Download prediction_cv_std.tif",
                    data=Path(outputs["std"]).read_bytes(),
                    file_name="prediction_cv_std.tif",
                    mime="application/octet-stream",
                )
            st.download_button(
                "Download prediction_cv_mean.tif",
                data=Path(outputs["mean"]).read_bytes(),
                file_name="prediction_cv_mean.tif",
                mime="application/octet-stream",
            )

            st.markdown("---")
            st.subheader("Single-variable SHAP spatial map")
            st.caption("After raster prediction, generate a SHAP raster for one selected predictor. This keeps the workflow interpretable while controlling memory use.")
            shap_feature = st.selectbox("Select one predictor for SHAP mapping", retained, key="shap_raster_feature")
            if st.button("Generate SHAP raster for selected predictor", width="stretch"):
                with st.spinner("Computing SHAP raster for the selected predictor..."):
                    raster_tempdir, saved_rasters = save_uploaded_rasters_to_temp(raster_files)
                    try:
                        shap_dir, shap_path = run_single_feature_shap_raster(
                            res["best_model"],
                            retained,
                            saved_rasters,
                            shap_feature,
                        )
                        res["shap_raster_outputs"] = {
                            "temp_rasters": str(raster_tempdir),
                            "out_dir": str(shap_dir),
                            "feature": shap_feature,
                            "path": str(shap_path),
                        }
                        st.success(f"SHAP raster completed for: {shap_feature}")
                    except Exception as e:
                        st.error(f"SHAP raster generation failed: {e}")

        if res.get("shap_raster_outputs"):
            shap_outputs = res["shap_raster_outputs"]
            st.pyplot(preview_raster_png(shap_outputs["path"]))
            st.download_button(
                f"Download SHAP raster: {shap_outputs['feature']}",
                data=Path(shap_outputs["path"]).read_bytes(),
                file_name=f"shap_{shap_outputs['feature']}.tif",
                mime="application/octet-stream",
            )

with tab_export:
    if res is None:
        st.info("Run the modelling workflow first.")
    else:
        st.subheader("Export model bundle")
        st.caption("Download the trained model, configuration, validation metrics, and interpretation tables as a compact archive.")
        config = {
            "target": res["target"],
            "x_coord": res["x_coord"],
            "y_coord": res["y_coord"],
            "drop_cols": res["drop_cols"],
            "random_state": random_state,
            "cv_splits": cv_splits,
            "n_estimators": n_estimators,
            "cv_repeats": cv_repeats,
            "spatial_blocks": spatial_blocks,
        }
        bundle = export_bundle(
            res["best_model"],
            list(res["X_filt"].columns),
            config,
            {
                "feature_selection_report": res["col_report"],
                "collinearity_pairs": res["col_pairs"],
                "repeated_cv_metrics": res["rep_folds"],
                "repeated_cv_oof": res["rep_oof"],
                "spatial_cv_metrics": res["spatial_folds"],
                "spatial_cv_oof": res["spatial_oof"],
                "rf_importance": res["rf_imp"],
                "permutation_importance": res["perm_imp"],
                "shap_importance": res["shap_imp"],
            },
        )
        st.download_button(
            "Download model bundle (.zip)",
            data=bundle,
            file_name="microfragment_model_bundle.zip",
            mime="application/zip",
            width="stretch",
        )
        st.markdown("**Quick notes**")
        st.write(
            "The bundle includes the trained model, retained feature list, full configuration, and the key validation and interpretation tables. "
            "Raster GeoTIFF outputs are downloaded separately from the Raster prediction tab."
        )
