# -*- coding: utf-8 -*-
import io
import json
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
from rasterio.windows import Window
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RepeatedKFold

try:
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
except Exception:
    WarpedVRT = None
    Resampling = None


warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed`")
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="MicroFragment Atlas Pro",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(30, 64, 175, 0.10), transparent 24%),
                radial-gradient(circle at 88% 16%, rgba(13, 148, 136, 0.10), transparent 22%),
                linear-gradient(180deg, #f8fafc 0%, #eef4f7 100%);
        }
        .block-container {
            max-width: 1320px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 1.8rem 2rem;
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(30, 64, 175, 0.08), rgba(13, 148, 136, 0.12)),
                rgba(255,255,255,0.78);
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(8px);
            margin-bottom: 1rem;
        }
        .kicker {
            display: inline-block;
            padding: .32rem .72rem;
            border-radius: 999px;
            background: rgba(13, 148, 136, 0.10);
            color: #0f766e;
            font-size: .80rem;
            font-weight: 700;
            letter-spacing: .05em;
            text-transform: uppercase;
            margin-bottom: .75rem;
        }
        .hero h1 {
            margin: 0 0 .35rem 0;
            color: #0f172a;
            font-size: 2.5rem;
            line-height: 1.02;
            letter-spacing: -.02em;
        }
        .hero p {
            margin: 0;
            color: #475569;
            font-size: 1.02rem;
            line-height: 1.55;
        }
        .glass {
            border-radius: 22px;
            border: 1px solid rgba(15, 23, 42, 0.07);
            background: rgba(255,255,255,0.82);
            box-shadow: 0 14px 40px rgba(15,23,42,.05);
            padding: 1rem 1.05rem .9rem 1.05rem;
        }
        .tiny {
            color: #64748b;
            font-size: .94rem;
            line-height: 1.6;
        }
        .section-title {
            margin-top: .2rem;
            margin-bottom: .45rem;
            color: #0f172a;
            font-weight: 800;
            letter-spacing: -.02em;
        }
        .stMetric {
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(15,23,42,.06);
            padding: .72rem .9rem;
            border-radius: 18px;
            box-shadow: 0 10px 28px rgba(15,23,42,.04);
        }
        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(15,23,42,.08);
        }
        .stButton>button, .stDownloadButton>button {
            border-radius: 14px;
            font-weight: 700;
        }
        .sidebar-note {
            color: #475569;
            font-size: .93rem;
            line-height: 1.58;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def metric_dict(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file, na_values=["#VALUE!", "NaN", "nan", "Inf", "inf"])


def make_clean_training_table(df: pd.DataFrame, target: str, x_coord=None, y_coord=None, drop_cols=None):
    drop_cols = drop_cols or []
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is not present in the uploaded table.")

    exclude = {target}
    if x_coord:
        exclude.add(x_coord)
    if y_coord:
        exclude.add(y_coord)

    predictors = [c for c in df.columns if c not in exclude and c not in set(drop_cols)]
    if not predictors:
        raise ValueError("No predictor columns remain after the current configuration.")

    X_df = df[predictors].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")

    coord_df = None
    if x_coord and y_coord and x_coord in df.columns and y_coord in df.columns:
        coord_df = df[[x_coord, y_coord]].apply(pd.to_numeric, errors="coerce")

    good = np.isfinite(X_df.values).all(axis=1) & np.isfinite(y.values)
    if coord_df is not None:
        good &= np.isfinite(coord_df.values).all(axis=1)

    X_df = X_df.loc[good].copy()
    y = y.loc[good].astype(float).values
    coord_df = None if coord_df is None else coord_df.loc[good].copy()

    nunique = X_df.nunique(dropna=True)
    non_constant = nunique[nunique > 1].index.tolist()
    X_df = X_df[non_constant].copy()

    if X_df.shape[0] < 10:
        raise ValueError("Too few valid rows remain after numeric conversion and NA removal. At least 10 rows are recommended.")
    if X_df.shape[1] == 0:
        raise ValueError("All predictors became constant or invalid after preprocessing.")
    if np.unique(y).size < 2:
        raise ValueError("The response variable has no usable variation after preprocessing.")

    return X_df, y, coord_df


def fit_rf_model(
    X_df,
    y,
    random_state=42,
    n_estimators=158,
    max_depth=14,
    min_samples_leaf=1,
    min_samples_split=2,
):
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        max_depth=None if max_depth in [None, "None"] else int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        min_samples_split=int(min_samples_split),
        n_jobs=1,
    )
    model.fit(X_df.values.astype("float32", copy=False), y)
    return model


def evaluate_repeated_cv(X_df, y, model_params, random_state=42, cv_splits=5, cv_repeats=3):
    rkf = RepeatedKFold(
        n_splits=int(cv_splits),
        n_repeats=int(cv_repeats),
        random_state=int(random_state),
    )
    rows = []
    oof_sum = np.zeros(len(y), dtype=float)
    oof_count = np.zeros(len(y), dtype=int)

    X = X_df.values.astype("float32", copy=False)

    for fold_id, (tr_idx, te_idx) in enumerate(rkf.split(X), start=1):
        model = RandomForestRegressor(**model_params)
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[te_idx])
        oof_sum[te_idx] += pred
        oof_count[te_idx] += 1
        rows.append({"fold": fold_id, **metric_dict(y[te_idx], pred)})

    oof_pred = np.divide(
        oof_sum,
        oof_count,
        out=np.full_like(oof_sum, np.nan, dtype=float),
        where=oof_count > 0,
    )
    oof_df = pd.DataFrame({"observed": y, "predicted_oof": oof_pred, "residual": y - oof_pred})
    return pd.DataFrame(rows), oof_df


def evaluate_spatial_cv(X_df, y, coord_df, model_params, random_state=42, spatial_blocks=5):
    if coord_df is None or coord_df.empty:
        return None, None

    if len(coord_df) < int(spatial_blocks):
        return None, None

    coords = coord_df.values.astype("float32", copy=False)
    try:
        km = KMeans(n_clusters=int(spatial_blocks), random_state=int(random_state), n_init=10)
        groups = km.fit_predict(coords)
    except Exception:
        return None, None

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return None, None

    splitter = GroupKFold(n_splits=min(len(unique_groups), int(spatial_blocks)))
    rows = []
    oof_pred = np.full(len(y), np.nan, dtype=float)
    X = X_df.values.astype("float32", copy=False)

    for fold_id, (tr_idx, te_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        model = RandomForestRegressor(**model_params)
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[te_idx])
        oof_pred[te_idx] = pred
        rows.append({"fold": fold_id, **metric_dict(y[te_idx], pred)})

    oof_df = pd.DataFrame({"observed": y, "predicted_spatial_oof": oof_pred, "residual": y - oof_pred})
    return pd.DataFrame(rows), oof_df


def compute_permutation_importance_table(model, X_df, y, random_state=42):
    X = X_df.values.astype("float32", copy=False)
    out = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=int(random_state),
        n_jobs=1,
    )
    df = pd.DataFrame(
        {
            "feature": X_df.columns,
            "perm_importance_mean": out.importances_mean,
            "perm_importance_std": out.importances_std,
        }
    ).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
    return df


def _shap_values_tree_explainer(explainer, X):
    try:
        out = explainer(X)
        sv = out.values if hasattr(out, "values") else out
    except Exception:
        sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]
    return sv


def compute_sample_level_shap(model, X_df, max_rows=1200):
    if len(X_df) > max_rows:
        X_use = X_df.sample(max_rows, random_state=42).copy()
    else:
        X_use = X_df.copy()

    X_np = X_use.values.astype("float32", copy=False)
    explainer = shap.TreeExplainer(model)
    shap_values = _shap_values_tree_explainer(explainer, X_np)
    shap_df = pd.DataFrame(shap_values, columns=X_use.columns, index=X_use.index)
    shap_imp = pd.DataFrame(
        {
            "feature": X_use.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return X_use, shap_df, shap_imp, explainer


def moving_average_curve(x, y, bins=40):
    order = np.argsort(x)
    x = np.asarray(x)[order]
    y = np.asarray(y)[order]
    if len(x) < 12:
        return x, y
    edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    xs = []
    ys = []
    for i in range(bins):
        mask = (x >= edges[i]) & (x <= edges[i + 1] if i == bins - 1 else x < edges[i + 1])
        if mask.sum() >= 3:
            xs.append(float(np.nanmean(x[mask])))
            ys.append(float(np.nanmean(y[mask])))
    return np.array(xs), np.array(ys)


def fig_observed_pred(df_oof, pred_col="predicted_oof", title="Observed vs predicted"):
    tmp = df_oof[[c for c in ["observed", pred_col] if c in df_oof.columns]].dropna()
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.scatter(tmp["observed"], tmp[pred_col], s=28, alpha=0.72)
    if not tmp.empty:
        lo = float(np.nanmin([tmp["observed"].min(), tmp[pred_col].min()]))
        hi = float(np.nanmax([tmp["observed"].max(), tmp[pred_col].max()]))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def fig_barh(df, value_col, label_col="feature", title="", top_n=15, xlabel=None):
    plot_df = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.barh(plot_df[label_col], plot_df[value_col])
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def fig_shap_driver(X_shap, shap_df, feature):
    x = pd.to_numeric(X_shap[feature], errors="coerce").values.astype(float)
    y = pd.to_numeric(shap_df[feature], errors="coerce").values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    axes[0].scatter(x, y, s=20, alpha=0.60)
    xs, ys = moving_average_curve(x, y, bins=36)
    if len(xs) > 1:
        axes[0].plot(xs, ys, linewidth=2.0)
    axes[0].axhline(0, linestyle="--", linewidth=1.0)
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("SHAP value")
    axes[0].set_title(f"Driver response: {feature}")
    axes[0].grid(alpha=0.2)

    axes[1].hist(x, bins=28, alpha=0.85)
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Observed distribution")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    return fig


def sanitize_feature_name(name: str):
    return Path(name).stem.lower().replace("1000", "")


def save_uploaded_rasters_to_temp(uploaded_rasters):
    temp_dir = tempfile.TemporaryDirectory()
    saved_paths = []
    for f in uploaded_rasters:
        out = Path(temp_dir.name) / f.name
        out.write_bytes(f.getbuffer())
        saved_paths.append(str(out))
    return temp_dir, saved_paths


def find_raster_for_feature(feature, raster_paths):
    target = sanitize_feature_name(feature)
    for p in raster_paths:
        stem = sanitize_feature_name(Path(p).stem)
        if stem == target:
            return p
    for p in raster_paths:
        stem = sanitize_feature_name(Path(p).stem)
        if target in stem or stem in target:
            return p
    raise FileNotFoundError(f"No uploaded raster matches predictor '{feature}'.")


def open_and_align_datasets(feature_names, raster_paths, resampling_name="bilinear"):
    if not feature_names:
        raise ValueError("No predictor names were provided.")
    if WarpedVRT is None or Resampling is None:
        raise RuntimeError("Raster alignment requires rasterio.vrt.WarpedVRT, which is not available in this environment.")

    stack = ExitStack()
    ds_dict = {}
    try:
        first_path = find_raster_for_feature(feature_names[0], raster_paths)
        ref_src = stack.enter_context(rasterio.open(first_path))
        ref_meta = {
            "crs": ref_src.crs,
            "transform": ref_src.transform,
            "width": ref_src.width,
            "height": ref_src.height,
        }
        ds_dict[feature_names[0]] = ref_src

        resampling_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
        }
        resampling = resampling_map.get(resampling_name, Resampling.bilinear)

        for feat in feature_names[1:]:
            path = find_raster_for_feature(feat, raster_paths)
            src = stack.enter_context(rasterio.open(path))
            same_grid = (
                src.crs == ref_meta["crs"]
                and src.transform == ref_meta["transform"]
                and src.width == ref_meta["width"]
                and src.height == ref_meta["height"]
            )
            if same_grid:
                ds_dict[feat] = src
            else:
                vrt = stack.enter_context(
                    WarpedVRT(
                        src,
                        crs=ref_meta["crs"],
                        transform=ref_meta["transform"],
                        width=ref_meta["width"],
                        height=ref_meta["height"],
                        resampling=resampling,
                    )
                )
                ds_dict[feat] = vrt

        return stack, ds_dict
    except Exception:
        stack.close()
        raise


def _make_profile(tmpl_ds, nodata=-9999.0):
    profile = tmpl_ds.profile.copy()
    profile.update(dtype="float32", count=1, compress="lzw", nodata=nodata)
    return profile


def read_predictor_block(datasets, feature_names, win):
    arrays = []
    invalid_mask = None
    for feat in feature_names:
        ds = datasets[feat]
        arr = ds.read(1, window=win).astype("float32")
        inv = ~np.isfinite(arr)
        if ds.nodata is not None:
            inv |= (arr == ds.nodata)
        invalid_mask = inv if invalid_mask is None else (invalid_mask | inv)
        arrays.append(arr)
    stack_arr = np.stack(arrays, axis=-1)
    return stack_arr, invalid_mask


def build_kfold_models(X_df, y, model_params, cv_splits=5, random_state=42):
    from sklearn.model_selection import KFold

    X = X_df.values.astype("float32", copy=False)
    splitter = KFold(n_splits=int(cv_splits), shuffle=True, random_state=int(random_state))
    models = []
    for tr_idx, _ in splitter.split(X):
        model = RandomForestRegressor(**model_params)
        model.fit(X[tr_idx], y[tr_idx])
        models.append(model)
    return models


def run_raster_prediction(best_model, feature_names, raster_paths, X_train_df, y, model_params, cv_splits=5, random_state=42, block=512):
    stack, datasets = open_and_align_datasets(feature_names, raster_paths, resampling_name="bilinear")
    try:
        first_key = feature_names[0]
        tmpl = datasets[first_key]
        out_dir = tempfile.TemporaryDirectory()

        pred_path = Path(out_dir.name) / "prediction.tif"
        mean_path = Path(out_dir.name) / "prediction_cv_mean.tif"
        std_path = Path(out_dir.name) / "prediction_cv_std.tif"

        profile = _make_profile(tmpl, nodata=-9999.0)
        fold_models = build_kfold_models(X_train_df, y, model_params, cv_splits=cv_splits, random_state=random_state)

        with rasterio.open(pred_path, "w", **profile) as dst_pred, \
             rasterio.open(mean_path, "w", **profile) as dst_mean, \
             rasterio.open(std_path, "w", **profile) as dst_std:

            for row0 in range(0, tmpl.height, block):
                for col0 in range(0, tmpl.width, block):
                    h = min(block, tmpl.height - row0)
                    w = min(block, tmpl.width - col0)
                    win = Window(col0, row0, w, h)

                    stack_arr, invalid_mask = read_predictor_block(datasets, feature_names, win)
                    Xw = stack_arr.reshape(-1, stack_arr.shape[-1]).astype("float32", copy=False)
                    valid = ~invalid_mask.reshape(-1)

                    pred = np.full(h * w, -9999.0, dtype="float32")
                    meanv = np.full(h * w, -9999.0, dtype="float32")
                    stdv = np.full(h * w, -9999.0, dtype="float32")

                    if np.any(valid):
                        pred_valid = best_model.predict(Xw[valid]).astype("float32")
                        pred[valid] = pred_valid

                        fold_preds = np.vstack([m.predict(Xw[valid]).astype("float32") for m in fold_models])
                        meanv[valid] = fold_preds.mean(axis=0)
                        stdv[valid] = fold_preds.std(axis=0)

                    dst_pred.write(pred.reshape(h, w), 1, window=win)
                    dst_mean.write(meanv.reshape(h, w), 1, window=win)
                    dst_std.write(stdv.reshape(h, w), 1, window=win)

        return out_dir, str(pred_path), str(mean_path), str(std_path)
    finally:
        stack.close()


def run_single_feature_shap_raster(model, feature_names, raster_paths, selected_feature, block=512):
    if selected_feature not in feature_names:
        raise ValueError("The selected feature is not part of the retained predictor set.")

    stack, datasets = open_and_align_datasets(feature_names, raster_paths, resampling_name="bilinear")
    try:
        tmpl = datasets[feature_names[0]]
        out_dir = tempfile.TemporaryDirectory()
        out_path = Path(out_dir.name) / f"shap_{selected_feature}.tif"
        profile = _make_profile(tmpl, nodata=-9999.0
