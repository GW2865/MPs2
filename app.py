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
        profile = _make_profile(tmpl, nodata=-9999.0)

        explainer = shap.TreeExplainer(model)
        feat_idx = feature_names.index(selected_feature)

        with rasterio.open(out_path, "w", **profile) as dst:
            for row0 in range(0, tmpl.height, block):
                for col0 in range(0, tmpl.width, block):
                    h = min(block, tmpl.height - row0)
                    w = min(block, tmpl.width - col0)
                    win = Window(col0, row0, w, h)

                    stack_arr, invalid_mask = read_predictor_block(datasets, feature_names, win)
                    Xw = stack_arr.reshape(-1, stack_arr.shape[-1]).astype("float32", copy=False)
                    valid = ~invalid_mask.reshape(-1)

                    out_block = np.full(h * w, -9999.0, dtype="float32")
                    if np.any(valid):
                        shap_values = _shap_values_tree_explainer(explainer, Xw[valid])
                        out_block[valid] = shap_values[:, feat_idx].astype("float32")

                    dst.write(out_block.reshape(h, w), 1, window=win)

        return out_dir, str(out_path)
    finally:
        stack.close()


def preview_raster_png(raster_path):
    with rasterio.open(raster_path) as ds:
        arr = ds.read(1).astype("float32")
        nodata = ds.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        im = ax.imshow(arr, cmap="viridis")
        ax.set_title(Path(raster_path).name)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig


def export_bundle(model, feature_names, config, tables):
    import zipfile
    bundle = io.BytesIO()
    with io.BytesIO() as m:
        joblib.dump({"model": model, "features": feature_names, "config": config}, m)
        model_bytes = m.getvalue()

    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.joblib", model_bytes)
        z.writestr("config.json", json.dumps(config, indent=2))
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame):
                z.writestr(f"{name}.csv", df.to_csv(index=False))
    bundle.seek(0)
    return bundle


with st.sidebar:
    st.title("MicroFragment Atlas Pro")
    st.markdown(
        '<div class="sidebar-note">A streamlined scientific app for sampling-point modelling, sample-level SHAP interpretation, and regional raster prediction with optional single-feature SHAP mapping.</div>',
        unsafe_allow_html=True,
    )
    csv_file = st.file_uploader("Upload sampling CSV", type=["csv"])
    raster_files = st.file_uploader("Upload predictor TIFF files", type=["tif", "tiff"], accept_multiple_files=True)

    st.markdown("---")
    st.subheader("Model settings")
    random_state = st.number_input("Random state", min_value=1, max_value=9999, value=42, step=1)
    n_estimators = st.slider("Number of trees", 100, 1000, 158, 1)
    max_depth_option = st.selectbox("Maximum depth", ["None", 5, 10, 14, 15, 20, 30], index=3)
    min_samples_leaf = st.slider("Minimum samples per leaf", 1, 8, 1, 1)
    min_samples_split = st.slider("Minimum samples to split", 2, 20, 2, 1)
    cv_splits = st.slider("CV splits", 3, 10, 5)
    cv_repeats = st.slider("Repeated-CV repeats", 1, 5, 3)
    spatial_blocks = st.slider("Spatial blocks", 3, 10, 5)

    st.markdown("---")
    st.subheader("Interpretation")
    enable_shap = st.toggle("Enable SHAP analysis", value=False)
    compute_perm = st.toggle("Compute permutation importance", value=True)

st.markdown(
    """
    <div class="hero">
      <div class="kicker">Research-grade environmental modelling</div>
      <h1>MicroFragment Atlas Pro</h1>
      <p>Fit a robust random forest model from sample data, compare repeated and spatial cross-validation, inspect sample-level driver responses with SHAP, generate regional GeoTIFF predictions, and optionally map a single predictor’s SHAP contribution across space.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Overview",
        "Data setup",
        "Model and validation",
        "Sample-level SHAP",
        "Raster prediction",
        "Export",
    ]
)
tab_overview, tab_data, tab_model, tab_shap, tab_raster, tab_export = tabs

if csv_file is None:
    with tab_overview:
        st.info("Upload a sampling CSV from the sidebar to begin.")
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="glass"><h4 class="section-title">Stable training workflow</h4><p class="tiny">The app uses a streamlined random forest pipeline designed for deployment stability on Streamlit Cloud.</p></div>', unsafe_allow_html=True)
        c2.markdown('<div class="glass"><h4 class="section-title">Interpretation before mapping</h4><p class="tiny">Sample-level SHAP is placed before raster prediction so the driver mechanisms can be inspected before regional extrapolation.</p></div>', unsafe_allow_html=True)
        c3.markdown('<div class="glass"><h4 class="section-title">Spatial SHAP after prediction</h4><p class="tiny">After successful raster prediction, a single selected predictor can be mapped as a spatial SHAP GeoTIFF.</p></div>', unsafe_allow_html=True)
    st.stop()

df = load_csv(csv_file)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", int(df.shape[0]))
    c2.metric("Columns", int(df.shape[1]))
    c3.metric("Missing cells", int(df.isna().sum().sum()))
    c4.metric("Uploaded TIFFs", len(raster_files) if raster_files else 0)
    st.subheader("Sampling table preview")
    st.dataframe(df.head(15), height=420, width="stretch")

with tab_data:
    st.subheader("Dataset configuration")
    all_cols = list(df.columns)
    default_target = all_cols.index("MPs") if "MPs" in all_cols else 0
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Response variable", all_cols, index=default_target)
    x_default = ([""] + all_cols).index("Longitude") if "Longitude" in all_cols else 0
    y_default = ([""] + all_cols).index("Latitude") if "Latitude" in all_cols else 0
    x_coord = c2.selectbox("X coordinate", [""] + all_cols, index=x_default)
    y_coord = c3.selectbox("Y coordinate", [""] + all_cols, index=y_default)

    possible_drop = [c for c in all_cols if c not in {target, x_coord, y_coord}]
    drop_cols = st.multiselect("Exclude columns from predictors", possible_drop, default=[])

    try:
        X_df, y, coord_df = make_clean_training_table(df, target, x_coord or None, y_coord or None, drop_cols)
        s1, s2, s3 = st.columns(3)
        s1.metric("Usable rows", int(len(X_df)))
        s2.metric("Retained predictors", int(X_df.shape[1]))
        s3.metric("Coordinate columns", 2 if coord_df is not None else 0)

        st.markdown("**Retained predictor list**")
        st.dataframe(pd.DataFrame({"feature": X_df.columns}), height=320, width="stretch")
    except Exception as e:
        st.error(f"Dataset configuration error: {e}")
        X_df, y, coord_df = None, None, None

run_model = False
with tab_model:
    run_model = st.button("Run modelling workflow", type="primary", width="stretch")

if "results" not in st.session_state:
    st.session_state["results"] = None

if run_model:
    try:
        X_df, y, coord_df = make_clean_training_table(df, target, x_coord or None, y_coord or None, drop_cols)
        model_params = {
            "n_estimators": int(n_estimators),
            "random_state": int(random_state),
            "max_depth": None if str(max_depth_option) == "None" else int(max_depth_option),
            "min_samples_leaf": int(min_samples_leaf),
            "min_samples_split": int(min_samples_split),
            "n_jobs": 1,
        }

        with st.spinner("Training model and computing validation outputs..."):
            best_model = fit_rf_model(
                X_df,
                y,
                random_state=random_state,
                n_estimators=n_estimators,
                max_depth=None if str(max_depth_option) == "None" else int(max_depth_option),
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
            )
            rep_folds, rep_oof = evaluate_repeated_cv(
                X_df, y, model_params, random_state=random_state, cv_splits=cv_splits, cv_repeats=cv_repeats
            )
            spatial_folds, spatial_oof = evaluate_spatial_cv(
                X_df, y, coord_df, model_params, random_state=random_state, spatial_blocks=spatial_blocks
            )

            rf_imp = pd.DataFrame(
                {"feature": X_df.columns, "rf_importance": best_model.feature_importances_}
            ).sort_values("rf_importance", ascending=False).reset_index(drop=True)

            perm_imp = None
            if compute_perm:
                perm_imp = compute_permutation_importance_table(best_model, X_df, y, random_state=random_state)

            X_shap = shap_df = shap_imp = shap_explainer = None
            if enable_shap:
                X_shap, shap_df, shap_imp, shap_explainer = compute_sample_level_shap(best_model, X_df)

        st.session_state["results"] = {
            "target": target,
            "x_coord": x_coord,
            "y_coord": y_coord,
            "drop_cols": drop_cols,
            "X_df": X_df,
            "y": y,
            "coord_df": coord_df,
            "model_params": model_params,
            "best_model": best_model,
            "rep_folds": rep_folds,
            "rep_oof": rep_oof,
            "spatial_folds": spatial_folds,
            "spatial_oof": spatial_oof,
            "rf_imp": rf_imp,
            "perm_imp": perm_imp,
            "X_shap": X_shap,
            "shap_df": shap_df,
            "shap_imp": shap_imp,
            "raster_outputs": None,
            "shap_raster_output": None,
        }
        st.success("Modelling workflow completed successfully.")
    except Exception as e:
        st.session_state["results"] = None
        st.error(f"Workflow failed: {e}")

res = st.session_state["results"]

with tab_model:
    if res is None:
        st.info("Configure the table in the Data setup tab, then run the modelling workflow.")
    else:
        st.subheader("Model summary")
        c1, c2, c3 = st.columns(3)
        rep_mean_r2 = float(res["rep_folds"]["R2"].mean())
        c1.metric("Retained predictors", int(res["X_df"].shape[1]))
        c2.metric("Repeated-CV mean R²", f"{rep_mean_r2:.3f}")
        c3.metric("Trees", int(res["model_params"]["n_estimators"]))

        st.markdown("**Random forest parameters**")
        st.json(res["model_params"])

        st.subheader("Repeated cross-validation")
        left, right = st.columns([1.0, 1.1])
        rep_summary = (
            res["rep_folds"][["R2", "RMSE", "MAE"]]
            .agg(["mean", "std"])
            .T.reset_index()
            .rename(columns={"index": "metric"})
        )
        left.dataframe(rep_summary, width="stretch")
        right.pyplot(fig_observed_pred(res["rep_oof"], "predicted_oof", "Repeated-CV prediction"))
        st.dataframe(res["rep_folds"], height=240, width="stretch")

        st.subheader("Spatial cross-validation")
        if res["spatial_folds"] is None:
            st.warning("Spatial cross-validation was skipped because valid coordinate columns were not available.")
        else:
            s1, s2 = st.columns([1.0, 1.1])
            spatial_summary = (
                res["spatial_folds"][["R2", "RMSE", "MAE"]]
                .agg(["mean", "std"])
                .T.reset_index()
                .rename(columns={"index": "metric"})
            )
            s1.dataframe(spatial_summary, width="stretch")
            s2.pyplot(fig_observed_pred(res["spatial_oof"], "predicted_spatial_oof", "Spatial-CV prediction"))
            st.dataframe(res["spatial_folds"], height=220, width="stretch")

with tab_shap:
    if res is None:
        st.info("Run the modelling workflow first.")
    elif not enable_shap:
        st.info("SHAP analysis is currently disabled. Turn on 'Enable SHAP analysis' in the sidebar and rerun the workflow.")
    else:
        st.subheader("Global interpretation")
        cols = st.columns(3 if res["perm_imp"] is not None else 2)
        cols[0].pyplot(fig_barh(res["rf_imp"], "rf_importance", "feature", "Random forest importance", 15, "Importance"))
        if res["perm_imp"] is not None:
            cols[1].pyplot(fig_barh(res["perm_imp"], "perm_importance_mean", "feature", "Permutation importance", 15, "Mean importance"))
            cols[2].pyplot(fig_barh(res["shap_imp"], "mean_abs_shap", "feature", "SHAP importance", 15, "Mean |SHAP|"))
        else:
            cols[1].pyplot(fig_barh(res["shap_imp"], "mean_abs_shap", "feature", "SHAP importance", 15, "Mean |SHAP|"))

        st.subheader("Single-feature driver response")
        feature_for_driver = st.selectbox("Select a predictor for detailed SHAP interpretation", list(res["X_shap"].columns))
        st.pyplot(fig_shap_driver(res["X_shap"], res["shap_df"], feature_for_driver))

        st.markdown("**Driver-response table**")
        driver_table = pd.DataFrame(
            {
                "feature_value": res["X_shap"][feature_for_driver].values,
                "shap_value": res["shap_df"][feature_for_driver].values,
            }
        )
        st.dataframe(driver_table.head(500), height=280, width="stretch")

        with st.expander("Show all interpretation tables"):
            e1, e2 = st.columns(2)
            e1.dataframe(res["shap_imp"], height=260, width="stretch")
            if res["perm_imp"] is not None:
                e2.dataframe(res["perm_imp"], height=260, width="stretch")

with tab_raster:
    if res is None:
        st.info("Run the modelling workflow first.")
    else:
        st.subheader("Regional raster prediction")
        retained = list(res["X_df"].columns)
        st.caption("Upload one TIFF per retained predictor. Raster file names should match predictor names, optionally with a 1000 suffix.")
        st.write("Retained predictors:", ", ".join(retained))

        if not raster_files:
            st.warning("No predictor TIFF files have been uploaded yet.")
        else:
            st.write("Uploaded TIFF files:", ", ".join([f.name for f in raster_files]))
            run_raster = st.button("Run raster prediction and uncertainty export", width="stretch")
            if run_raster:
                with st.spinner("Generating prediction, cross-validated mean, and uncertainty rasters..."):
                    raster_tempdir, saved_rasters = save_uploaded_rasters_to_temp(raster_files)
                    try:
                        out_dir, pred_path, mean_path, std_path = run_raster_prediction(
                            res["best_model"],
                            retained,
                            saved_rasters,
                            res["X_df"],
                            res["y"],
                            res["model_params"],
                            cv_splits=cv_splits,
                            random_state=random_state,
                            block=512,
                        )
                        res["raster_outputs"] = {
                            "temp_rasters": raster_tempdir,
                            "out_dir": out_dir,
                            "prediction": pred_path,
                            "mean": mean_path,
                            "std": std_path,
                            "saved_rasters": saved_rasters,
                        }
                        st.success("Raster prediction completed successfully.")
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
                    width="stretch",
                )
            with p2:
                st.pyplot(preview_raster_png(outputs["std"]))
                st.download_button(
                    "Download prediction_cv_std.tif",
                    data=Path(outputs["std"]).read_bytes(),
                    file_name="prediction_cv_std.tif",
                    mime="application/octet-stream",
                    width="stretch",
                )

            st.download_button(
                "Download prediction_cv_mean.tif",
                data=Path(outputs["mean"]).read_bytes(),
                file_name="prediction_cv_mean.tif",
                mime="application/octet-stream",
                width="stretch",
            )

            st.markdown("---")
            st.subheader("Single-feature SHAP spatial map")
            if not enable_shap:
                st.info("Enable SHAP analysis in the sidebar and rerun the modelling workflow to use spatial SHAP mapping.")
            else:
                shap_feature = st.selectbox("Select a predictor for spatial SHAP mapping", retained)
                if st.button("Generate SHAP raster for selected feature", width="stretch"):
                    with st.spinner("Computing the selected feature's SHAP contribution across space..."):
                        try:
                            shap_out_dir, shap_path = run_single_feature_shap_raster(
                                res["best_model"],
                                retained,
                                outputs["saved_rasters"],
                                shap_feature,
                                block=512,
                            )
                            res["shap_raster_output"] = {"out_dir": shap_out_dir, "path": shap_path, "feature": shap_feature}
                            st.success("Spatial SHAP raster completed.")
                        except Exception as e:
                            st.error(f"Spatial SHAP raster failed: {e}")

                if res.get("shap_raster_output"):
                    shap_out = res["shap_raster_output"]
                    st.pyplot(preview_raster_png(shap_out["path"]))
                    st.download_button(
                        f"Download shap_{shap_out['feature']}.tif",
                        data=Path(shap_out["path"]).read_bytes(),
                        file_name=f"shap_{shap_out['feature']}.tif",
                        mime="application/octet-stream",
                        width="stretch",
                    )

with tab_export:
    if res is None:
        st.info("Run the modelling workflow first.")
    else:
        st.subheader("Export model bundle")
        config = {
            "target": res["target"],
            "x_coord": res["x_coord"],
            "y_coord": res["y_coord"],
            "drop_cols": res["drop_cols"],
            "random_state": int(random_state),
            "n_estimators": int(n_estimators),
            "max_depth": str(max_depth_option),
            "min_samples_leaf": int(min_samples_leaf),
            "min_samples_split": int(min_samples_split),
            "cv_splits": int(cv_splits),
            "cv_repeats": int(cv_repeats),
            "spatial_blocks": int(spatial_blocks),
            "enable_shap": bool(enable_shap),
            "compute_permutation_importance": bool(compute_perm),
        }
        tables = {
            "repeated_cv_metrics": res["rep_folds"],
            "repeated_cv_oof": res["rep_oof"],
            "rf_importance": res["rf_imp"],
        }
        if res["spatial_folds"] is not None:
            tables["spatial_cv_metrics"] = res["spatial_folds"]
            tables["spatial_cv_oof"] = res["spatial_oof"]
        if res["perm_imp"] is not None:
            tables["permutation_importance"] = res["perm_imp"]
        if res["shap_imp"] is not None:
            tables["shap_importance"] = res["shap_imp"]

        bundle = export_bundle(
            res["best_model"],
            list(res["X_df"].columns),
            config,
            tables,
        )
        st.download_button(
            "Download model bundle (.zip)",
            data=bundle,
            file_name="microfragment_atlas_bundle.zip",
            mime="application/zip",
            width="stretch",
        )
        st.markdown("**Included in the bundle**")
        st.write(
            "The bundle contains the fitted model, retained predictor list, the main configuration, and key validation and interpretation tables. Raster GeoTIFF outputs are downloaded separately from the Raster prediction tab."
        )
