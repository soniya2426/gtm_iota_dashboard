import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import statsmodels.api as sm


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="IOTA Water UAE | GTM Analytics Dashboard", layout="wide")

DATA_PATH = "data/gip final data.csv"  # <-- your exact filename with spaces


# =========================
# LOAD DATA
# =========================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

df_raw = load_data(DATA_PATH)
if df_raw is None:
    st.error(
        f"Dataset not found at `{DATA_PATH}`.\n\n"
        "Fix:\n"
        "1) In GitHub create folder `data/`\n"
        "2) Upload your CSV inside `data/`\n"
        "3) Ensure filename is exactly `gip final data.csv`\n"
        "4) Streamlit Cloud → Manage app → Reboot"
    )
    st.stop()


# =========================
# CLEAN COLUMN NAMES
# =========================
def standardize_col(col: str) -> str:
    col = str(col).replace("\ufeff", "").strip().lower()
    col = re.sub(r"[^\w\s]", " ", col)
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")

df = df_raw.copy()
df.columns = [standardize_col(c) for c in df.columns]

# drop common junk cols
for junk in ["column1", "unnamed_0", "unnamed_1", "index"]:
    if junk in df.columns:
        df = df.drop(columns=[junk], errors="ignore")


# =========================
# IMPUTATION
# =========================
def median_impute(s: pd.Series) -> pd.Series:
    return s.fillna(s.median()) if not s.dropna().empty else s

def mode_impute(s: pd.Series) -> pd.Series:
    return s.fillna(s.dropna().mode().iloc[0]) if not s.dropna().empty else s

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

for c in num_cols:
    df[c] = median_impute(df[c])
for c in cat_cols:
    df[c] = mode_impute(df[c])

numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
all_columns = df.columns.tolist()


# =========================
# MODEL ENCODER (for clustering/regression)
# =========================
def encode_for_modeling(df_in: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df_in[cols].copy()

    # try to coerce mostly-numeric object cols into numeric
    for c in X.columns:
        if X[c].dtype == "object":
            coerced = pd.to_numeric(X[c], errors="coerce")
            if coerced.notna().mean() > 0.80:
                X[c] = coerced

    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]

    for c in num:
        X[c] = median_impute(X[c])
    for c in cat:
        X[c] = mode_impute(X[c])

    if cat:
        X = pd.get_dummies(X, columns=cat, drop_first=False)

    return X


# =========================
# HEADER
# =========================
st.title("IOTA Water UAE | GTM Analytics Dashboard")
st.caption("Dynamic GTM insights: KPIs, correlation, regression, STP segmentation, perceptual mapping, and trend lines.")


# =========================
# SIDEBAR NAV
# =========================
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        [
            "Data Overview",
            "KPI Metrics",
            "Correlation Heatmap",
            "Regression",
            "Segmentation (STP)",
            "Positioning & Perceptual Mapping",
        ],
        index=1
    )
    st.divider()
    st.caption("Dataset path used:")
    st.code(DATA_PATH)


# =========================
# PAGE: DATA OVERVIEW
# =========================
if page == "Data Overview":
    st.subheader("Data Overview & Health Checks")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Numeric columns", f"{len(numeric_candidates):,}")
    c4.metric("Missing cells (after impute)", f"{int(df.isna().sum().sum()):,}")

    with st.expander("Show column types"):
        dtypes_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
        st.dataframe(dtypes_df, use_container_width=True, height=350)

    with st.expander("Preview data"):
        st.dataframe(df.head(40), use_container_width=True)

    st.caption(
        "Insight: Confirms the dataset loads, columns are standardized, and missing values are handled. "
        "GTM implication: stable data health enables reliable strategy decisions from the rest of the dashboard."
    )


# =========================
# PAGE: KPI METRICS (Dynamic)
# =========================
elif page == "KPI Metrics":
    st.subheader("KPI Metrics (Executive Snapshot)")

    if not numeric_candidates:
        st.warning("No numeric columns found. KPI metrics require numeric fields.")
        st.stop()

    # Pick KPI columns dynamically
    cA, cB = st.columns(2)
    with cA:
        kpi_main = st.selectbox("Primary KPI metric (numeric)", options=numeric_candidates, index=0)
    with cB:
        kpi_secondary = st.selectbox("Secondary KPI metric (numeric)", options=numeric_candidates, index=min(1, len(numeric_candidates)-1))

    avg_main = float(df[kpi_main].mean())
    med_main = float(df[kpi_main].median())
    avg_sec = float(df[kpi_secondary].mean())
    med_sec = float(df[kpi_secondary].median())

    t1, t2, t3, t4 = st.columns(4)
    t1.metric(f"Avg {kpi_main}", f"{avg_main:,.2f}")
    t2.metric(f"Median {kpi_main}", f"{med_main:,.2f}")
    t3.metric(f"Avg {kpi_secondary}", f"{avg_sec:,.2f}")
    t4.metric(f"Median {kpi_secondary}", f"{med_sec:,.2f}")

    st.caption(
        "Insight: KPIs summarize market value potential and intensity on selected numeric measures. "
        "GTM implication: use KPI gaps to prioritize pricing tier and where to focus distribution first."
    )

    st.divider()

    # Dynamic line chart
    st.markdown("### Line Chart (Dynamic)")
    x_axis = st.selectbox("X-axis (numeric)", options=numeric_candidates, index=0)
    y_axis = st.selectbox("Y-axis (numeric)", options=[c for c in numeric_candidates if c != x_axis], index=0)

    tmp = df[[x_axis, y_axis]].dropna()
    if tmp.empty:
        st.info("Not enough data for this selection.")
    else:
        tmp = tmp.groupby(x_axis)[y_axis].mean().reset_index().sort_values(x_axis)
        fig = px.line(tmp, x=x_axis, y=y_axis, markers=True, title=f"Average {y_axis} vs {x_axis}")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Insight: Shows how the selected outcome changes across the selected axis. "
            "GTM implication: identify high-value bands and design offers/pricing to win them."
        )

    # Spend concentration style chart (optional)
    st.markdown("### Distribution (Optional)")
    dist_col = st.selectbox("Distribution column (numeric)", options=numeric_candidates, index=0)
    figd = px.histogram(df, x=dist_col, nbins=20, title=f"Distribution of {dist_col}")
    figd.update_layout(height=380)
    st.plotly_chart(figd, use_container_width=True)

    st.caption(
        "Insight: Distribution shows whether the market is concentrated (few high values) or broad-based. "
        "GTM implication: concentrated distributions support premium niche targeting; broad distributions support mass play."
    )


# =========================
# PAGE: CORRELATION HEATMAP (Dynamic)
# =========================
elif page == "Correlation Heatmap":
    st.subheader("Correlation Heatmap (Dynamic)")

    if len(numeric_candidates) < 3:
        st.warning("Need at least 3 numeric columns for a correlation heatmap.")
        st.stop()

    selected = st.multiselect(
        "Choose numeric attributes for correlation",
        options=numeric_candidates,
        default=numeric_candidates[:10]
    )

    if len(selected) < 3:
        st.warning("Select at least 3 numeric columns.")
        st.stop()

    corr = df[selected].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap")
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Insight: Correlation shows which drivers move together in consumer perception. "
        "GTM implication: build positioning around coherent bundles of drivers, not scattered claims."
    )


# =========================
# PAGE: REGRESSION (Safe + Formatted)
# =========================
elif page == "Regression":
    st.subheader("Regression (Dynamic)")

    if len(numeric_candidates) == 0:
        st.error("No numeric columns available for regression outcome.")
        st.stop()

    y_col = st.selectbox("Outcome (dependent variable)", options=numeric_candidates, index=0)

    X_cols = st.multiselect(
        "Predictors (independent) — numeric + categorical allowed",
        options=all_columns,
        default=[c for c in numeric_candidates if c != y_col][:3]
    )

    if len(X_cols) < 2:
        st.warning("Pick at least 2 predictors.")
        st.stop()

    model_df = df[[y_col] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if model_df.empty:
        st.error("No usable rows after dropping missing values. Reduce selected columns.")
        st.stop()

    X = pd.get_dummies(model_df[X_cols], drop_first=False)
    X = sm.add_constant(X)
    y = model_df[y_col]

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        st.error(f"Regression failed: {e}")
        st.stop()

    results = pd.DataFrame({
        "feature": model.params.index,
        "coef": model.params.values,
        "p_value_raw": model.pvalues.values
    }).sort_values("p_value_raw")

    # format to 3 decimals
    results["coef"] = results["coef"].round(3)
    results["p_value"] = results["p_value_raw"].apply(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
    results = results.drop(columns=["p_value_raw"])

    c1, c2 = st.columns([1.25, 0.75])
    c1.dataframe(results.head(60), use_container_width=True, height=520)

    c2.metric("R-squared", f"{model.rsquared:.3f}")
    c2.metric("Observations", f"{len(model_df):,}")

    sig_mask = (model.pvalues.index != "const") & (model.pvalues < 0.05)
    sig_features = model.pvalues[sig_mask].sort_values().head(8).index.tolist()

    c2.markdown("**Top significant drivers (p < 0.05):**")
    if not sig_features:
        c2.write("No significant drivers in this configuration.")
    else:
        for f in sig_features:
            coef = float(model.params[f])
            direction = "↑" if coef > 0 else "↓"
            c2.write(f"- {direction} `{f}` (coef={coef:.3f})")

    st.caption(
        "Insight: Regression quantifies which selected attributes are associated with the selected outcome. "
        "GTM implication: use significant drivers to justify pricing, messaging, and channel investment priorities."
    )

    st.markdown("### Linear Fit: Actual vs Predicted")
    pred = model.predict(X)
    fit_df = pd.DataFrame({"actual": y.values, "predicted": pred.values})

    figfit = px.scatter(fit_df, x="predicted", y="actual", trendline="ols", title="Actual vs Predicted (trendline)")
    figfit.update_layout(height=450)
    st.plotly_chart(figfit, use_container_width=True)

    st.caption(
        "Insight: Strong alignment indicates the selected drivers explain the outcome well. "
        "GTM implication: better model fit increases confidence in your driver-led GTM strategy."
    )


# =========================
# PAGE: SEGMENTATION (STP) - KMeans
# =========================
elif page == "Segmentation (STP)":
    st.subheader("STP Segmentation (KMeans Clustering)")

    seg_cols = st.multiselect(
        "Select attributes for clustering (mix numeric + categorical allowed)",
        options=all_columns,
        default=numeric_candidates[:6] if len(numeric_candidates) >= 6 else numeric_candidates
    )

    if len(seg_cols) < 3:
        st.warning("Pick at least 3 attributes for clustering.")
        st.stop()

    k = st.slider("Number of segments (K)", 3, 8, 4)

    Xseg = encode_for_modeling(df, seg_cols)
    Xs = StandardScaler().fit_transform(Xseg)

    km = KMeans(n_clusters=k, random_state=42, n_init=25)
    seg_labels = km.fit_predict(Xs)

    df_seg = df.copy()
    df_seg["segment"] = seg_labels
    st.session_state["df_seg"] = df_seg  # reuse in perceptual map

    sizes = df_seg["segment"].value_counts().sort_index().reset_index()
    sizes.columns = ["segment", "respondents"]

    a, b = st.columns([0.7, 1.3])
    a.dataframe(sizes, use_container_width=True, height=240)
    fig = px.bar(sizes, x="segment", y="respondents", title="Segment Sizes")
    fig.update_layout(height=320)
    b.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Insight: Segment size reveals where scale exists versus niche opportunities. "
        "GTM implication: pick 1–2 primary segments first to keep positioning sharp and execution focused."
    )

    # Segment profile line (for numeric columns only)
    numeric_in_seg = [c for c in seg_cols if c in numeric_candidates]
    if len(numeric_in_seg) >= 2:
        st.markdown("### Line: Segment Profiles (Selected Numeric Attributes)")
        prof = df_seg.groupby("segment")[numeric_in_seg].mean().reset_index()
        prof_long = prof.melt(id_vars="segment", var_name="attribute", value_name="avg_value")

        figline = px.line(prof_long, x="attribute", y="avg_value", color="segment", markers=True,
                          title="Segment Profiles Across Selected Attributes")
        figline.update_layout(height=520)
        st.plotly_chart(figline, use_container_width=True)

        st.caption(
            "Insight: Each segment shows a distinct ‘fingerprint’ across the selected drivers. "
            "GTM implication: tailor value proposition and offers per segment, instead of a one-size-fits-all message."
        )

    with st.expander("Download segmented dataset"):
        csv = df_seg.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV with segment labels", csv, file_name="iota_segmented_output.csv", mime="text/csv")


# =========================
# PAGE: POSITIONING & PERCEPTUAL MAPPING (PCA)
# =========================
elif page == "Positioning & Perceptual Mapping":
    st.subheader("Positioning & Perceptual Mapping (PCA)")

    if len(numeric_candidates) < 3:
        st.warning("Need at least 3 numeric attributes for PCA mapping.")
        st.stop()

    pca_cols = st.multiselect(
        "Select numeric attributes for perceptual map (PCA)",
        options=numeric_candidates,
        default=numeric_candidates[:8] if len(numeric_candidates) >= 8 else numeric_candidates
    )

    if len(pca_cols) < 3:
        st.warning("Select at least 3 numeric attributes.")
        st.stop()

    Xp = df[pca_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if Xp.empty:
        st.error("No usable rows for PCA. Reduce selected columns.")
        st.stop()

    Xps = StandardScaler().fit_transform(Xp)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xps)

    df_map = df.loc[Xp.index].copy()
    df_map["pc1"] = coords[:, 0]
    df_map["pc2"] = coords[:, 1]

    explained = pca.explained_variance_ratio_
    st.caption(f"PCA variance explained: PC1={explained[0]*100:.1f}% | PC2={explained[1]*100:.1f}%")

    overlay = st.checkbox("Overlay KMeans clusters on perceptual map", value=True)
    if overlay:
        k = st.slider("K (overlay clusters)", 3, 8, 4)
        km = KMeans(n_clusters=k, random_state=42, n_init=25)
        df_map["segment"] = km.fit_predict(Xps)
        color_col = "segment"
    else:
        color_col = None

    fig = px.scatter(df_map, x="pc1", y="pc2", color=color_col, opacity=0.75,
                     title="Perceptual Map (PCA on selected attributes)")
    fig.update_layout(height=650, xaxis_title="Perceptual Axis 1 (PC1)", yaxis_title="Perceptual Axis 2 (PC2)")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Insight: This map shows how consumers distribute in preference space based on the attributes selected. "
        "GTM implication: position IOTA to win the most attractive zone and align messaging to the drivers defining it."
    )

    st.markdown("### Biplot: Attribute Loadings (Axis meaning)")
    loadings = pca.components_.T
    load_df = pd.DataFrame(loadings, index=pca_cols, columns=["pc1_loading", "pc2_loading"]).reset_index()
    load_df.rename(columns={"index": "attribute"}, inplace=True)

    arrow_scale = st.slider("Arrow scale (visibility)", 2, 14, 7)

    fig2 = go.Figure()
    for _, r in load_df.iterrows():
        fig2.add_trace(
            go.Scatter(
                x=[0, r["pc1_loading"] * arrow_scale],
                y=[0, r["pc2_loading"] * arrow_scale],
                mode="lines+markers+text",
                text=["", r["attribute"]],
                textposition="top center",
                showlegend=False
            )
        )
    fig2.update_layout(
        title="Attribute loadings on PC1 and PC2",
        height=650,
        xaxis_title="PC1 loading",
        yaxis_title="PC2 loading"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Insight: Arrows show which attributes define each axis and in what direction. "
        "GTM implication: build IOTA’s positioning around the attributes that point toward your target zone."
    )

    if overlay and "segment" in df_map.columns:
        st.markdown("### Segment Centroids (Executive summary)")
        centroids = df_map.groupby("segment")[["pc1", "pc2"]].mean().reset_index()
        centroids["size"] = df_map["segment"].value_counts().sort_index().values

        fig3 = px.scatter(centroids, x="pc1", y="pc2", size="size", color="segment", text="segment",
                          title="Segment centroids on perceptual map (size-weighted)")
        fig3.update_traces(textposition="top center")
        fig3.update_layout(height=600)
        st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            "Insight: Centroids summarize each cluster’s center of gravity in preference space. "
            "GTM implication: target the centroid with the best value potential and differentiate clearly."
        )

