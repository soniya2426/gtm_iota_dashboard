
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


# ======================
# App config
# ======================
st.set_page_config(page_title="IOTA Water UAE | GTM Dashboard", layout="wide")
st.title("IOTA Water UAE | GTM Analytics Dashboard")
st.caption("Dynamic GTM insights: KPIs, correlation, regression, STP segmentation, perceptual maps, and line charts.")

DATA_PATH = "data/gip_final_data.csv"


# ======================
# Load data
# ======================
@st.cache_data(show_spinner=False)
def load_data(path: str):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

df_raw = load_data(DATA_PATH)
if df_raw is None:
    st.error(
        f"Dataset not found at `{DATA_PATH}`.\n\n"
        "Fix:\n"
        "1) Create `data/` folder in repo\n"
        "2) Upload CSV named exactly `gip_final_data.csv`\n"
        "3) Reboot/redeploy in Streamlit Cloud"
    )
    st.stop()


# ======================
# Helpers
# ======================
def standardize_col(col: str) -> str:
    col = str(col).replace("\ufeff", "")
    col = col.strip().lower()
    col = re.sub(r"[^\w\s]", " ", col)
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")

def median_impute(s: pd.Series) -> pd.Series:
    return s.fillna(s.median()) if not s.dropna().empty else s

def mode_impute(s: pd.Series) -> pd.Series:
    return s.fillna(s.dropna().mode().iloc[0]) if not s.dropna().empty else s

def find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def spend_to_aed(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()

    if "below" in s and "50" in s:
        return 25.0
    if "50" in s and "100" in s:
        return 75.0
    if "100" in s and "200" in s:
        return 150.0
    if "200" in s and "300" in s:
        return 250.0
    if "300" in s and "500" in s:
        return 400.0
    if "500" in s or "above" in s:
        return 600.0

    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan

def freq_to_score(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if "more than once a week" in s:
        return 5
    if "once a week" in s:
        return 4
    if "fortnight" in s or "fortnite" in s:
        return 3
    if "once a month" in s or "month" in s:
        return 2
    if "rare" in s:
        return 1
    if "never" in s:
        return 0
    return np.nan

def yesno_to_bin(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ["yes", "y", "true", "1"]:
        return 1.0
    if s in ["no", "n", "false", "0"]:
        return 0.0
    return np.nan

def split_multi(x) -> list:
    if pd.isna(x):
        return []
    return [i.strip() for i in str(x).split(",") if i.strip()]

def encode_for_modeling(df_in: pd.DataFrame, cols: list) -> pd.DataFrame:
    X = df_in[cols].copy()

    # numeric coercion for mostly-numeric object columns
    for c in X.columns:
        if X[c].dtype == "object":
            coerced = pd.to_numeric(X[c], errors="coerce")
            if coerced.notna().mean() > 0.75:
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


# ======================
# Clean data
# ======================
df = df_raw.copy()
df.columns = [standardize_col(c) for c in df.columns]

# drop common junk index cols
for junk in ["column1", "unnamed_0", "unnamed_1"]:
    if junk in df.columns:
        df = df.drop(columns=[junk], errors="ignore")

cols = df.columns.tolist()

# detect spend/freq columns (if present)
col_spend = find_col(cols, [
    "what_is_your_average_monthly_spent_on_water_in_a_month",
    "average_monthly_spent_on_water",
    "monthly_spent_on_water"
])
col_freq = find_col(cols, [
    "how_often_do_you_purchase_packaged_drinking_water",
    "purchase_frequency"
])
col_eatout = find_col(cols, [
    "how_often_do_you_eat_out",
    "eat_out_frequency"
])
col_buy_eatout = find_col(cols, [
    "do_you_buy_water_while_eating_out",
    "buy_water_while_eating_out"
])
col_channel = find_col(cols, [
    "where_do_you_usually_buy_bottled_water",
    "purchase_channel"
])
col_pack = find_col(cols, [
    "size_of_bottled_water",
    "pack_size"
])
col_awareness = find_col(cols, [
    "which_brands_are_you_aware_of",
    "brand_awareness"
])
col_brand_buy = find_col(cols, [
    "which_brands_of_bottled_water_do_you_purchase_most_frequently",
    "most_frequently_purchased_brand",
    "most_purchased_brand"
])

# engineered numeric features (only if source column exists)
if col_spend:
    df["monthly_spend_aed"] = df[col_spend].apply(spend_to_aed)
else:
    df["monthly_spend_aed"] = np.nan

if col_freq:
    df["purchase_freq_score"] = df[col_freq].apply(freq_to_score)
else:
    df["purchase_freq_score"] = np.nan

if col_eatout:
    df["eatout_freq_score"] = df[col_eatout].apply(freq_to_score)
else:
    df["eatout_freq_score"] = np.nan

if col_buy_eatout:
    df["buys_water_when_eating_out"] = df[col_buy_eatout].apply(yesno_to_bin)
else:
    df["buys_water_when_eating_out"] = np.nan

# impute
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]
for c in num_cols:
    df[c] = median_impute(df[c])
for c in cat_cols:
    df[c] = mode_impute(df[c])

numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
all_columns = df.columns.tolist()


# ======================
# Sidebar navigation + global filters
# ======================
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        [
            "Data Overview",
            "KPI Metrics",
            "Consumer Insights",
            "Correlation Heatmap",
            "Regression",
            "Segmentation (STP)",
            "Positioning & Perceptual Mapping",
        ],
        index=1
    )
    st.divider()
    st.caption("Dataset path:")
    st.code(DATA_PATH)


# ======================
# PAGE: Data Overview
# ======================
if page == "Data Overview":
    st.subheader("Data Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Numeric columns", f"{len(numeric_candidates):,}")

    with st.expander("Preview data"):
        st.dataframe(df.head(30), use_container_width=True)

    st.caption(
        "Insight: Confirms the dataset is loaded and cleaned successfully. "
        "GTM implication: stable loading means the rest of the analytics will run reliably on Streamlit Cloud."
    )


# ======================
# PAGE: KPI Metrics (Dynamic)
# ======================
elif page == "KPI Metrics":
    st.subheader("KPI Metrics (Executive Snapshot)")

    # Core KPI tiles (safe even if spend/freq missing)
    avg_spend = float(df["monthly_spend_aed"].mean()) if "monthly_spend_aed" in df.columns else np.nan
    med_spend = float(df["monthly_spend_aed"].median()) if "monthly_spend_aed" in df.columns else np.nan
    avg_freq = float(df["purchase_freq_score"].mean()) if "purchase_freq_score" in df.columns else np.nan
    eatout_buy_pct = float(df["buys_water_when_eating_out"].mean() * 100) if "buys_water_when_eating_out" in df.columns else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Monthly Spend (AED proxy)", f"{avg_spend:,.0f}" if not np.isnan(avg_spend) else "N/A")
    c2.metric("Median Monthly Spend (AED proxy)", f"{med_spend:,.0f}" if not np.isnan(med_spend) else "N/A")
    c3.metric("Avg Purchase Frequency Score", f"{avg_freq:.2f}" if not np.isnan(avg_freq) else "N/A")
    c4.metric("% Buy Water While Eating Out", f"{eatout_buy_pct:.1f}%" if not np.isnan(eatout_buy_pct) else "N/A")

    st.caption(
        "Insight: These KPIs summarize category intensity (frequency), value potential (spend), and out-of-home opportunity. "
        "GTM implication: they guide pricing tier, channel priorities, and whether HoReCa is a real growth lever."
    )

    st.divider()

    # Dynamic KPI chart selector (so charts adapt when attributes change)
    st.markdown("### KPI Trend Lines (choose the metric + axis)")
    x_axis = st.selectbox(
        "X axis (must be numeric or ordered numeric proxy)",
        options=[c for c in numeric_candidates if c != "monthly_spend_aed"] + ["purchase_freq_score"],
        index=0
    )
    y_metric = st.selectbox(
        "Y metric (numeric)",
        options=[c for c in numeric_candidates if c != x_axis] + ["monthly_spend_aed"],
        index=0
    )

    tmp = df[[x_axis, y_metric]].copy()
    tmp = tmp.dropna()
    if tmp.empty:
        st.info("Not enough data for this selection.")
    else:
        tmp = tmp.groupby(x_axis)[y_metric].mean().reset_index().sort_values(x_axis)
        fig = px.line(tmp, x=x_axis, y=y_metric, markers=True, title=f"Average {y_metric} vs {x_axis}")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Insight: This line shows how the chosen outcome changes across the chosen axis. "
            "GTM implication: use this to identify high-value bands and target them with pricing + channel tactics."
        )

    # Optional channel and pack charts
    colA, colB = st.columns(2)
    if col_channel:
        vc = df[col_channel].value_counts(normalize=True).reset_index()
        vc.columns = ["channel", "share"]
        figc = px.bar(vc, x="channel", y="share", title="Preferred Purchase Channel Share")
        figc.update_layout(height=400, yaxis_tickformat=".0%")
        colA.plotly_chart(figc, use_container_width=True)
        colA.caption(
            "Insight: Channel share indicates where demand already exists. "
            "GTM implication: win the top channel first to maximize early traction."
        )
    else:
        colA.info("Channel column not detected in dataset.")

    if col_pack:
        vp = df[col_pack].value_counts(normalize=True).reset_index()
        vp.columns = ["pack_size", "share"]
        figp = px.bar(vp, x="pack_size", y="share", title="Preferred Pack Size Share")
        figp.update_layout(height=400, yaxis_tickformat=".0%")
        colB.plotly_chart(figp, use_container_width=True)
        colB.caption(
            "Insight: Pack size preference signals usage context (single-serve vs stock-up). "
            "GTM implication: align SKU mix to channel economics."
        )
    else:
        colB.info("Pack size column not detected in dataset.")


# ======================
# PAGE: Consumer Insights
# ======================
elif page == "Consumer Insights":
    st.subheader("Consumer Insights")

    if col_freq:
        fig = px.histogram(df, x=col_freq, title="Purchase Frequency Distribution")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Insight: Frequency shows how habitual bottled water is for consumers. "
            "GTM implication: high habituality supports subscriptions and bulk packs."
        )

    if col_awareness:
        all_aw = []
        for x in df[col_awareness].tolist():
            all_aw.extend(split_multi(x))
        if all_aw:
            aw = pd.Series(all_aw).value_counts().head(12).reset_index()
            aw.columns = ["brand", "mentions"]
            figA = px.bar(aw, x="brand", y="mentions", title="Top Brand Awareness (mentions)")
            figA.update_layout(height=420)
            st.plotly_chart(figA, use_container_width=True)
            st.caption(
                "Insight: Awareness shows which brands are already in the consumer consideration set. "
                "GTM implication: IOTA must differentiate strongly where incumbents dominate mindshare."
            )

    if col_brand_buy:
        buy = df[col_brand_buy].astype(str).value_counts().head(12).reset_index()
        buy.columns = ["brand", "respondents"]
        figB = px.bar(buy, x="brand", y="respondents", title="Most Frequently Purchased Brand")
        figB.update_layout(height=420)
        st.plotly_chart(figB, use_container_width=True)
        st.caption(
            "Insight: Purchase preference shows who is actually winning wallet share. "
            "GTM implication: benchmark IOTA against top-purchased competitors for pricing + claims."
        )


# ======================
# PAGE: Correlation Heatmap (Dynamic)
# ======================
elif page == "Correlation Heatmap":
    st.subheader("Correlation Heatmap (Dynamic)")

    selected = st.multiselect(
        "Choose numeric attributes for correlation heatmap",
        options=numeric_candidates,
        default=numeric_candidates[:12] if len(numeric_candidates) >= 12 else numeric_candidates
    )

    if len(selected) < 3:
        st.warning("Select at least 3 numeric columns.")
        st.stop()

    corr = df[selected].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap")
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Insight: Correlation shows which attributes move together in consumer perception. "
        "GTM implication: build positioning around coherent bundles of drivers, not scattered messages."
    )


# ======================
# PAGE: Regression (Dynamic)
# ======================
elif page == "Regression":
    st.subheader("Regression (Dynamic)")

    y_col = st.selectbox(
        "Choose outcome variable (dependent)",
        options=numeric_candidates,
        index=numeric_candidates.index("monthly_spend_aed") if "monthly_spend_aed" in numeric_candidates else 0
    )

    X_cols = st.multiselect(
        "Choose predictor variables (independent) — mix numeric + categorical allowed",
        options=all_columns,
        default=[c for c in ["purchase_freq_score", "eatout_freq_score", "buys_water_when_eating_out"] if c in all_columns]
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

    model = sm.OLS(y, X).fit()

    results = pd.DataFrame({
        "feature": model.params.index,
        "coef": model.params.values,
        "p_value": model.pvalues.values
    }).sort_values("p_value")

    c1, c2 = st.columns([1.25, 0.75])
    c1.dataframe(results.head(60), use_container_width=True, height=520)
    c2.metric("R-squared", f"{model.rsquared:.3f}")
    c2.metric("Observations", f"{len(model_df):,}")

    sig = results[(results["feature"] != "const") & (results["p_value"] < 0.05)].head(8)
    c2.markdown("**Top significant drivers (p < 0.05):**")
    if sig.empty:
        c2.write("No significant drivers in this configuration.")
    else:
        for _, r in sig.iterrows():
            direction = "↑" if r["coef"] > 0 else "↓"
            c2.write(f"- {direction} `{r['feature']}` (coef={r['coef']:.2f})")

    st.caption(
        "Insight: Regression quantifies which selected attributes are associated with the selected outcome. "
        "GTM implication: use statistically meaningful drivers to justify pricing, messaging, and channel investments."
    )

    # Linear: Actual vs Predicted
    st.markdown("### Linear: Actual vs Predicted (with trend)")
    pred = model.predict(X)
    fit_df = pd.DataFrame({"actual": y.values, "predicted": pred.values})

    figfit = px.scatter(fit_df, x="predicted", y="actual", trendline="ols", title="Actual vs Predicted")
    figfit.update_layout(height=450)
    st.plotly_chart(figfit, use_container_width=True)

    st.caption(
        "Insight: The closer points align to the trend, the more the model explains the outcome. "
        "GTM implication: stronger fit increases confidence in the driver story for GTM decisions."
    )


# ======================
# PAGE: Segmentation (Dynamic KMeans)
# ======================
elif page == "Segmentation (STP)":
    st.subheader("STP Segmentation (KMeans)")

    seg_cols = st.multiselect(
        "Choose attributes for clustering (numeric + categorical allowed)",
        options=all_columns,
        default=[c for c in ["monthly_spend_aed", "purchase_freq_score", "eatout_freq_score", "buys_water_when_eating_out"] if c in all_columns]
    )
    if len(seg_cols) < 4:
        st.warning("Pick at least 4 attributes for stable clustering.")
        st.stop()

    k = st.slider("Number of segments (K)", 3, 8, 4)

    Xseg = encode_for_modeling(df, seg_cols)
    Xs = StandardScaler().fit_transform(Xseg)

    km = KMeans(n_clusters=k, random_state=42, n_init=25)
    df_seg = df.copy()
    df_seg["segment"] = km.fit_predict(Xs)

    st.session_state["df_seg"] = df_seg  # reuse in perceptual mapping

    sizes = df_seg["segment"].value_counts().sort_index().reset_index()
    sizes.columns = ["segment", "respondents"]

    c1, c2 = st.columns([0.7, 1.3])
    c1.dataframe(sizes, use_container_width=True, height=240)
    fig = px.bar(sizes, x="segment", y="respondents", title="Segment Size")
    fig.update_layout(height=320)
    c2.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Insight: Segment sizes show where scale lives vs niche opportunities. "
        "GTM implication: pick 1–2 segments to target first to keep positioning sharp."
    )

    # Segment profile lines (dynamic on chosen numeric columns)
    num_in_seg = [c for c in seg_cols if pd.api.types.is_numeric_dtype(df_seg[c])]
    if len(num_in_seg) >= 3:
        st.markdown("### Line: Segment Profiles Across Selected Numeric Attributes")
        prof = df_seg.groupby("segment")[num_in_seg].mean().reset_index()
        prof_long = prof.melt(id_vars="segment", var_name="attribute", value_name="avg_value")

        figline = px.line(prof_long, x="attribute", y="avg_value", color="segment", markers=True,
                          title="Segment Profiles (Selected Numeric Attributes)")
        figline.update_layout(height=520, xaxis_title="Attribute", yaxis_title="Average")
        st.plotly_chart(figline, use_container_width=True)

        st.caption(
            "Insight: Each segment has a distinct profile across your selected attributes. "
            "GTM implication: use the highest-scoring drivers per segment to tailor messaging and offers."
        )

    with st.expander("Download segmented dataset"):
        csv = df_seg.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV with segment labels", csv, file_name="iota_segmented_output.csv", mime="text/csv")


# ======================
# PAGE: Perceptual Mapping (Dynamic PCA + overlay)
# ======================
elif page == "Positioning & Perceptual Mapping":
    st.subheader("Perceptual Mapping (Dynamic PCA)")

    st.write(
        "Choose numeric attributes for PCA. PCA compresses many attributes into 2 perceptual axes. "
        "Optional: overlay KMeans clusters and show centroids."
    )

    pca_cols = st.multiselect(
        "Choose numeric attributes for PCA (at least 4 recommended)",
        options=numeric_candidates,
        default=numeric_candidates[:8] if len(numeric_candidates) >= 8 else numeric_candidates
    )

    if len(pca_cols) < 3:
        st.warning("Select at least 3 numeric attributes for PCA.")
        st.stop()

    Xp = df[pca_cols].copy().replace([np.inf, -np.inf], np.nan).dropna()
    if Xp.empty:
        st.error("No usable rows for PCA after dropping missing values. Reduce columns.")
        st.stop()

    Xps = StandardScaler().fit_transform(Xp)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xps)

    df_map = df.loc[Xp.index].copy()
    df_map["pc1"] = coords[:, 0]
    df_map["pc2"] = coords[:, 1]

    explained = pca.explained_variance_ratio_
    st.caption(f"PCA variance explained: PC1={explained[0]*100:.1f}% | PC2={explained[1]*100:.1f}%")

    overlay = st.checkbox("Overlay KMeans clusters on the PCA map", value=True)
    if overlay:
        k = st.slider("K (overlay)", 3, 8, 4)
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
        "Insight: This map shows how consumers spread across the preference space defined by your selected attributes. "
        "GTM implication: target a cluster and position IOTA to dominate the attributes driving that cluster."
    )

    # Biplot arrows (loadings)
    st.markdown("### Biplot: Attribute Loadings (what defines the axes?)")
    loadings = pca.components_.T
    loading_df = pd.DataFrame(loadings, index=pca_cols, columns=["pc1_loading", "pc2_loading"]).reset_index()
    loading_df.rename(columns={"index": "attribute"}, inplace=True)

    arrow_scale = st.slider("Arrow scale (visibility)", 2, 14, 7)

    fig2 = go.Figure()
    for _, r in loading_df.iterrows():
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
        "Insight: Arrows show which attributes define each perceptual axis and in what direction. "
        "GTM implication: align IOTA’s messaging with the arrows that point toward your target zone."
    )

    # Centroids if overlay
    if overlay and "segment" in df_map.columns:
        st.markdown("### Segment centroid map (executive summary)")
        centroids = df_map.groupby("segment")[["pc1", "pc2"]].mean().reset_index()
        centroids["size"] = df_map["segment"].value_counts().sort_index().values

        fig3 = px.scatter(centroids, x="pc1", y="pc2", size="size", color="segment", text="segment",
                          title="Segment centroids (size-weighted)")
        fig3.update_traces(textposition="top center")
        fig3.update_layout(height=600)
        st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            "Insight: Centroids summarize each segment’s center of gravity in preference space. "
            "GTM implication: prioritize the most attractive centroid and tailor your go-to-market accordingly."
        )

