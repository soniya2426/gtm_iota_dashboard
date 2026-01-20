
---

## 5) app.py (FULL end-to-end, with insights under every analytic)

Copy-paste this entire code into `app.py`.

```python
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


# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="IOTA Water UAE | GTM Dashboard", layout="wide")
st.title("IOTA Water UAE | GTM Analytics Dashboard")
st.caption("Executive-ready GTM insights: KPIs, correlation, regression, STP segmentation, and perceptual mapping.")

DATA_PATH = "data/gip final data.csv"


# ======================================================
# LOAD DATA (local repo path first)
# ======================================================
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
        "1) Create folder `data/` in your repo\n"
        "2) Upload CSV named exactly: `gip final data.csv`\n"
        "3) Redeploy Streamlit Cloud"
    )
    st.stop()


# ======================================================
# HELPERS
# ======================================================
def standardize_col(col: str) -> str:
    col = str(col).replace("\ufeff", "")  # remove BOM
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
    """Return first matching column from standardized candidates."""
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

def split_brands(x) -> list:
    if pd.isna(x):
        return []
    return [b.strip() for b in str(x).split(",") if b.strip()]

def encode_for_modeling(df_in: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Numeric columns -> numeric (median impute)
    Categorical columns -> mode impute + one-hot encoding
    """
    X = df_in[cols].copy()

    # Try numeric coercion if column is mostly numeric in text
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


# ======================================================
# CLEAN + STANDARDIZE
# ======================================================
df = df_raw.copy()
df.columns = [standardize_col(c) for c in df.columns]

# Remove typical junk ID columns
for junk in ["column1", "unnamed_0", "unnamed_1"]:
    if junk in df.columns:
        df = df.drop(columns=[junk], errors="ignore")

cols = df.columns.tolist()

# Detect key columns (robust)
col_spend = find_col(cols, [
    "what_is_your_average_monthly_spent_on_water_in_a_month",
    "average_monthly_spent_on_water",
    "monthly_spent_on_water"
])
col_freq = find_col(cols, [
    "how_often_do_you_purchase_packaged_drinking_water",
    "purchase_frequency",
    "how_often_purchase_packaged_drinking_water"
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
    "purchase_channel",
    "where_do_you_buy_bottled_water"
])
col_pack = find_col(cols, [
    "size_of_bottled_water",
    "what_size_of_bottled_water_do_you_buy_most_often",
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

# Attribute ratings (core perceptual drivers)
attribute_map = {
    "value_for_money": ["value_for_money", "value_for_money_in_purchasing_bottled_water", "value_for_money_rating"],
    "packaging_type": ["packaging_type", "packaging_type_in_purchasing_bottled_water"],
    "added_benefits": ["added_benefits", "added_benefits_like_alkaline_zero_sodium_added_minerals"],
    "source_of_water": ["source_of_water", "source_of_water_in_purchasing_bottled_water"],
    "availability": ["availability", "availability_in_purchasing_bottled_water"],
    "taste": ["taste", "taste_in_purchasing_bottled_water"],
    "brand_name": ["brand_name", "brand_name_in_purchasing_bottled_water"],
    "attractive_promotions": ["attractive_promotions", "attractive_promotions_in_purchasing_bottled_water"],
}

attribute_cols = []
canonical_attr = {}
for canon, cand_list in attribute_map.items():
    found = find_col(cols, cand_list)
    if found:
        canonical_attr[canon] = found
        attribute_cols.append(found)

# Feature engineering (numeric proxies)
df["monthly_spend_aed"] = df[col_spend].apply(spend_to_aed) if col_spend else np.nan
df["purchase_freq_score"] = df[col_freq].apply(freq_to_score) if col_freq else np.nan
df["eatout_freq_score"] = df[col_eatout].apply(freq_to_score) if col_eatout else np.nan
df["buys_water_when_eating_out"] = df[col_buy_eatout].apply(yesno_to_bin) if col_buy_eatout else np.nan

# Global impute
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]
for c in num_cols:
    df[c] = median_impute(df[c])
for c in cat_cols:
    df[c] = mode_impute(df[c])


# ======================================================
# SIDEBAR NAV
# ======================================================
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


# ======================================================
# PAGE: DATA OVERVIEW
# ======================================================
if page == "Data Overview":
    st.subheader("Data Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Attribute drivers detected", f"{len(attribute_cols):,}")
    c4.metric("Has brand column?", "Yes" if col_brand_buy else "No")

    st.dataframe(df.head(25), use_container_width=True)

    st.caption(
        "Insight: This confirms the app can read the dataset correctly and the key GTM variables are available. "
        "GTM implication: if this page loads cleanly, the rest of the dashboard will be stable in Streamlit Cloud."
    )


# ======================================================
# PAGE: KPI METRICS (EXECUTIVE SNAPSHOT)
# ======================================================
elif page == "KPI Metrics":
    st.subheader("KPI Metrics (Executive Snapshot)")

    # KPI tiles
    avg_spend = float(df["monthly_spend_aed"].mean())
    med_spend = float(df["monthly_spend_aed"].median())
    heavy_spend_cut = float(df["monthly_spend_aed"].quantile(0.75))
    heavy_spend_pct = float((df["monthly_spend_aed"] >= heavy_spend_cut).mean() * 100)

    avg_freq = float(df["purchase_freq_score"].mean())
    weekly_plus_pct = float((df["purchase_freq_score"] >= 4).mean() * 100)

    eatout_buy_pct = float(df["buys_water_when_eating_out"].mean() * 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Monthly Spend (AED, proxy)", f"{avg_spend:,.0f}")
    c2.metric("Median Monthly Spend (AED, proxy)", f"{med_spend:,.0f}")
    c3.metric("Heavy Buyers (Top 25%)", f"{heavy_spend_pct:.1f}%")
    c4.metric("Weekly+ Buyers", f"{weekly_plus_pct:.1f}%")

    st.caption(
        "Insight: Spend and frequency quickly separate ‘premium potential’ from ‘volume play’. "
        "GTM implication: if heavy buyers are sizable, premium positioning + higher-margin packs becomes viable."
    )

    st.divider()

    # Channel share
    if col_channel:
        vc = df[col_channel].value_counts(normalize=True).reset_index()
        vc.columns = ["channel", "share"]
        fig = px.bar(vc, x="channel", y="share", title="Preferred Purchase Channel Share")
        fig.update_layout(height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Insight: The dominant channel tells you where consumer demand already lives. "
            "GTM implication: prioritize distribution and promotions in the top channel before expanding."
        )
    else:
        st.info("Channel column not found in this dataset export.")

    # Pack size share
    if col_pack:
        vc2 = df[col_pack].value_counts(normalize=True).reset_index()
        vc2.columns = ["pack_size", "share"]
        fig2 = px.bar(vc2, x="pack_size", y="share", title="Preferred Pack Size Share")
        fig2.update_layout(height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Insight: Pack size preference signals whether the market is ‘grab-and-go’ or ‘stock-up’. "
            "GTM implication: match SKUs to channel economics (bulk online vs single-serve convenience)."
        )

    # Top purchase drivers
    if attribute_cols:
        means = df[attribute_cols].mean().sort_values(ascending=False).reset_index()
        means.columns = ["driver", "avg_rating"]
        fig3 = px.bar(means, x="avg_rating", y="driver", orientation="h", title="Top Purchase Drivers (Average Rating)")
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            "Insight: These are the category decision criteria consumers actually use. "
            "GTM implication: your packaging + ads should scream the top 2–3 drivers, not everything at once."
        )

    # Brand awareness vs purchase
    cA, cB = st.columns(2)
    if col_awareness:
        all_aw = []
        for x in df[col_awareness].tolist():
            all_aw.extend(split_brands(x))
        aw = pd.Series(all_aw).value_counts().head(12).reset_index()
        aw.columns = ["brand", "mentions"]
        figA = px.bar(aw, x="brand", y="mentions", title="Top Brand Awareness (Mentions)")
        figA.update_layout(height=420)
        cA.plotly_chart(figA, use_container_width=True)
        cA.caption(
            "Insight: Awareness is mindshare, not market share. "
            "GTM implication: if incumbents dominate awareness, IOTA needs a sharp wedge message to break through."
        )
    else:
        cA.info("Brand awareness column not found.")

    if col_brand_buy:
        buy = df[col_brand_buy].astype(str).value_counts().head(12).reset_index()
        buy.columns = ["brand", "respondents"]
        figB = px.bar(buy, x="brand", y="respondents", title="Most Frequently Purchased Brand")
        figB.update_layout(height=420)
        cB.plotly_chart(figB, use_container_width=True)
        cB.caption(
            "Insight: ‘Most purchased’ is the competitive reality check. "
            "GTM implication: your first distribution and messaging must beat these brands where it matters (shelf, online, restaurants)."
        )
    else:
        cB.info("Most-purchased brand column not found.")


# ======================================================
# PAGE: CONSUMER INSIGHTS (BEHAVIOR)
# ======================================================
elif page == "Consumer Insights":
    st.subheader("Consumer Insights (Behavior + Brand)")

    # Frequency distribution
    if col_freq:
        fig = px.histogram(df, x=col_freq, title="Purchase Frequency Distribution")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Insight: Frequency tells you whether water is treated as a staple (high repeat) or occasional add-on. "
            "GTM implication: staples reward subscriptions, bulk packs, and always-available distribution."
        )

    # Eat-out behavior
    if col_buy_eatout:
        fig2 = px.histogram(df, x=col_buy_eatout, title="Do Consumers Buy Water While Eating Out?")
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Insight: Out-of-home purchase is a separate battleground with different economics and brand cues. "
            "GTM implication: if ‘Yes’ is high, restaurant/café placements can accelerate trial and premium perception."
        )


# ======================================================
# PAGE: CORRELATION HEATMAP
# ======================================================
elif page == "Correlation Heatmap":
    st.subheader("Correlation Heatmap (Drivers + Spend/Frequency)")

    numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    default_heat = [c for c in (attribute_cols + ["monthly_spend_aed", "purchase_freq_score", "eatout_freq_score"]) if c in numeric_candidates]
    default_heat = default_heat if len(default_heat) >= 3 else numeric_candidates[:10]

    selected = st.multiselect(
        "Select numeric columns for correlation",
        options=numeric_candidates,
        default=default_heat
    )

    if len(selected) < 3:
        st.warning("Select at least 3 numeric columns.")
        st.stop()

    corr = df[selected].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap")
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Insight: Correlation shows which preferences move together in consumers’ minds. "
        "GTM implication: build positioning around coherent bundles (e.g., taste + source), not random feature lists."
    )


# ======================================================
# PAGE: REGRESSION
# ======================================================
elif page == "Regression":
    st.subheader("Regression (Drivers of Spend / WTP Proxy)")

    numeric_outcomes = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if "monthly_spend_aed" not in numeric_outcomes:
        st.error("monthly_spend_aed not available. Check spend mapping in dataset.")
        st.stop()

    y_col = st.selectbox("Outcome (Dependent variable)", numeric_outcomes, index=numeric_outcomes.index("monthly_spend_aed"))

    default_X = attribute_cols + ["purchase_freq_score", "eatout_freq_score", "buys_water_when_eating_out"]
    default_X = [c for c in default_X if c in df.columns]

    X_cols = st.multiselect("Drivers (Independent variables)", df.columns.tolist(), default=default_X)

    if len(X_cols) < 3:
        st.warning("Select at least 3 driver columns.")
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
    c1.dataframe(results.head(50), use_container_width=True, height=520)
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
        "Insight: Regression highlights which drivers are most associated with higher spend (a WTP proxy). "
        "GTM implication: prioritize significant drivers in pricing, packaging claims, and channel strategy."
    )


# ======================================================
# PAGE: STP SEGMENTATION (KMeans, selectable attributes)
# ======================================================
elif page == "Segmentation (STP)":
    st.subheader("STP Segmentation (KMeans Clustering)")

    st.markdown(
        "Choose any attributes you want (demographics, behavior, perceptions). "
        "Categorical variables are automatically one-hot encoded."
    )

    default_seg = attribute_cols + ["monthly_spend_aed", "purchase_freq_score", "eatout_freq_score", "buys_water_when_eating_out"]
    default_seg = [c for c in default_seg if c in df.columns]

    seg_cols = st.multiselect("Segmentation attributes", df.columns.tolist(), default=default_seg)
    if len(seg_cols) < 4:
        st.warning("Pick at least 4 attributes for stable segmentation.")
        st.stop()

    k = st.slider("Number of segments (K)", 3, 8, 4)

    X = encode_for_modeling(df, seg_cols)
    Xs = StandardScaler().fit_transform(X)

