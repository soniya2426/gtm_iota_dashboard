# IOTA Water UAE | GTM Analytics Dashboard (Streamlit)

Interactive Go-To-Market analytics dashboard for IOTA Water (UAE bottled water market), built using Streamlit.

## What it does
- Executive KPI metrics (spend, frequency, channel mix, pack sizes, top drivers)
- Consumer insights (behavior + brand awareness/purchase)
- Correlation heatmap (driver relationships)
- Regression (drivers of willingness-to-pay proxy)
- STP segmentation (KMeans clustering using selectable attributes)
- Positioning & perceptual maps (PCA-based + segment overlay + brand centroids)

## Dataset
Place the CSV here:
`data/gip final data.csv`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
