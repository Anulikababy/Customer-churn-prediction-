import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️", layout="wide")
st.title("🛍️ Mall Customer Segmentation")
st.write("Use K-Means clustering to group customers by age, annual income and spending score.")

st.markdown("""
Upload the **Mall_Customers.csv** file. It should contain columns for age, annual income
and spending score. The app also accepts common column-name variations.
""")

uploaded = st.file_uploader("Upload Mall_Customers.csv", type=["csv"])
if uploaded is None:
    st.warning("Please upload the Mall Customer Segmentation CSV to continue.")
    st.markdown("[Download the dataset from Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)")
    st.stop()

df = pd.read_csv(uploaded)
df.columns = [c.strip() for c in df.columns]

def find_column(candidates):
    lookup = {c.lower().replace("_", "").replace(" ", "").replace("(", "").replace(")", ""): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace("_", "").replace(" ", "").replace("(", "").replace(")", "")
        if key in lookup:
            return lookup[key]
    return None

age_col = find_column(["Age"])
income_col = find_column(["Annual Income (k$)", "Annual Income", "Income"])
spend_col = find_column(["Spending Score (1-100)", "Spending Score", "Spending"])
if not all([age_col, income_col, spend_col]):
    st.error("Required columns were not found. The CSV must include Age, Annual Income (k$), and Spending Score (1-100).")
    st.write("Columns found:", list(df.columns))
    st.stop()

features = df[[age_col, income_col, spend_col]].apply(pd.to_numeric, errors="coerce").dropna()
if len(features) < 3:
    st.error("At least three complete customer records are required.")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(df.head(10), use_container_width=True)
st.write(f"Usable customer records: **{len(features)}**")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

st.subheader("Choose number of customer groups")
max_k = min(10, len(features) - 1)
k = st.slider("Number of clusters (K)", min_value=2, max_value=max_k, value=min(5, max_k))

# Elbow method
inertias = []
k_values = list(range(1, min(10, len(features) - 1) + 1))
for n in k_values:
    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(k_values, inertias, marker="o")
ax.set_xlabel("Number of clusters (K)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method")
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

model = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)
score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else float("nan")

# Give each cluster an understandable label using its average income and spending score.
clustered = features.copy()
clustered["Cluster"] = labels
means = clustered.groupby("Cluster")[[income_col, spend_col]].mean()
income_mid = clustered[income_col].median()
spend_mid = clustered[spend_col].median()

def segment_name(row):
    high_income = row[income_col] >= income_mid
    high_spend = row[spend_col] >= spend_mid
    if high_income and high_spend:
        return "High income, high spenders"
    if high_income and not high_spend:
        return "High income, low spenders"
    if not high_income and high_spend:
        return "Lower income, high spenders"
    return "Lower income, low spenders"

names = {cluster: segment_name(row) for cluster, row in means.iterrows()}
clustered["Segment"] = clustered["Cluster"].map(names)

st.subheader("Segmentation results")
m1, m2 = st.columns(2)
m1.metric("Silhouette score", f"{score:.3f}")
m2.metric("Clusters selected", str(k))
st.caption("A silhouette score closer to 1 generally indicates better-separated clusters. Compare scores across different K values.")

# PCA projection to two dimensions
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Segment": clustered["Segment"].values})
fig, ax = plt.subplots(figsize=(8, 5))
for name, group in plot_df.groupby("Segment"):
    ax.scatter(group["PC1"], group["PC2"], label=name, alpha=0.75)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("Customer Groups Visualized with PCA")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
st.pyplot(fig)
plt.close(fig)

st.subheader("What each segment means")
summary = clustered.groupby(["Cluster", "Segment"]).agg(
    Customers=(age_col, "count"),
    Average_Age=(age_col, "mean"),
    Average_Income=(income_col, "mean"),
    Average_Spending_Score=(spend_col, "mean")
).reset_index()
st.dataframe(summary.round(2), use_container_width=True)

st.subheader("Identify a new customer's segment")
with st.form("customer_form"):
    a, b, c = st.columns(3)
    age = a.number_input("Age", min_value=1, max_value=100, value=30)
    income = b.number_input("Annual income (k$)", min_value=0.0, value=50.0)
    spending = c.number_input("Spending score (1–100)", min_value=1.0, max_value=100.0, value=50.0)
    submitted = st.form_submit_button("Find customer segment", type="primary")

if submitted:
    new_scaled = scaler.transform(pd.DataFrame([[age, income, spending]], columns=features.columns))
    cluster = int(model.predict(new_scaled)[0])
    st.success(f"This customer belongs to: **{names[cluster]}**")
    st.write(f"Cluster number: {cluster}")

csv = clustered.to_csv(index=False).encode("utf-8")
st.download_button("Download customers with segment labels", csv, "customer_segments.csv", "text/csv")
