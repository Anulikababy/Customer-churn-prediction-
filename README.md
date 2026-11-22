# Project 3: Customer Segmentation (Unsupervised Learning)

## Goal
Group mall shoppers into customer segments so a business can target them differently.

## Dataset
Mall Customer Segmentation dataset (200 rows, 5 columns), available from Kaggle:
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Download the CSV and place it in this project folder with the filename `Mall_Customers.csv` to run `analysis.py`. The Streamlit app lets you upload the CSV directly.

Expected columns:
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

The model uses age, annual income and spending score as requested. Customer ID and Gender are not used as clustering features.

## Project activities
- Inspect the data and summarize the three numeric features.
- Standardize the features with StandardScaler.
- Use the elbow method to inspect candidate cluster counts.
- Fit K-Means (K=5 in the analysis script; the app lets the user choose K).
- Evaluate clustering with the silhouette score.
- Use PCA to visualize the groups in two dimensions.
- Give clusters plain-language names based on their average income and spending score.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Download Mall_Customers.csv from the Kaggle link above.
python analysis.py --csv Mall_Customers.csv
streamlit run app.py
```

## Streamlit app
Upload the CSV, choose the number of clusters, view the elbow plot, silhouette score, PCA visualization and cluster summaries. You can also enter a new customer's age, annual income and spending score to predict their segment, then download the labelled records.

## Deploy on Streamlit Community Cloud
1. Create a public GitHub repository and upload all files in this folder.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
4. Deploy. Copy the generated app URL for submission.

## Interpretation note
The four plain-language segment names are based on whether a cluster's mean income and mean spending score are above or below the overall medians. If K-Means creates more than four clusters, some clusters may share a plain-language name; their cluster numbers and statistics remain available in the results table.
