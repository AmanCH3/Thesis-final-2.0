import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------
# CONFIG
# --------------------------
MODEL_PATH = "models/best_model_CatBoost.pkl"
DATA_PATH = "data/processed/model_ready.csv"
OUTPUT_DIR = "reports/shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------
# LOAD MODEL & DATA
# --------------------------
print("📌 Loading model...")
model = joblib.load(MODEL_PATH)

print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Remove leakage
drop_cols = ["basket_value", "primary_item_bought", "purchase_prob_simulated"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

target = "purchase_made"
X = df.drop(columns=[target])
y = df[target]


# --------------------------
# GET PIPELINE PARTS (IMPORTANT FIX)
# --------------------------
print("🔧 Extracting pipeline steps from calibrated model...")

pipeline = model.estimator          # Pipeline inside CalibratedClassifierCV
preprocess = pipeline.named_steps["preprocess"]
model_only = pipeline.named_steps["model"]

print("✔ Extracted preprocess + model")


# --------------------------
# GET REAL FEATURE NAMES
# --------------------------
cat_cols = preprocess.transformers_[0][2]
num_cols = preprocess.transformers_[1][2]

cat_encoded = preprocess.named_transformers_["cat"].get_feature_names_out(cat_cols)

feature_names = list(cat_encoded) + list(num_cols)

# Transform dataset
X_processed = preprocess.transform(X)


# --------------------------
# CREATE SHAP EXPLAINER
# --------------------------
print("⚡ Creating SHAP explainer (CatBoost)...")
explainer = shap.TreeExplainer(model_only)
shap_values = explainer.shap_values(X_processed)


# --------------------------
# SUMMARY PLOT
# --------------------------
print("📊 Generating SHAP Summary Plot...")
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_processed, feature_names=feature_names, show=False)
plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=300)
plt.close()


# --------------------------
# BAR PLOT
# --------------------------
print("📊 Generating SHAP Bar Plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_processed, feature_names=feature_names, plot_type="bar", show=False)
plt.savefig(f"{OUTPUT_DIR}/shap_bar.png", dpi=300)
plt.close()


# --------------------------
# FEATURE IMPORTANCE CSV
# --------------------------
print("📈 Saving SHAP feature importance ranking...")
mean_abs = pd.DataFrame({
    "feature": feature_names,
    "importance": np.abs(shap_values).mean(axis=0)
}).sort_values("importance", ascending=False)

mean_abs.to_csv(f"{OUTPUT_DIR}/shap_feature_ranking.csv", index=False)


# --------------------------
# FORCE PLOT FOR ONE SAMPLE
# --------------------------
print("🎯 Generating Local Force Plot...")
sample_idx = 50
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx],
    feature_names=feature_names,
    matplotlib=False
)
shap.save_html(f"{OUTPUT_DIR}/force_plot_sample.html", force_plot)


# --------------------------
# DECISION PLOT
# --------------------------
print("📈 Generating Decision Plot...")
plt.figure(figsize=(12, 6))
shap.decision_plot(
    explainer.expected_value,
    shap_values[:50],
    feature_names=feature_names,
    show=False
)
plt.savefig(f"{OUTPUT_DIR}/decision_plot.png", dpi=300)
plt.close()


# --------------------------
# INTERACTION VALUES
# --------------------------
print("🔍 Computing SHAP interaction values...")
interaction_values = explainer.shap_interaction_values(X_processed)

plt.figure(figsize=(12, 6))
shap.summary_plot(
    interaction_values, 
    X_processed, 
    feature_names=feature_names, 
    show=False
)
plt.savefig(f"{OUTPUT_DIR}/interaction_summary.png", dpi=300)
plt.close()

