import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RAW_DATA_PATH = "data/raw/male_electronics_behavior.csv"
SAVE_FIG_DIR = "reports/figures"
os.makedirs(SAVE_FIG_DIR, exist_ok=True)


def run_basic_eda():
    print("🔍 Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    print("\n====================")
    print("BASIC INFO")
    print("====================")
    print(df.info())

    print("\n====================")
    print("SUMMARY STATS")
    print("====================")
    print(df.describe().T)

    print("\n====================")
    print("NULL VALUES")
    print("====================")
    print(df.isnull().sum())

    print("\n====================")
    print("TARGET DISTRIBUTION")
    print("====================")
    print(df['purchase_made'].value_counts(normalize=True))

    # Purchase distribution plot
    sns.countplot(data=df, x='purchase_made')
    plt.title("Purchase Distribution")
    plt.savefig(f"{SAVE_FIG_DIR}/purchase_distribution.png")
    plt.clf()

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.savefig(f"{SAVE_FIG_DIR}/correlation_heatmap.png")
    plt.clf()

    print("\n📊 EDA Completed! Plots saved to reports/figures/")


if __name__ == "__main__":
    run_basic_eda()
