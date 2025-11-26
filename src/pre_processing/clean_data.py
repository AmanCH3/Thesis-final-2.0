import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/male_electronics_behavior.csv"
PROCESSED_PATH = "data/processed/model_ready.csv"
os.makedirs("data/processed", exist_ok=True)


def clean_dataset():
    print("📥 Loading raw dataset...")
    df = pd.read_csv(RAW_PATH)

    # Remove negative or unrealistic values
    df = df[df['price_level'] > 1000]
    df = df[df['discount_pct'].between(0, 60)]
    df = df[df['age'].between(18, 60)]
    df = df[df['choice_set_size'].between(4, 30)]
    df = df[df['time_spent_section_sec'].between(60, 3600)]

    # Remove duplicates if any
    df = df.drop_duplicates()

    # Check missing values & fill if needed
    df = df.fillna({
        "basket_value": 0,
        "primary_item_bought": "None"
    })

    # Save cleaned dataset
    df.to_csv(PROCESSED_PATH, index=False)

    print("\n✅ Cleaning completed!")
    print(f"Cleaned dataset saved at: {PROCESSED_PATH}")
    print(f"Rows after cleaning: {df.shape[0]}")


if __name__ == "__main__":
    clean_dataset()
