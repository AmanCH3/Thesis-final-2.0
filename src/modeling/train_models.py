import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


PROCESSED_PATH = "data/processed/model_ready.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    return pd.read_csv(PROCESSED_PATH)


def prepare_features(df):
    target = "purchase_made"
    drop_cols = ["basket_value", "primary_item_bought", "purchase_prob_simulated"]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    X = df.drop(columns=[target])
    y = df[target]

    categorical = X.select_dtypes(include="object").columns.tolist()
    numeric = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    return X, y, preprocessor


def get_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=3000, C=0.6, penalty="l2", class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=40,
            min_child_samples=50
        ),
        "CatBoost": CatBoostClassifier(
            depth=6,
            learning_rate=0.03,
            iterations=500,
            l2_leaf_reg=5,
            verbose=False
        ),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }


def train_all_models():
    print("📥 Loading cleaned dataset...")
    df = load_data()

    X, y, preprocessor = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = get_models()
    results = {}
    best_model_name = None
    best_score = -1
    best_pipeline = None

    print("\n🚀 Training models...\n")

    for name, model in models.items():
        print(f"🔧 Training {name}...")

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        # Calibration
        calibrated = CalibratedClassifierCV(pipeline, cv=3)
        calibrated.fit(X_train, y_train)

        metrics = evaluate(calibrated, X_test, y_test)
        results[name] = metrics

        print(f"   ✔ Accuracy: {metrics['accuracy']:.4f}")
        print(f"   ✔ ROC-AUC:  {metrics['roc_auc']:.4f}")
        print(f"   ✔ F1:       {metrics['f1_score']:.4f}\n")

        if metrics["roc_auc"] > best_score:
            best_model_name = name
            best_score = metrics["roc_auc"]
            best_pipeline = calibrated

    print("\n🏆 BEST MODEL:", best_model_name)
    print("⭐ ROC-AUC:", best_score)

    best_model_path = f"{MODEL_DIR}/best_model_{best_model_name}.pkl"
    joblib.dump(best_pipeline, best_model_path)

    results_df = pd.DataFrame(results).T
    results_df.to_csv(f"{MODEL_DIR}/model_comparison_results.csv")

    print("\n📊 Model comparison saved!")
    print(f"💾 Best model saved at: {best_model_path}")


if __name__ == "__main__":
    train_all_models()
