import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

class RetailAIEngine:
    """
    Unified Smart Product Recommendation Engine for Nepalese Retail:
    
    FEATURES:
    -------------------------
    ✓ Predict purchase probability (ML model)
    ✓ Behavioral psychology scoring:
        - Anchoring Effect
        - Novelty Bias
        - Choice Paralysis
    ✓ Product Placement Recommendation
    ✓ SKU Optimization Recommendation
    ✓ Local SHAP Explanation (Top Drivers)
    ✓ Unified Final Recommendation Output
    """

    def __init__(
        self,
        model_path="models/best_model_CatBoost.pkl",
        shap_importance_path="reports/shap/shap_feature_ranking.csv"
    ):
        # Load model
        print("📌 Loading model...")
        self.model = joblib.load(model_path)

        # Detect where pipeline is stored
        print("🔍 Detecting pipeline inside model...")

        if isinstance(self.model, CalibratedClassifierCV):
            # sklearn stores calibrated classifiers as a list
            self.pipeline = self.model.calibrated_classifiers_[0].estimator
            print("✔ Pipeline extracted from CalibratedClassifierCV.estimator")
        elif isinstance(self.model, Pipeline):
            self.pipeline = self.model
            print("✔ Model is a direct Pipeline")
        elif hasattr(self.model, "estimator"):
            self.pipeline = self.model.estimator
            print("✔ Pipeline found at model.estimator")
        else:
            raise ValueError("❌ Unable to locate pipeline inside saved model.")

        # Extract preprocess and model
        self.preprocess = self.pipeline.named_steps["preprocess"]
        self.model_only = self.pipeline.named_steps["model"]
        print("✔ Extracted preprocess + model")

        # Load SHAP feature ranking
        self.shap_importance = pd.read_csv(shap_importance_path)

        # Create SHAP explainer
        print("⚡ Initializing SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model_only)

    # ----------------------------------------------------------
    # 1. PREDICT PURCHASE PROBABILITY
    # ----------------------------------------------------------
    def predict_purchase(self, features: dict):
        df = pd.DataFrame([features])
        X_processed = self.preprocess.transform(df)
        prob = self.model.predict_proba(df)[0][1]
        return float(prob)

    # ----------------------------------------------------------
    # 2. BEHAVIORAL PSYCHOLOGY ENGINE
    # ----------------------------------------------------------
    def compute_behavioral_scores(self, features: dict):
        """Calculate Anchoring, Novelty, Choice Paralysis"""
        # Anchoring: Higher "relative price" → stronger anchoring
        anchoring_strength = max(0, (features["relative_price_to_anchor"] - 0.5) * 2)

        # Choice paralysis if too many SKUs
        if features["choice_set_size"] > 15:
            paralysis_risk = 1.0
        elif features["choice_set_size"] > 12:
            paralysis_risk = 0.7
        elif features["choice_set_size"] > 8:
            paralysis_risk = 0.4
        else:
            paralysis_risk = 0.1

        # Novelty boost
        novelty_boost = features["novelty_preference"]
        if features["is_new_arrival"] == 1:
            novelty_boost *= 1.5

        return {
            "anchoring_strength": float(anchoring_strength),
            "choice_paralysis_risk": float(paralysis_risk),
            "novelty_boost": float(novelty_boost)
        }

    # ----------------------------------------------------------
    # 3. STORE PLACEMENT RECOMMENDATION ENGINE
    # ----------------------------------------------------------
    def recommend_placement(self, features: dict):
        """
        Suggest best store placement based on:
        - Eye Level Display
        - Hotspot presence
        - Novelty preference
        - SKU overload
        """
        
        # Derive hotspot from shelf_zone
        hotspot_zones = ["CenterHotspot", "Entrance"]
        is_hotspot = 1 if features.get("shelf_zone", "") in hotspot_zones else 0

        score = features["eye_level_display"] * 0.6 + is_hotspot * 0.4

        # Priority recommendation rules
        if score >= 0.7:
            return "Eye-Level Hotspot", float(score)

        if features["novelty_preference"] > 0.7:
            return "Place in New Arrival Section", float(score)

        if features["choice_set_size"] > 12:
            return "Reduce SKU Count & Reorganize Shelf", float(score)

        return "Standard Eye-Level Shelf", float(score)

    # ----------------------------------------------------------
    # 4. SKU COUNT OPTIMIZATION
    # ----------------------------------------------------------
    def recommend_sku_count(self, choice_set_size: int):
        if choice_set_size > 15:
            return "Reduce SKU count to 8–12 (Choice paralysis detected)"
        if choice_set_size < 5:
            return "Increase SKU variety to 8–12 (Not enough options)"
        return "SKU count is optimal (8–12 recommended)"

    # ----------------------------------------------------------
    # 5. SHAP LOCAL EXPLANATION
    # ----------------------------------------------------------
    def explain_prediction(self, features: dict):
        df = pd.DataFrame([features])
        X_processed = self.preprocess.transform(df)
        shap_values = self.explainer.shap_values(X_processed)

        top_features = (
            self.shap_importance
            .head(10)
            .to_dict(orient="records")
        )

        return {
            "expected_value": float(self.explainer.expected_value),
            "shap_values": shap_values[0].tolist(),
            "top_feature_importance": top_features
        }

    # ----------------------------------------------------------
    # 6. FINAL MERGED RECOMMENDATION ENGINE
    # ----------------------------------------------------------
    def generate_recommendation(self, features: dict):
        """Main engine combining prediction + psychology + placement"""

        purchase_probability = self.predict_purchase(features)
        behavioral_scores = self.compute_behavioral_scores(features)
        placement, placement_score = self.recommend_placement(features)
        sku_suggestion = self.recommend_sku_count(features["choice_set_size"])
        explanation = self.explain_prediction(features)

        return {
            "purchase_probability": purchase_probability,
            "behavioral_scores": behavioral_scores,
            "recommended_placement": placement,
            "placement_score": placement_score,
            "sku_suggestion": sku_suggestion,
            "explanation": explanation
        }


    # ----------------------------------------------------------
    # 7. MARKET BASKET ANALYSIS ENGINE
    # ----------------------------------------------------------
    def generate_market_basket_data(self):
        """
        Simulate transaction data for Market Basket Analysis.
        Returns:
            transactions (list of lists): Simulated baskets
            products (list): List of unique products
        """
        products = ["Smartphone", "Laptop", "Headphones", "Smart Watch", 
                   "Phone Case", "Laptop Bag", "Screen Guard", "Charger", 
                   "Gaming Mouse", "Keyboard", "Speaker", "Power Bank"]
        
        # Define strong pairs (Ground Truth)
        strong_pairs = [
            ("Smartphone", "Phone Case"), ("Smartphone", "Screen Guard"),
            ("Laptop", "Laptop Bag"), ("Laptop", "Gaming Mouse"),
            ("Smartphone", "Headphones"), ("Laptop", "Keyboard"),
            ("Smart Watch", "Phone Case"), ("Charger", "Power Bank")
        ]
        
        transactions = []
        np.random.seed(42)
        
        for _ in range(1000):
            basket_size = np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])
            basket = []
            
            # 40% chance to pick a strong pair
            if np.random.random() < 0.4:
                pair = strong_pairs[np.random.randint(0, len(strong_pairs))]
                basket.extend(list(pair))
            
            # Fill remaining slots with random products
            while len(basket) < basket_size:
                item = np.random.choice(products)
                if item not in basket:
                    basket.append(item)
            
            transactions.append(basket)
            
        return transactions, products

    def get_association_rules(self):
        """
        Calculate Co-occurrence Matrix and Lift Scores.
        """
        transactions, products = self.generate_market_basket_data()
        
        # 1. Co-occurrence Matrix
        co_matrix = pd.DataFrame(0, index=products, columns=products)
        item_counts = {p: 0 for p in products}
        
        for basket in transactions:
            for item in basket:
                item_counts[item] += 1
            
            # Count pairs
            for i in range(len(basket)):
                for j in range(i + 1, len(basket)):
                    item1, item2 = basket[i], basket[j]
                    co_matrix.loc[item1, item2] += 1
                    co_matrix.loc[item2, item1] += 1
        
        # 2. Calculate Lift Matrix
        total_transactions = len(transactions)
        lift_matrix = pd.DataFrame(0.0, index=products, columns=products)
        
        rules = []
        
        for p1 in products:
            for p2 in products:
                if p1 == p2:
                    continue
                
                prob_p1 = item_counts[p1] / total_transactions
                prob_p2 = item_counts[p2] / total_transactions
                prob_both = co_matrix.loc[p1, p2] / total_transactions
                
                if prob_p1 > 0 and prob_p2 > 0:
                    lift = prob_both / (prob_p1 * prob_p2)
                    lift_matrix.loc[p1, p2] = lift
                    
                    if lift > 1.2:  # Only interesting rules
                        rules.append({
                            "antecedent": p1,
                            "consequent": p2,
                            "lift": lift,
                            "confidence": prob_both / prob_p1,
                            "support": prob_both
                        })
        
        return {
            "co_occurrence": co_matrix,
            "lift_matrix": lift_matrix,
            "rules": pd.DataFrame(rules).sort_values("lift", ascending=False)
        }


# ----------------------------------------------------------
# EXAMPLE USAGE (for testing)
# ----------------------------------------------------------
if __name__ == "__main__":
    engine = RetailAIEngine()

    example = {
        "discount_pct": 10,
        "choice_set_size": 14,
        "relative_price_anchor": 0.7,
        "novelty_preference": 0.8,
        "is_new_arrival": 1,
        "eye_level_display": 1,
        "hotspot_zone": 1
    }

    output = engine.generate_recommendation(example)
    print("\n🔥 FINAL RECOMMENDATION OUTPUT:")
    print(output)
