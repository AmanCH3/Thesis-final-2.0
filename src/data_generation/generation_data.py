import numpy as np
import pandas as pd
from typing import Tuple


def _sample_categorical(choices, probs, size):
    return np.random.choice(choices, p=probs, size=size)


def generate_male_electronics_dataset(
    n_rows: int = 50000, random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic behavioural dataset for male consumers
    in the electronics section of Nepalese retail stores.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate (25_000 to 300_000 recommended).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        Synthetic dataset.
    """
    rng = np.random.default_rng(random_state)

    # ---------------------------
    # Customer-level attributes
    # ---------------------------
    customer_ids = rng.integers(1, int(n_rows * 0.4), size=n_rows)
    age = rng.integers(18, 60, size=n_rows)

    income_level = _sample_categorical(
        ["Low", "Middle", "High"], [0.35, 0.5, 0.15], n_rows
    )

    store_region = _sample_categorical(
        ["Kathmandu", "Pokhara", "Biratnagar", "Butwal", "Chitwan"],
        [0.45, 0.18, 0.14, 0.12, 0.11],
        n_rows,
    )

    visit_day_type = _sample_categorical(
        ["Weekday", "Weekend"], [0.7, 0.3], n_rows
    )

    time_of_day = _sample_categorical(
        ["Morning", "Afternoon", "Evening"],
        [0.25, 0.45, 0.30],
        n_rows,
    )

    festival_season = _sample_categorical(
        ["None", "Dashain", "Tihar", "NewYear", "OtherFest"],
        [0.65, 0.12, 0.10, 0.05, 0.08],
        n_rows,
    )

    payment_method = _sample_categorical(
        ["Cash", "Card", "eSewa", "Khalti", "Fonepay"],
        [0.35, 0.22, 0.22, 0.12, 0.09],
        n_rows,
    )

    # Behavioural traits (customer-level tendencies)
    novelty_pref = np.clip(rng.normal(0.55, 0.18, n_rows), 0, 1)
    anchoring_sens = np.clip(rng.normal(0.6, 0.2, n_rows), 0, 1)
    choice_paralysis = np.clip(rng.normal(0.5, 0.2, n_rows), 0, 1)

    # ---------------------------
    # Product / shelf attributes
    # ---------------------------

    product_category = _sample_categorical(
        ["Smartphone", "Laptop", "TV", "Headphones", "Gaming", "Appliance"],
        [0.32, 0.22, 0.18, 0.12, 0.08, 0.08],
        n_rows,
    )

    brand_tier = _sample_categorical(
        ["Budget", "MidRange", "Premium"],
        [0.4, 0.4, 0.2],
        n_rows,
    )

    # Base price ranges by category/tier
    base_price = []
    for cat, tier in zip(product_category, brand_tier):
        if cat == "Smartphone":
            if tier == "Budget":
                price = rng.normal(18000, 3000)
            elif tier == "MidRange":
                price = rng.normal(35000, 5000)
            else:
                price = rng.normal(70000, 15000)
        elif cat == "Laptop":
            if tier == "Budget":
                price = rng.normal(45000, 8000)
            elif tier == "MidRange":
                price = rng.normal(80000, 12000)
            else:
                price = rng.normal(140000, 25000)
        elif cat == "TV":
            if tier == "Budget":
                price = rng.normal(25000, 5000)
            elif tier == "MidRange":
                price = rng.normal(55000, 8000)
            else:
                price = rng.normal(105000, 20000)
        elif cat == "Headphones":
            if tier == "Budget":
                price = rng.normal(2500, 600)
            elif tier == "MidRange":
                price = rng.normal(6000, 1200)
            else:
                price = rng.normal(15000, 3000)
        elif cat == "Gaming":
            if tier == "Budget":
                price = rng.normal(40000, 7000)
            elif tier == "MidRange":
                price = rng.normal(90000, 15000)
            else:
                price = rng.normal(170000, 30000)
        else:  # Appliance (microwave, mixer, etc.)
            if tier == "Budget":
                price = rng.normal(7000, 1200)
            elif tier == "MidRange":
                price = rng.normal(16000, 2500)
            else:
                price = rng.normal(35000, 6000)

        base_price.append(max(1500, price))

    price_level = np.array(base_price).round().astype(int)

    discount_pct = np.clip(rng.normal(12, 8, n_rows), 0, 40)

    is_new_arrival = rng.binomial(1, 0.25, n_rows)
    is_featured_display = rng.binomial(1, 0.3, n_rows)

    shelf_zone = _sample_categorical(
        ["Entrance", "CenterHotspot", "BackWall", "SideAisle", "CashCounter"],
        [0.15, 0.3, 0.25, 0.2, 0.10],
        n_rows,
    )

    eye_level_display = rng.binomial(1, 0.55, n_rows)

    choice_set_size = rng.integers(4, 31, size=n_rows)

    # Anchor price: usually higher than most options in that bay
    anchor_price = (price_level * rng.uniform(1.1, 1.6, n_rows)).round().astype(int)
    relative_price_to_anchor = price_level / anchor_price

    # Behaviour during visit
    time_spent_section_sec = np.clip(
        rng.normal(600, 220, n_rows), 120, 2400
    )  # 2–40 mins
    num_products_viewed = np.clip(
        (time_spent_section_sec / rng.normal(90, 20, n_rows)).astype(int),
        1,
        40,
    )

    # ---------------------------
    # Purchase decision model
    # ---------------------------

    # Base purchase probability by category & income
    base_prob = np.full(n_rows, 0.25)

    base_prob += np.where(product_category == "Smartphone", 0.05, 0)
    base_prob += np.where(product_category == "Laptop", 0.03, 0)
    base_prob += np.where(product_category == "Headphones", 0.02, 0)

    base_prob += np.where(income_level == "High", 0.06, 0)
    base_prob += np.where(income_level == "Middle", 0.02, 0)

    # Festival effect (people buy more electronics)
    festival_boost = np.where(
        np.isin(festival_season, ["Dashain", "Tihar", "NewYear"]), 0.08, 0.0
    )
    base_prob += festival_boost

    # Novelty bias: new arrivals with high novelty preference
    novelty_effect = is_new_arrival * (novelty_pref * 0.15)
    base_prob += novelty_effect

    # Anchoring effect: if discounted price looks good vs anchor
    discount_factor = discount_pct / 100.0
    relative_anchor_factor = 1 - relative_price_to_anchor  # positive if cheaper than anchor
    anchoring_effect = (
        anchoring_sens * np.clip(relative_anchor_factor, -0.3, 0.5) * 0.5
        + anchoring_sens * discount_factor * 0.4
    )
    base_prob += anchoring_effect

    # Choice paralysis: too many options hurt conversion
    # Optimal band roughly 8–14 SKUs
    optimal_band = ((choice_set_size >= 8) & (choice_set_size <= 14)).astype(int)
    choice_effect = (
        optimal_band * (1 - choice_paralysis) * 0.05
        - (choice_paralysis * ((choice_set_size - 14) / 25.0))
    )
    base_prob += choice_effect

    # Layout & shelf positioning
    base_prob += np.where(shelf_zone == "CenterHotspot", 0.05, 0)
    base_prob += np.where(shelf_zone == "Entrance", 0.03, 0)
    base_prob += np.where(shelf_zone == "BackWall", -0.03, 0)
    base_prob += eye_level_display * 0.04
    base_prob += is_featured_display * 0.03

    # Time spent & products viewed (up to a point)
    base_prob += np.clip((time_spent_section_sec - 300) / 2000.0, -0.05, 0.07)
    base_prob += np.clip((num_products_viewed - 5) / 40.0, -0.05, 0.05)

    # Clamp probabilities
    purchase_prob = np.clip(base_prob, 0.01, 0.85)

    purchase_made = rng.binomial(1, purchase_prob)

    # Basket value: if purchase_made=1, add some noise over price_level
    basket_value = np.where(
        purchase_made == 1,
        (price_level * rng.uniform(0.9, 1.3, n_rows)).round(),
        0,
    )

    primary_item_bought = np.where(
        purchase_made == 1, product_category, "None"
    )

    # ---------------------------
    # Assemble DataFrame
    # ---------------------------

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(1, n_rows + 1),
            "customer_id": customer_ids,
            "gender": "Male",
            "age": age,
            "income_level": income_level,
            "store_region": store_region,
            "visit_day_type": visit_day_type,
            "time_of_day": time_of_day,
            "festival_season": festival_season,
            "payment_method": payment_method,
            "product_category": product_category,
            "brand_tier": brand_tier,
            "price_level": price_level,
            "discount_pct": discount_pct.round(1),
            "is_new_arrival": is_new_arrival,
            "is_featured_display": is_featured_display,
            "shelf_zone": shelf_zone,
            "eye_level_display": eye_level_display,
            "choice_set_size": choice_set_size,
            "anchor_price": anchor_price,
            "relative_price_to_anchor": relative_price_to_anchor.round(3),
            "time_spent_section_sec": time_spent_section_sec.astype(int),
            "num_products_viewed": num_products_viewed.astype(int),
            "novelty_preference": novelty_pref.round(3),
            "anchoring_sensitivity": anchoring_sens.round(3),
            "choice_paralysis_sensitivity": choice_paralysis.round(3),
            "purchase_prob_simulated": purchase_prob.round(3),
            "purchase_made": purchase_made,
            "basket_value": basket_value.astype(int),
            "primary_item_bought": primary_item_bought,
        }
    )

    return df


if __name__ == "__main__":
    # Example usage: generate 100k rows
    N_ROWS = 100_000
    df = generate_male_electronics_dataset(n_rows=N_ROWS, random_state=42)
    df.to_csv("data/raw/male_electronics_behavior.csv", index=False)
    print(f"Saved {N_ROWS} rows to data/raw/male_electronics_behavior.csv")
