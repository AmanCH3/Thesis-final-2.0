# Central config so all scripts use the same paths

RAW_DATA_PATH = "data/raw/male_electronics_behavior.csv"
PROCESSED_DATA_PATH = "data/processed/model_ready.csv"

FEATURE_COLUMNS = [
    "age", "income_level", "store_region", "time_of_day", "visit_day_type",
    "festival_season", "payment_method", "product_category", "brand_tier",
    "price_level", "discount_pct", "is_new_arrival", "is_featured_display",
    "shelf_zone", "eye_level_display", "choice_set_size", "anchor_price",
    "relative_price_to_anchor", "time_spent_section_sec",
    "num_products_viewed", "novelty_preference", "anchoring_sensitivity",
    "choice_paralysis_sensitivity"
]

TARGET_COLUMN = "purchase_made"
