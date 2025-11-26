import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
import shap

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engine.product_recommendation_engine import RetailAIEngine


# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="RetailAI – Consumer Behavior & Product Placement",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛒 RetailAI – Consumer Behavior & Product Placement Intelligence")
st.markdown("### Interactive Dashboard for Understanding Consumer Psychology & Optimizing Product Placement")


# ============================
# LOAD ENGINE + DATA
# ============================
@st.cache_resource
def load_engine():
    return RetailAIEngine()

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/model_ready.csv")
    return df

engine = load_engine()
df = load_data()


# ============================
# SIDEBAR - CUSTOMER & PRODUCT CONFIGURATION
# ============================
st.sidebar.header("🎛️ Configure Scenario")

st.sidebar.subheader("👤 Customer Profile")
customer_age = st.sidebar.slider("Age", 18, 70, 30)
customer_income = st.sidebar.selectbox("Income Level", ["Low", "Middle", "High"])
customer_gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
visit_time = st.sidebar.selectbox("Visit Time", ["Morning", "Afternoon", "Evening"])
visit_day = st.sidebar.selectbox("Day Type", ["Weekday", "Weekend"])
festival = st.sidebar.selectbox("Festival Season", ["None", "Dashain", "Tihar", "NewYear", "OtherFest"])

st.sidebar.subheader("📦 Product Configuration")
product_category = st.sidebar.selectbox("Product Category", 
    ["Smartphone", "Laptop", "TV", "Gaming", "Appliance", "Headphones"])
brand_tier = st.sidebar.selectbox("Brand Tier", ["Budget", "MidRange", "Premium"])
product_price = st.sidebar.slider("Product Price (NPR)", 100, 50000, 10000, step=100)
discount_pct = st.sidebar.slider("Discount %", 0, 50, 10)
is_new_arrival = st.sidebar.checkbox("New Arrival", value=False)

st.sidebar.subheader("🏪 Store Configuration")
store_region = st.sidebar.selectbox("Store Location", 
    ["Kathmandu", "Pokhara", "Chitwan", "Biratnagar", "Butwal"])
anchor_price = st.sidebar.slider("Anchor Price (NPR)", 100, 50000, 15000, step=100)
choice_set_size = st.sidebar.slider("Number of Similar Products on Shelf", 2, 25, 10)


# ============================
# TABS FOR DIFFERENT ANALYSES
# ============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Purchase Prediction & Placement", 
    "🧠 Behavioral Psychology Analysis",
    "📊 Factor Impact Analysis", 
    "🔬 SHAP Explainability",
    "🛒 Market Basket Analysis"
])


# ============================
# TAB 1: PURCHASE PREDICTION & PLACEMENT RECOMMENDATION
# ============================
with tab1:
    st.header("🎯 Purchase Prediction & Optimal Placement")
    
    # Build features dictionary
    features = {
        "transaction_id": 1,
        "customer_id": 1,
        "gender": customer_gender,
        "age": customer_age,
        "income_level": customer_income,
        "store_region": store_region,
        "visit_day_type": visit_day,
        "time_of_day": visit_time,
        "festival_season": festival if festival != "None" else "nan",
        "payment_method": "eSewa",
        "product_category": product_category,
        "brand_tier": brand_tier,
        "price_level": int(product_price),
        "discount_pct": discount_pct,
        "is_new_arrival": 1 if is_new_arrival else 0,
        "is_featured_display": 1,
        "shelf_zone": "CenterHotspot",
        "eye_level_display": 1,
        "choice_set_size": choice_set_size,
        "anchor_price": anchor_price,
        "relative_price_to_anchor": product_price / anchor_price,
        "time_spent_section_sec": 120,
        "num_products_viewed": choice_set_size // 2,
        "novelty_preference": 0.6,
        "anchoring_sensitivity": 0.6,
        "choice_paralysis_sensitivity": 0.5
    }
    
    # Generate recommendation
    with st.spinner("Analyzing consumer behavior and generating recommendations..."):
        recommendation = engine.generate_recommendation(features)
    
    # Display results in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        purchase_prob = recommendation['purchase_probability']
        st.metric(
            "🎯 Purchase Probability",
            f"{purchase_prob*100:.1f}%",
            delta=f"{(purchase_prob - 0.5)*100:.1f}% vs avg" if purchase_prob > 0.5 else None
        )
    
    with col2:
        placement_score = recommendation['placement_score']
        st.metric(
            "📍 Placement Score",
            f"{placement_score:.2f}",
            delta="Good" if placement_score > 0.7 else "Needs Improvement"
        )
    
    with col3:
        behavioral = recommendation["behavioral_scores"]
        avg_behavioral = (behavioral['anchoring_strength'] + 
                         behavioral['novelty_boost'] - 
                         behavioral['choice_paralysis_risk']) / 3
        st.metric(
            "🧠 Behavioral Impact",
            f"{avg_behavioral:.2f}",
            delta="Positive" if avg_behavioral > 0.3 else "Negative"
        )
    
    st.markdown("---")
    
    # Recommendations
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("📍 Placement Recommendation")
        placement = recommendation['recommended_placement']
        
        if "Eye-Level Hotspot" in placement:
            st.success(f"✅ **{placement}**")
            st.write("This product should be placed at eye-level in a high-traffic area for maximum visibility.")
        elif "New Arrival" in placement:
            st.info(f"💡 **{placement}**")
            st.write("Feature this in the new arrivals section to capitalize on novelty preference.")
        elif "Reduce SKU" in placement:
            st.warning(f"⚠️ **{placement}**")
            st.write("Too many options are causing choice paralysis. Reduce clutter around this product.")
        else:
            st.info(f"📌 **{placement}**")
        
        # Placement visualization
        zones = ["Eye-Level\nHotspot", "New Arrival\nSection", "Standard\nShelf", "End Cap", "Back Wall"]
        scores = [placement_score if "Eye-Level" in placement else 0.3,
                 placement_score if "New Arrival" in placement else 0.4,
                 0.5, 0.4, 0.2]
        
        fig_placement = go.Figure(data=[
            go.Bar(x=zones, y=scores, marker_color=['#00cc96' if s == max(scores) else '#636EFA' for s in scores])
        ])
        fig_placement.update_layout(
            title="Placement Zone Effectiveness",
            xaxis_title="Store Zone",
            yaxis_title="Effectiveness Score",
            height=300
        )
        st.plotly_chart(fig_placement, use_container_width=True)
    
    with col_b:
        st.subheader("🛍️ SKU Optimization")
        sku_suggestion = recommendation['sku_suggestion']
        
        if "optimal" in sku_suggestion.lower():
            st.success(f"✅ {sku_suggestion}")
        elif "Reduce" in sku_suggestion:
            st.warning(f"⚠️ {sku_suggestion}")
            st.write(f"Current: **{choice_set_size}** products → Recommended: **8-12** products")
        else:
            st.info(f"💡 {sku_suggestion}")
        
        # Choice paralysis visualization
        choice_range = list(range(2, 26))
        conversion_rates = [1 / (1 + 0.05 * max(0, x - 8)) for x in choice_range]
        
        fig_choice = go.Figure()
        fig_choice.add_trace(go.Scatter(
            x=choice_range, y=conversion_rates,
            mode='lines', name='Conversion Rate',
            line=dict(color='#636EFA', width=2)
        ))
        fig_choice.add_vline(x=choice_set_size, line_dash="dash", 
                            line_color="red", annotation_text="Current")
        fig_choice.add_vline(x=10, line_dash="dash", 
                            line_color="green", annotation_text="Optimal")
        
        fig_choice.update_layout(
            title="Choice Set Size Impact on Conversion",
            xaxis_title="Number of Products",
            yaxis_title="Relative Conversion Rate",
            height=300
        )
        st.plotly_chart(fig_choice, use_container_width=True)


# ============================
# TAB 2: BEHAVIORAL PSYCHOLOGY ANALYSIS
# ============================
with tab2:
    st.header("🧠 Behavioral Psychology Analysis")
    
    behavioral_scores = recommendation["behavioral_scores"]
    
    # Display behavioral metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        anchoring = behavioral_scores['anchoring_strength']
        st.metric("⚓ Anchoring Effect", f"{anchoring:.2f}")
        st.caption(f"Price relative to anchor: {(product_price/anchor_price):.2%}")
    
    with col2:
        novelty = behavioral_scores['novelty_boost']
        st.metric("✨ Novelty Bias", f"{novelty:.2f}")
        st.caption("New arrival" if is_new_arrival else "Regular product")
    
    with col3:
        paralysis = behavioral_scores['choice_paralysis_risk']
        st.metric("🌀 Choice Paralysis Risk", f"{paralysis:.2f}")
        st.caption(f"{choice_set_size} options available")
    
    st.markdown("---")
    
    # Detailed behavioral analysis
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("⚓ Anchoring Effect Analysis")
        st.write(f"""
        **Current Setup:**
        - Anchor Price: NPR {anchor_price:,}
        - Product Price: NPR {product_price:,}
        - Relative Price: {(product_price/anchor_price)*100:.1f}%
        
        **Impact:** {'Strong anchoring effect - product appears cheaper!' if product_price < anchor_price else 'Weak anchoring - product appears expensive.'}
        """)
        
        # Anchoring sensitivity curve
        price_range = np.linspace(0.3, 1.5, 100)
        anchoring_effect = np.maximum(0, (price_range - 0.5) * 2)
        
        fig_anchor = go.Figure()
        fig_anchor.add_trace(go.Scatter(
            x=price_range * 100, y=anchoring_effect,
            mode='lines', fill='tozeroy',
            name='Anchoring Strength',
            line=dict(color='#FF6692')
        ))
        fig_anchor.add_vline(x=(product_price/anchor_price)*100, 
                            line_dash="dash", line_color="red",
                            annotation_text="Current")
        
        fig_anchor.update_layout(
            title="Anchoring Effect vs Price Ratio",
            xaxis_title="Price as % of Anchor",
            yaxis_title="Anchoring Strength",
            height=300
        )
        st.plotly_chart(fig_anchor, use_container_width=True)
    
    with col_b:
        st.subheader("✨ Novelty Bias Impact")
        
        if is_new_arrival:
            st.success("🆕 This is a **New Arrival** - 50% novelty boost!")
            st.write("New products attract more attention and higher purchase intent.")
        else:
            st.info("📦 Regular Product - Standard appeal")
            st.write("Consider marking as 'New Arrival' to boost interest.")
        
        # Novelty decay over time
        days = np.linspace(0, 60, 100)
        novelty_over_time = 0.6 * (1.5 if is_new_arrival else 1.0) * np.exp(-days / 20)
        
        fig_novelty = go.Figure()
        fig_novelty.add_trace(go.Scatter(
            x=days, y=novelty_over_time,
            mode='lines', fill='tozeroy',
            name='Novelty Effect',
            line=dict(color='#00CC96')
        ))
        
        fig_novelty.update_layout(
            title="Novelty Effect Decay Over Time",
            xaxis_title="Days Since Launch",
            yaxis_title="Novelty Effect Strength",
            height=300
        )
        st.plotly_chart(fig_novelty, use_container_width=True)
    
    st.markdown("---")
    
    # Combined behavioral impact
    st.subheader("📊 Combined Behavioral Impact")
    
    behaviors = ['Anchoring\nEffect', 'Novelty\nBias', 'Choice\nParalysis\n(Negative)', 'Net\nBehavioral\nImpact']
    values = [
        anchoring,
        novelty,
        -paralysis,
        anchoring + novelty - paralysis
    ]
    colors = ['#FF6692', '#00CC96', '#EF553B', '#636EFA']
    
    fig_combined = go.Figure(data=[
        go.Bar(x=behaviors, y=values, marker_color=colors)
    ])
    fig_combined.update_layout(
        title="Behavioral Factors Impact on Purchase Decision",
        yaxis_title="Impact Score",
        height=350
    )
    st.plotly_chart(fig_combined, use_container_width=True)


# ============================
# TAB 3: FACTOR IMPACT ANALYSIS
# ============================
with tab3:
    st.header("📊 Interactive Factor Impact Analysis")
    st.write("Explore how different factors influence purchase behavior")
    
    # Factor selection
    factor_to_analyze = st.selectbox(
        "Select Factor to Analyze",
        ["Discount %", "Choice Set Size", "Anchor Price", "Product Price", 
         "Income Level", "Time of Day", "Festival Season"]
    )
    
    st.markdown("---")
    
    if factor_to_analyze == "Discount %":
        st.subheader("💰 Impact of Discount on Purchase Probability")
        
        discount_range = range(0, 51, 5)
        probabilities = []
        
        for disc in discount_range:
            test_features = features.copy()
            test_features['discount_pct'] = disc
            prob = engine.predict_purchase(test_features)
            probabilities.append(prob)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(discount_range), y=probabilities,
            mode='lines+markers',
            name='Purchase Probability',
            line=dict(color='#00CC96', width=3),
            marker=dict(size=8)
        ))
        fig.add_vline(x=discount_pct, line_dash="dash", line_color="red",
                     annotation_text=f"Current: {discount_pct}%")
        
        fig.update_layout(
            title="Purchase Probability vs Discount Percentage",
            xaxis_title="Discount %",
            yaxis_title="Purchase Probability",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        optimal_discount = discount_range[probabilities.index(max(probabilities))]
        st.info(f"💡 **Insight:** Optimal discount for this scenario is **{optimal_discount}%** with {max(probabilities)*100:.1f}% purchase probability")
    
    elif factor_to_analyze == "Choice Set Size":
        st.subheader("🌀 Impact of Product Variety on Purchase Decision")
        
        choice_range = range(2, 26, 2)
        probabilities = []
        
        for choice in choice_range:
            test_features = features.copy()
            test_features['choice_set_size'] = choice
            test_features['num_products_viewed'] = choice // 2
            prob = engine.predict_purchase(test_features)
            probabilities.append(prob)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(choice_range), y=probabilities,
            mode='lines+markers',
            name='Purchase Probability',
            line=dict(color='#AB63FA', width=3),
            marker=dict(size=8)
        ))
        fig.add_vline(x=choice_set_size, line_dash="dash", line_color="red",
                     annotation_text=f"Current: {choice_set_size}")
        fig.add_vrect(x0=8, x1=12, fillcolor="green", opacity=0.1,
                     annotation_text="Optimal Range")
        
        fig.update_layout(
            title="Purchase Probability vs Number of Product Options",
            xaxis_title="Number of Products on Shelf",
            yaxis_title="Purchase Probability",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning("⚠️ **Choice Paralysis:** Too many options reduce purchase likelihood!")
    
    elif factor_to_analyze == "Anchor Price":
        st.subheader("⚓ Impact of Anchor Price on Purchase Decision")
        
        anchor_range = np.linspace(product_price * 0.5, product_price * 2, 20)
        probabilities = []
        
        for anchor in anchor_range:
            test_features = features.copy()
            test_features['anchor_price'] = int(anchor)
            test_features['relative_price_to_anchor'] = product_price / anchor
            prob = engine.predict_purchase(test_features)
            probabilities.append(prob)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=anchor_range, y=probabilities,
            mode='lines+markers',
            name='Purchase Probability',
            line=dict(color='#FF6692', width=3),
            marker=dict(size=8)
        ))
        fig.add_vline(x=anchor_price, line_dash="dash", line_color="red",
                     annotation_text=f"Current: NPR {anchor_price:,}")
        
        fig.update_layout(
            title="Purchase Probability vs Anchor Price",
            xaxis_title="Anchor Price (NPR)",
            yaxis_title="Purchase Probability",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Anchoring Effect:** Higher anchor prices make the product seem like a better deal!")
    
    elif factor_to_analyze == "Income Level":
        st.subheader("💼 Impact of Customer Income Level")
        
        income_levels = ["Low", "Middle", "High"]
        probabilities = []
        
        for income in income_levels:
            test_features = features.copy()
            test_features['income_level'] = income
            prob = engine.predict_purchase(test_features)
            probabilities.append(prob)
        
        fig = go.Figure(data=[
            go.Bar(x=income_levels, y=probabilities, 
                  marker_color=['#EF553B', '#636EFA', '#00CC96'])
        ])
        
        fig.update_layout(
            title="Purchase Probability by Income Level",
            xaxis_title="Income Level",
            yaxis_title="Purchase Probability",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif factor_to_analyze == "Time of Day":
        st.subheader("⏰ Impact of Visit Time on Purchase Behavior")
        
        times = ["Morning", "Afternoon", "Evening"]
        probabilities = []
        
        for time in times:
            test_features = features.copy()
            test_features['time_of_day'] = time
            prob = engine.predict_purchase(test_features)
            probabilities.append(prob)
        
        fig = go.Figure(data=[
            go.Bar(x=times, y=probabilities,
                  marker_color=['#FFA15A', '#19D3F3', '#B6E880'])
        ])
        
        fig.update_layout(
            title="Purchase Probability by Time of Day",
            xaxis_title="Time of Day",
            yaxis_title="Purchase Probability",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif factor_to_analyze == "Festival Season":
        st.subheader("🎉 Impact of Festival Season")
        
        festivals = ["nan", "Dashain", "Tihar", "NewYear", "OtherFest"]
        festival_labels = ["Regular", "Dashain", "Tihar", "New Year", "Other"]
        probabilities = []
        
        for fest in festivals:
            test_features = features.copy()
            test_features['festival_season'] = fest
            prob = engine.predict_purchase(test_features)
            probabilities.append(prob)
        
        fig = go.Figure(data=[
            go.Bar(x=festival_labels, y=probabilities,
                  marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'])
        ])
        
        fig.update_layout(
            title="Purchase Probability by Festival Season",
            xaxis_title="Festival Season",
            yaxis_title="Purchase Probability",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================
# TAB 4: SHAP EXPLAINABILITY
# ============================
with tab4:
    st.header("🔬 SHAP Explainability - Feature Importance")
    st.write("Understanding which factors drive purchase decisions globally")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📌 SHAP Summary Plot")
        if os.path.exists("reports/shap/shap_summary.png"):
            st.image("reports/shap/shap_summary.png", width="stretch")
        else:
            st.warning("⚠️ SHAP summary plot not found. Run `python src/explainability/shap_analysis.py` first.")
        
        st.subheader("📊 SHAP Feature Importance (Bar)")
        if os.path.exists("reports/shap/shap_bar.png"):
            st.image("reports/shap/shap_bar.png", width="stretch")
        else:
            st.warning("⚠️ SHAP bar plot not found.")
    
    with col2:
        st.subheader("🎯 Top Feature Influences")
        if os.path.exists("reports/shap/shap_feature_ranking.csv"):
            shap_csv = pd.read_csv("reports/shap/shap_feature_ranking.csv")
            st.dataframe(shap_csv.head(20), height=400)
            
            # Top 10 features visualization
            top_features = shap_csv.head(10)
            fig_top = go.Figure(data=[
                go.Bar(y=top_features['feature'][::-1], 
                      x=top_features['importance'][::-1],
                      orientation='h',
                      marker_color='#636EFA')
            ])
            fig_top.update_layout(
                title="Top 10 Most Important Features",
                xaxis_title="SHAP Importance",
                yaxis_title="Feature",
                height=400
            )
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.warning("⚠️ SHAP feature ranking not found.")
    
    st.markdown("---")
    
    # Force plot
    if os.path.exists("reports/shap/force_plot_sample.html"):
        st.subheader("🎯 SHAP Force Plot (Local Explanation)")
        st.caption("Shows how each feature contributes to a specific prediction")
        st.components.v1.html(
            open("reports/shap/force_plot_sample.html", "r", encoding="utf-8").read(),
            height=300,
            scrolling=True
        )


# ============================
# TAB 5: MARKET BASKET ANALYSIS
# ============================
with tab5:
    st.header("🛒 Market Basket Analysis & Cross-Sell Strategy")
    st.write("Analyze product co-occurrences to optimize store layout and increase basket size.")
    
    # Get MBA Data
    mba_data = engine.get_association_rules()
    lift_matrix = mba_data["lift_matrix"]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔥 Product Co-occurrence Heatmap")
        
        # Toggle between Lift and Raw Counts
        metric_type = st.radio(
            "Select Metric:",
            ["Lift Score (Strength of Association)", "Co-occurrence Count (Frequency)"],
            horizontal=True
        )
        
        if "Lift" in metric_type:
            matrix_data = mba_data["lift_matrix"]
            title = "Cross-Sell Opportunity Matrix (Lift Score)"
            zmid = 1.0
            colorscale = 'RdBu_r'
            fmt = ".1f"
            st.caption("Darker red indicates products frequently bought together (High Lift)")
        else:
            matrix_data = mba_data["co_occurrence"]
            title = "Product Co-occurrence Matrix (Raw Counts)"
            zmid = None
            colorscale = 'Blues'
            fmt = "d"
            st.caption("Darker blue indicates higher purchase frequency.")
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=matrix_data.values,
            x=matrix_data.columns,
            y=matrix_data.index,
            colorscale=colorscale,
            zmid=zmid,
            text=np.round(matrix_data.values, 1) if "Lift" in metric_type else matrix_data.values,
            texttemplate="%{text}",
            showscale=True
        ))
        
        fig_heatmap.update_layout(
            title=title,
            height=500,
            xaxis_title="Product B (Consequent)",
            yaxis_title="Product A (Antecedent)"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    with col2:
        st.subheader("💡 Strategic Placement Advice")
        
        rules = mba_data["rules"]
        top_rules = rules.drop_duplicates(subset=['lift']).head(5)
        
        for _, rule in top_rules.iterrows():
            st.success(f"**{rule['antecedent']} + {rule['consequent']}**")
            st.write(f"Strong Signal (Lift: {rule['lift']:.1f}x)")
            st.caption(f"👉 Place {rule['consequent']} next to {rule['antecedent']} to boost sales.")
            st.markdown("---")

    st.markdown("---")
    
    # Real-time Recommender
    st.subheader("🤖 Real-Time Cross-Sell Recommender")
    
    selected_cart = st.multiselect(
        "Select items currently in customer's cart:",
        options=lift_matrix.index.tolist(),
        default=["Smartphone"]
    )
    
    if selected_cart:
        recommendations = {}
        
        for item in selected_cart:
            # Get top 3 matches for this item
            matches = lift_matrix[item].sort_values(ascending=False)
            for match_item, score in matches.items():
                if match_item not in selected_cart and score > 1.1:
                    if match_item in recommendations:
                        recommendations[match_item] = max(recommendations[match_item], score)
                    else:
                        recommendations[match_item] = score
        
        if recommendations:
            st.write("### ✅ Recommended Add-ons:")
            
            rec_cols = st.columns(len(recommendations))
            
            sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:4]
            
            for i, (item, score) in enumerate(sorted_recs):
                with st.container():
                    st.info(f"**{item}**")
                    st.caption(f"Lift Score: {score:.1f}x")
                    st.progress(min(1.0, score/3.0))
        else:
            st.warning("No strong recommendations found for this combination.")


# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown("""
**RetailAI Dashboard** | Built with Streamlit, CatBoost, and SHAP  
*Analyze consumer behavior, optimize product placement, and maximize sales with AI-powered insights*
""")
