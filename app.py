import streamlit as st
import numpy as np
import joblib
import xgboost
# 1. بارکردنی مۆدێلەکان
@st.cache_resource
def load_models():
    clf = joblib.load('outputs/risk_model.joblib')
    reg = joblib.load('outputs/limit_model.joblib')
    scaler = joblib.load('outputs/scaler.joblib')
    return clf, reg, scaler

xgb_clf, xgb_reg, scaler = load_models()

# 2. ڕێکخستنی ڕووکاری وێبسایتەکە
st.set_page_config(page_title="AI Credit Scoring", page_icon="💳", layout="centered")

st.title("💳 Intelligent Credit Limit & Risk Scoring")
st.markdown("""
This AI model predicts whether a B2B customer is **High Risk** or **Low Risk**, 
and recommends an optimized **Credit Limit** based on their financial data.
""")

st.divider()

st.header("👤 Customer Financial Profile")

# 3. وەرگرتنی زانیاری لە بەکارهێنەر
col1, col2 = st.columns(2)

with col1:
    annual_income = st.number_input("Annual Income ($)", min_value=10000, value=120000, step=5000)
    current_debt = st.number_input("Current Debt ($)", min_value=0, value=30000, step=1000)
    years_in_business = st.number_input("Years in Business", min_value=1, max_value=50, value=5)

with col2:
    prev_missed = st.number_input("Previous Missed Payments", min_value=0, max_value=20, value=0)
    avg_order_value = st.number_input("Average Order Value ($)", min_value=100, value=1500, step=100)

st.divider()

# 4. دوگمەی شیکردنەوە و بڕیاردان
if st.button("🔮 Analyze Customer Risk", use_container_width=True):
    with st.spinner("AI is analyzing the profile..."):
        
        # ئامادەکردنی داتاکە ڕێک وەک ئەوەی مۆدێلەکە دەیخوێنێتەوە
        input_data = np.array([[annual_income, current_debt, years_in_business, prev_missed, avg_order_value]])
        input_scaled = scaler.transform(input_data)
        
        # پێشبینیکردن
        risk_pred = xgb_clf.predict(input_scaled)[0]
        risk_prob = xgb_clf.predict_proba(input_scaled)[0][1] * 100
        
        limit_pred = xgb_reg.predict(input_scaled)[0]
        
    # 5. پیشاندانی ئەنجامەکان
    st.subheader("📊 AI Recommendation")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if risk_pred == 1:
            st.error(f"**Risk Level: HIGH**\n\nProbability: {risk_prob:.1f}%")
            st.caption("⚠️ High chance of defaulting.")
        else:
            st.success(f"**Risk Level: LOW**\n\nProbability: {risk_prob:.1f}%")
            st.caption("✅ Safe to extend credit.")
            
    with res_col2:
        st.info(f"**Recommended Credit Limit:**\n\n### ${limit_pred:,.2f}")
        st.caption("Optimized based on financial capacity.")