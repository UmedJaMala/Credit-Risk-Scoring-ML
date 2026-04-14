import streamlit as st
import numpy as np
import joblib
import os

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="سیستەمی زیرەکی نمرەدانی مەترسی و سنووری قەرز",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="auto",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  —  Premium Financial Theme (Navy & Teal)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;600;800&display=swap');

/* بنچینەی پەڕە و فۆنت */
html, body, [class*="css"] {
    font-family: 'Noto Sans Arabic', sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
    color: #e8f4fd;
}

/* باکگراوندی گشتی */
.stApp {
    background: linear-gradient(160deg, #011928 0%, #001e38 40%, #00111f 100%);
}

/* 💠 دیزاینی مۆدێرنی بەتنی (دەربارەی پڕۆژە) - Expander Header 💠 */
.streamlit-expanderHeader {
    background-color: rgba(0, 212, 255, 0.05) !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
    transition: all 0.3s ease-in-out !important;
    margin-bottom: 10px !important;
}

.streamlit-expanderHeader:hover {
    background-color: rgba(0, 212, 255, 0.15) !important;
    border-color: #00d4ff !important;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.2) !important;
    transform: translateY(-2px);
}

.streamlit-expanderHeader p {
    color: #00d4ff !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.5px;
}

.streamlit-expanderHeader svg {
    fill: #00d4ff !important;
}

/* ── Hero & Layout Fixes ── */
.hero {
    background: linear-gradient(120deg, rgba(0,45,75,0.9) 0%, rgba(0,62,102,0.85) 100%);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 18px;
    padding: 2rem;
    margin-bottom: 1.8rem;
    text-align: center;
    box-shadow: 0 4px 40px rgba(0,0,0,0.4);
}
.hero-title { font-size: 1.8rem; font-weight: 900; color: #ffffff; }
.hero-title span { color: #00d4ff; }

.input-card {
    background: rgba(0, 45, 75, 0.4);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

/* ئەنجامی شیکاری */
.result-card {
    border-radius: 18px;
    padding: 1.8rem;
    box-shadow: 0 6px 32px rgba(0,0,0,0.3);
    margin-bottom: 1rem;
}
.rc-low { background: linear-gradient(135deg, #001e38 0%, #004d30 100%); border: 1.5px solid #00e5a0; }
.rc-high { background: linear-gradient(135deg, #001e38 0%, #4b0a19 100%); border: 1.5px solid #ff4d6d; }
.rc-limit { background: linear-gradient(135deg, #001e38 0%, #003d66 100%); border: 1.5px solid #00d4ff; }

/* Responsive Mobile */
@media (max-width: 768px) {
    div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    base = "outputs"
    try:
        risk_model  = joblib.load(os.path.join(base, "risk_model.joblib"))
        limit_model = joblib.load(os.path.join(base, "limit_model.joblib"))
        scaler      = joblib.load(os.path.join(base, "scaler.joblib"))
        return risk_model, limit_model, scaler, True
    except:
        return None, None, None, False

risk_model, limit_model, scaler, models_loaded = load_models()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div style="font-size: 3rem;">🏦</div>
    <div class="hero-title">سیستەمی زیرەکی <span>نمرەدانی مەترسی</span> و سنووری قەرز</div>
    <div style="color: #7ecfee; opacity: 0.7;">Intelligent Credit Risk AI Engine</div>
</div>
""", unsafe_allow_html=True)

# 🔘 بەتنی مۆدێرنی زانیاری پڕۆژە
with st.expander("📋 دەربارەی پڕۆژە و گەشەپێدەر", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **📝 کورتەی پڕۆژە**  
        ئەم سیستەمە مۆدێلی XGBoost بەکاردەهێنێت بۆ پێشبینیکردنی توانای دارایی کڕیاران و دیاریکردنی ئاستی مەترسی قەرز بە شێوەیەکی زانستی.
        """)
    with col_b:
        st.markdown("""
        **👨‍💻 گەشەپێدەر**  
        ئومێد جەمال نووری  
        ئەندازیاری کارەبا - قۆناغی ٣  
        ساڵی خوێندن: ٢٠٢٥ — ٢٠٢٦
        """)

st.divider()

# Inputs
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("💰 زانیاری دارایی")
    annual_income = st.number_input("داهاتی ساڵانە ($)", 1000, 10000000, 50000)
    current_debt = st.number_input("کۆی قەرزەکانی ئێستا ($)", 0, 5000000, 5000)
    avg_order = st.number_input("تێکڕای بەهای کڕین ($)", 0, 500000, 1200)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("🏢 مێژووی بازرگانی")
    years_biz = st.slider("ساڵانی کارکردن", 0, 50, 5)
    missed_payments = st.selectbox("پێشینەی پارە نەدان", list(range(11)), format_func=lambda x: "هیچ" if x==0 else f"{x} جار")
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🔮 شیکردنەوە و بڕیاردانی AI", use_container_width=True):
    if models_loaded:
        input_data = np.array([[annual_income, current_debt, years_biz, missed_payments, avg_order]])
        input_scaled = scaler.transform(input_data)
        
        risk_pred = risk_model.predict(input_scaled)[0]
        risk_prob = risk_model.predict_proba(input_scaled)[0][1] * 100
        limit_pred = limit_model.predict(input_scaled)[0]
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if risk_pred == 1:
                st.markdown(f"<div class='result-card rc-high'><h3>⚠️ مەترسی: بەرز</h3><p>ئەگەر: {risk_prob:.1f}%</p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-card rc-low'><h3>✅ مەترسی: نزم</h3><p>ئەگەر: {risk_prob:.1f}%</p></div>", unsafe_allow_html=True)
        with res_col2:
            st.markdown(f"<div class='result-card rc-limit'><h3>💳 سنووری قەرز</h3><h2>${limit_pred:,.0f}</h2></div>", unsafe_allow_html=True)
    else:
        st.error("مۆدێلەکان بار نەکراون!")

st.markdown("<br><center><small>Developed by Umed Jamal Nouri | 2026</small></center>", unsafe_allow_html=True)