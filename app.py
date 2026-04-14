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
#  GLOBAL CSS - Fixed for Mobile Responsiveness
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --navy: #002d4b;
    --teal: #00d4ff;
    --text-main: #e8f4fd;
    --border: rgba(0, 212, 255, 0.18);
}

/* بنچینەی گشتی */
html, body, [class*="css"] {
    font-family: 'Noto Sans Arabic', sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
}

.stApp {
    background: linear-gradient(160deg, #011928 0%, #001e38 40%, #00111f 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }

/* ڕێکخستنی شاشە بۆ مۆبایل و کۆمپیوتەر */
.block-container {
    padding: 2rem 2rem !important;
    max-width: 1000px !important;
}

/* 💠 شێوازی ایکسپاندەر (بەتنی زانیاری پڕۆژە) */
.streamlit-expanderHeader p {
    color: #00d4ff !important;
    font-weight: bold !important;
    font-size: 1rem !important;
}

/* ════════════════════════════════════════════════════════
   HERO BANNER
════════════════════════════════════════════════════════ */
.hero {
    background: linear-gradient(120deg, rgba(0,45,75,0.9) 0%, rgba(0,62,102,0.85) 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 40px rgba(0,0,0,0.4);
}
.hero-title {
    font-size: clamp(1.2rem, 5vw, 2rem);
    font-weight: 900;
    color: var(--text-main);
}
.hero-title span { color: var(--teal); }

/* ════════════════════════════════════════════════════════
   INPUT CARDS
════════════════════════════════════════════════════════ */
.input-card {
    background: rgba(0,45,75,0.5);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}

/* ════════════════════════════════════════════════════════
   RESULT CARDS
════════════════════════════════════════════════════════ */
.result-card {
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 32px rgba(0,0,0,0.3);
}
.rc-low { background: linear-gradient(135deg, #001e38 0%, #004d30 100%); border: 1.5px solid #00e5a0; color: #00e5a0; }
.rc-high { background: linear-gradient(135deg, #001e38 0%, #4b0a19 100%); border: 1.5px solid #ff4d6d; color: #ff4d6d; }
.rc-limit { background: linear-gradient(135deg, #001e38 0%, #003d66 100%); border: 1.5px solid #00d4ff; color: #00d4ff; }

/* 📱 چاککردنی کێشەی مۆبایل (Media Query) */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 0.8rem !important;
    }
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    .hero {
        padding: 1.5rem 1rem;
    }
    .rc-value {
        font-size: 1.8rem !important;
    }
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
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div style="font-size: 2.5rem;">🏦</div>
    <div class="hero-title">سیستەمی زیرەکی <span>نمرەدانی مەترسی</span> و سنووری قەرز</div>
    <div style="color: #7ecfee; opacity: 0.6; font-size: 0.8rem;">Intelligent Credit Risk AI System</div>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ زانیاری زیاتر دەربارەی پڕۆژە و گەشەپێدەر", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**📋 کورتەی پڕۆژە:**")
        st.write("ئەم پڕۆژەیە مۆدێلی XGBoost بەکاردێنێت بۆ شیکردنەوەی داتای دارایی و بازرگانی کڕیاران بە مەبەستی کەمکردنەوەی زیانی قەرز.")
    with col_b:
        st.write("**👨‍💻 گەشەپێدەر:**")
        st.write("ئومێد جمال نوری | ئەندازیاری کارەبا - قۆناغی سێیەم (٢٠٢٦)")

st.divider()

# Inputs
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("💰 زانیاری دارایی")
    annual_income = st.number_input("داهاتی ساڵانە ($)", 1000, 10000000, 120000)
    current_debt = st.number_input("کۆی قەرزەکانی ئێستا ($)", 0, 5000000, 30000)
    avg_order = st.number_input("تێکڕای بەهای کڕینەکان ($)", 0, 500000, 1500)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("🏢 مێژووی بازرگانی")
    years_biz = st.slider("ساڵانی کارکردن (بزنس)", 0, 50, 5)
    missed_payments = st.selectbox("پێشینەی پارە نەدان (وەسڵەکان)", list(range(11)), index=1, format_func=lambda x: "هیچ" if x==0 else f"{x} جار")
    st.markdown('</div>', unsafe_allow_html=True)

# Analyze Button
if st.button("🔮 شیکردنەوە و بڕیاردانی AI", use_container_width=True):
    if models_loaded:
        input_data = np.array([[annual_income, current_debt, years_biz, missed_payments, avg_order]])
        input_scaled = scaler.transform(input_data)
        
        risk_pred = risk_model.predict(input_scaled)[0]
        risk_prob = risk_model.predict_proba(input_scaled)[0][1] * 100
        limit_pred = limit_model.predict(input_scaled)[0]
        
        st.subheader("📋 ئەنجامی شیکاری")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            status = "بەرز ⚠️" if risk_pred == 1 else "نزم ✅"
            style = "rc-high" if risk_pred == 1 else "rc-low"
            st.markdown(f"<div class='result-card {style}'><h3>ئاستی مەترسی: {status}</h3><p>ئەگەری مەترسی: {risk_prob:.1f}%</p></div>", unsafe_allow_html=True)
        
        with res_col2:
            st.markdown(f"<div class='result-card rc-limit'><h3>سنووری قەرز</h3><h2>${limit_pred:,.2f}</h2></div>", unsafe_allow_html=True)
    else:
        st.error("⚠️ هەڵە: مۆدێلەکان بار نەکراون!")

st.markdown("<br><center><small style='color: #555;'>Powered by XGBoost AI Engine | 2026</small></center>", unsafe_allow_html=True)