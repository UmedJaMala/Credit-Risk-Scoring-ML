# --- START OF FILE app.py ---

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="سیستەمی زیرەکی نمرەدانی مەترسی - کۆگاکانی هەولێر",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMIC MODEL METRICS LOAD
# ══════════════════════════════════════════════════════════════════════════════
metrics_data = {}
metrics_path = os.path.join("outputs", "model_metrics.json")

if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS - DARK LIQUID GLASS EDITION 
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700;800;900&family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg-base:      #050505;
    --glass-bg:     rgba(15, 17, 21, 0.45);
    --glass-blur:   blur(24px) saturate(180%);
    --glass-border: rgba(255, 255, 255, 0.06);
    --glass-hi:     rgba(255, 255, 255, 0.15);
    --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);

    --cyan:         #22d3ee;
    --cyan-dim:     rgba(34,211,238,0.15);
    --cyan-bdr:     rgba(34,211,238,0.35);
    
    --violet:       #a78bfa;
    --violet-dim:   rgba(167,139,250,0.15);
    
    --burgundy:     #800020;
    --burgundy-dim: rgba(128,0,32,0.25);

    --green:        #34d399;
    --green-dim:    rgba(52,211,153,0.15);
    
    --red:          #fb7185;
    --red-dim:      rgba(251,113,133,0.15);
    
    --amber:        #fbbf24;

    --text:         #f8fafc;
    --text-2:       #cbd5e1;
    --text-3:       #64748b;

    --blue:         #3b82f6;
    --blue-dim:     rgba(59,130,246,0.15);
    --blue-bdr:     rgba(59,130,246,0.4);

    --r:            20px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Noto Sans Arabic', 'Inter', sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
}

p, h1, h2, h3, h4, h5, h6, span, li,
div[data-testid="stMarkdownContainer"],
div[data-testid="stMarkdownContainer"] p {
    color: var(--text) !important;
}

.stApp {
    background-color: var(--bg-base);
    background-image: 
        radial-gradient(circle at 15% 10%, var(--burgundy-dim), transparent 35%),
        radial-gradient(circle at 85% 85%, var(--blue-dim), transparent 35%),
        radial-gradient(circle at 50% 50%, rgba(34,211,238,0.05), transparent 60%);
    background-attachment: fixed;
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 1.6rem 4rem !important; max-width: 1060px !important; }

.liquid-glass {
    background: var(--glass-bg);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid var(--glass-border);
    border-top: 1px solid var(--glass-hi);
    border-left: 1px solid rgba(255,255,255,0.1);
    box-shadow: var(--glass-shadow);
    border-radius: var(--r);
}

.hero { padding: 2.4rem 2rem 2rem; margin-bottom: 1.8rem; text-align: center; position: relative; overflow: hidden; }
.hero-icon { font-size: 3rem; line-height: 1; margin-bottom: 0.7rem; filter: drop-shadow(0 0 15px rgba(59,130,246,0.6)); }
.hero-title { font-size: clamp(1.4rem, 3.5vw, 2.2rem); font-weight: 900; color: var(--text); line-height: 1.45; margin-bottom: 0.4rem; }
.hero-title span { background: linear-gradient(to left, var(--blue), #bfdbfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 20px rgba(59,130,246,0.3); }
.hero-sub  { color: var(--text-2); font-size: 0.95rem; font-weight: 500; letter-spacing: 0.03em; }
.hero-pill { display: inline-block; margin-top: 1.1rem; background: rgba(59,130,246,0.15); border: 1px solid var(--blue-bdr); border-radius: 50px; padding: 0.35rem 1.2rem; font-size: 0.75rem; font-weight: 800; color: #bfdbfe; letter-spacing: 0.10em; backdrop-filter: blur(10px); }

div[data-testid="stBaseButton-primary"] button, button[kind="primary"] {
    width: 100% !important; background: linear-gradient(135deg, rgba(59,130,246,0.2) 0%, rgba(59,130,246,0.05) 100%) !important; backdrop-filter: blur(12px) !important; border: 1px solid var(--blue-bdr) !important; border-top: 1px solid rgba(255,255,255,0.3) !important; color: var(--blue) !important; font-family: 'Noto Sans Arabic', sans-serif !important; font-size: 1.05rem !important; font-weight: 900 !important; border-radius: 16px !important; padding: 0.85rem 1.2rem !important; letter-spacing: 0.05em !important; box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important; transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}
div[data-testid="stBaseButton-primary"] button:hover, button[kind="primary"]:hover { transform: translateY(-3px) !important; background: linear-gradient(135deg, rgba(59,130,246,0.3) 0%, rgba(59,130,246,0.1) 100%) !important; box-shadow: 0 8px 30px rgba(59,130,246,0.25) !important; color: #fff !important; }

div[data-testid="stBaseButton-secondary"] button, button[kind="secondary"] {
    width: 100% !important; background: var(--glass-bg) !important; backdrop-filter: blur(12px) !important; border: 1px solid var(--glass-border) !important; border-top: 1px solid var(--glass-hi) !important; color: var(--cyan) !important; font-family: 'Inter', sans-serif !important; font-size: 0.95rem !important; font-weight: 800 !important; border-radius: 16px !important; padding: 0.85rem 1.2rem !important; box-shadow: var(--glass-shadow) !important; transition: all 0.3s ease !important;
}
div[data-testid="stBaseButton-secondary"] button:hover, button[kind="secondary"]:hover { transform: translateY(-2px) !important; background: rgba(34,211,238,0.1) !important; border-color: var(--cyan-bdr) !important; color: #fff !important; }

.sec-head { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1.2rem; }
.sec-head-line { flex: 1; height: 1px; background: linear-gradient(90deg, rgba(255,255,255,0.2) 0%, transparent 100%); }
.sec-head-text { color: #fff; font-size: 0.85rem; font-weight: 800; letter-spacing: 0.10em; white-space: nowrap; text-shadow: 0 0 10px rgba(255,255,255,0.3); }

/* Dialog Section Styling (English Left-to-Right) */
.eng-dialog { direction: ltr !important; text-align: left !important; font-family: 'Inter', sans-serif !important; }
.eng-dialog p, .eng-dialog h1, .eng-dialog h2, .eng-dialog h3, .eng-dialog span, .eng-dialog div { direction: ltr !important; text-align: left !important; font-family: 'Inter', sans-serif !important; }

/* Custom Tabs Styling */
div[data-baseweb="tab-list"] { border-bottom: 1px solid rgba(255,255,255,0.1) !important; gap: 2rem; direction: rtl !important; }
button[data-baseweb="tab"] { background: transparent !important; padding: 1rem 0 !important; border: none !important; }
button[data-baseweb="tab"] p { color: var(--text-2) !important; font-weight: 700 !important; font-size: 1rem !important; }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--blue) !important; text-shadow: 0 0 10px rgba(59,130,246,0.4); }
div[data-baseweb="tab-highlight"] { background-color: var(--blue) !important; box-shadow: 0 0 10px rgba(59,130,246,0.5) !important; height: 3px !important; border-radius: 3px 3px 0 0 !important; }

.input-card { padding: 1.8rem 1.5rem 1.5rem; height: 100%; }
.card-title { color: #fff; font-size: 1rem; font-weight: 800; margin-bottom: 1.3rem; padding-bottom: 0.8rem; border-bottom: 1px solid var(--glass-border); display: flex; align-items: center; gap: 0.5rem; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }

label, div[data-testid="stWidgetLabel"] > p, .stSlider label, .stNumberInput label, .stSelectbox label { 
    color: var(--text-2) !important; 
    font-family: 'Noto Sans Arabic', sans-serif !important; 
    font-weight: 600 !important; 
    font-size: 0.9rem !important; 
    direction: rtl !important; 
    text-align: right !important; 
}

div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stNumberInput"] div[data-baseweb="base-input"],
div[data-testid="stNumberInput"] div[data-baseweb="input"] > div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: transparent !important;
    border: none !important;
}

div[data-testid="stNumberInputStepUp"],
div[data-testid="stNumberInputStepDown"] {
    background-color: rgba(255,255,255,0.05) !important;
    color: #fff !important;
    border-radius: 8px !important;
}

.stNumberInput input, .stSelectbox > div > div, .stSelectbox > div > div > div { 
    background-color: rgba(0,0,0,0.4) !important; 
    -webkit-appearance: none !important; 
    -moz-appearance: none !important;
    appearance: none !important;
    backdrop-filter: blur(10px) !important; 
    border: 1px solid var(--glass-border) !important; 
    border-top: 1px solid rgba(255,255,255,0.08) !important; 
    border-radius: 12px !important; 
    color: #fff !important; 
    font-family: 'Noto Sans Arabic', sans-serif !important; 
    font-size: 0.97rem !important; 
    font-weight: 700 !important; 
    padding: 0.6rem 1rem !important; 
    box-shadow: inset 0 2px 5px rgba(0,0,0,0.5) !important; 
    transition: all 0.3s ease; 
}
.stNumberInput input { direction: ltr !important; text-align: left !important; }
.stNumberInput input:focus, .stSelectbox > div > div:focus { 
    border-color: var(--blue) !important; 
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2), inset 0 2px 5px rgba(0,0,0,0.5) !important; 
    outline: none !important; 
    background-color: rgba(0,0,0,0.6) !important;
}

div[data-testid="stSlider"] { direction: ltr !important; padding: 0 0.2rem; }
div[data-testid="stSlider"] .rc-slider-rail, .stSlider .rc-slider-rail { background: rgba(0,0,0,0.5) !important; border-radius: 6px !important; height: 8px !important; box-shadow: inset 0 1px 3px rgba(0,0,0,0.6); }
div[data-testid="stSlider"] .rc-slider-track, .stSlider .rc-slider-track { background: linear-gradient(90deg, var(--burgundy), var(--blue)) !important; height: 8px !important; border-radius: 6px !important; }
div[data-testid="stSlider"] .rc-slider-handle, .stSlider .rc-slider-handle { width: 22px !important; height: 22px !important; margin-top: -7px !important; background: #fff !important; border: 4px solid var(--blue) !important; box-shadow: 0 0 15px rgba(59,130,246,0.5) !important; }

/* Small Helper Text */
.helper-text { font-size: 0.75rem; color: rgba(255,255,255,0.4); margin-top: -10px; margin-bottom: 15px; direction: rtl; text-align: right;}

.summary-card { background: rgba(255,255,255,0.03); backdrop-filter: blur(10px); border: 1px solid var(--glass-border); border-radius: 14px; padding: 1.2rem; margin-top: 1rem; direction: rtl; }
.summary-card-title { color: var(--blue); font-size: 0.8rem; font-weight: 800; margin-bottom: 0.8rem; letter-spacing: 0.05em; }
.summary-row { display: flex; justify-content: space-between; color: var(--text-2); font-size: 0.85rem; line-height: 2; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 0.2rem 0; }
.summary-row:last-child { border-bottom: none; }
.summary-val { color: #fff; font-weight: 800; direction: ltr; text-align: left; }

.result-wrap { animation: floatUp 0.6s cubic-bezier(0.22,1,0.36,1) both; }
@keyframes floatUp { from { opacity:0; transform:translateY(30px); } to { opacity:1; transform:translateY(0); } }
.result-card { padding: 2rem 1.8rem; position: relative; overflow: hidden; direction: rtl; text-align: center; }
.rc-low   { border-top: 2px solid var(--green); background: linear-gradient(180deg, rgba(52,211,153,0.1) 0%, var(--glass-bg) 100%); }
.rc-high  { border-top: 2px solid var(--red); background: linear-gradient(180deg, rgba(251,113,133,0.1) 0%, var(--glass-bg) 100%); }
.rc-limit { border-top: 2px solid var(--blue); background: linear-gradient(180deg, rgba(59,130,246,0.1) 0%, var(--glass-bg) 100%); }
.rc-eyebrow { font-size: 0.8rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1rem; }
.rc-value   { font-size: clamp(2.2rem, 5vw, 3.5rem); font-weight: 900; line-height: 1.1; margin-bottom: 0.3rem; letter-spacing: -0.02em; text-shadow: 0 4px 15px rgba(0,0,0,0.4); }
.rc-limit .rc-value { background: linear-gradient(to right, #fff, var(--blue)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.rc-en   { font-size: 0.85rem; color: var(--text-2); margin-bottom: 1.2rem; font-family: 'Inter', sans-serif; }
.rc-badge { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.4rem 1.2rem; border-radius: 50px; font-size: 0.8rem; font-weight: 800; backdrop-filter: blur(5px); }
.rc-low  .rc-eyebrow, .rc-low  .rc-value { color: var(--green); }
.rc-high .rc-eyebrow, .rc-high .rc-value { color: var(--red); }
.rc-limit .rc-eyebrow { color: var(--blue); }
.badge-low   { background: rgba(52,211,153,0.15); color: var(--green); border: 1px solid var(--green-dim); }
.badge-high  { background: rgba(251,113,133,0.15); color: var(--red); border: 1px solid var(--red-dim); }
.badge-limit { background: rgba(59,130,246,0.15); color: var(--blue); border: 1px solid var(--blue-dim); }

.metric-card { padding: 1rem 0.5rem; text-align: center; transition: transform 0.3s ease; border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; background: rgba(0,0,0,0.3); backdrop-filter: blur(10px);}
.metric-card:hover { transform: translateY(-3px); border-color: rgba(59,130,246,0.3); }
.metric-label { color: var(--text-2); font-size: 0.75rem; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: 0.05em; font-family: 'Inter', sans-serif; }
.metric-value { color: #fff; font-size: 1.5rem; font-weight: 900; line-height: 1; text-shadow: 0 2px 10px rgba(255,255,255,0.2); font-family: 'Inter', sans-serif; direction: ltr;}

div[data-testid="stModal"] > div, div[role="dialog"], section[data-testid="stDialog"] > div {
    background: rgba(10, 12, 16, 0.95) !important; 
    border: 1px solid rgba(255,255,255,0.1) !important; 
    border-top: 1px solid rgba(255,255,255,0.2) !important; 
    border-radius: 24px !important; 
    box-shadow: 0 20px 60px rgba(0,0,0,0.7) !important;
    direction: rtl !important;
}
div[role="dialog"] p, div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3, div[role="dialog"] span { color: #fff !important; }

.about-card { padding: 1.5rem; margin-bottom: 1.2rem; }
.about-card-title { color: var(--blue); font-size: 0.9rem; font-weight: 900; letter-spacing: 0.05em; margin-bottom: 0.8rem; padding-bottom: 0.6rem; border-bottom: 1px solid rgba(255,255,255,0.1); }
.about-card-body { color: var(--text-2); font-size: 0.88rem; line-height: 1.9; }
.about-card-body b { color: #fff; font-weight: 800; }
.tech-tag { display: inline-block; background: rgba(255,255,255,0.05); color: #fff; border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 0.3rem 0.8rem; font-size: 0.75rem; font-weight: 800; margin: 0.25rem 0.15rem; backdrop-filter: blur(5px); }
.about-center { text-align: center; padding: 1.5rem 0 1rem; }
.about-center-icon { font-size: 3rem; margin-bottom: 0.5rem; filter: drop-shadow(0 0 15px rgba(59,130,246,0.5)); }
.about-center-name { color: var(--blue); font-size: 1.1rem; font-weight: 900; letter-spacing: 0.08em; text-transform: uppercase; font-family: 'Inter', sans-serif;}

/* Selectbox Fix & Width Expansion */
div[data-testid="stSelectbox"] div[data-baseweb="select"] div[class*="singleValue"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #ffffff !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 0.95rem !important; 
    font-weight: 700 !important;
    line-height: normal !important;
    white-space: normal !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    min-height: 52px !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
}

div[data-baseweb="popover"] > div, div[role="listbox"], ul[role="listbox"] {
    background-color: #121418 !important;
    border: 1px solid rgba(59,130,246,0.4) !important;
    border-radius: 12px !important;
    width: auto !important;
    min-width: 100% !important;
}
div[role="listbox"] li, ul[role="listbox"] li { color: #ffffff !important; font-size: 0.95rem !important; font-weight: 700 !important; padding: 0.8rem !important; white-space: normal !important; }
div[role="listbox"] li:hover, div[role="listbox"] li[aria-selected="true"] { background-color: rgba(59,130,246,0.3) !important; }

/* Custom Alert Box for Information */
.info-box {
    background: rgba(59, 130, 246, 0.1);
    border-left: 4px solid var(--blue);
    padding: 1.2rem;
    border-radius: 8px 0 0 8px;
    margin-bottom: 1.5rem;
    direction: rtl;
    text-align: right;
}
.info-title { font-weight: 800; color: var(--blue); margin-bottom: 0.6rem; font-size: 1rem; }
.info-text { font-size: 0.9rem; color: var(--text-2); line-height: 2; }
.info-text b { color: #fff; }

.footer-section {
    text-align: center; 
    padding: 2.5rem 0 1rem; 
    color: rgba(255,255,255,0.4); 
    font-size: 0.8rem; 
    border-top: 1px solid rgba(255,255,255,0.05); 
    margin-top: 3rem; 
    font-family: 'Inter', sans-serif;
    direction: ltr;
}
.footer-section strong { color: #60a5fa; font-weight: 700; }

/* Image Styling for Mobile */
.chart-img {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    margin-bottom: 1.5rem;
    width: 100%;
    height: auto;
}

@media (max-width: 768px) {
    .block-container { padding: 1rem 0.9rem 3rem !important; }
    .hero { padding: 2rem 1rem; }
    .rc-value { font-size: 2.2rem; }
    .liquid-glass { backdrop-filter: blur(16px); } 
    div[data-baseweb="tab-list"] { gap: 1rem; }
    button[data-baseweb="tab"] p { font-size: 0.85rem !important; }
    
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        min-height: 60px !important;
        display: flex !important;
        align-items: center !important;
        padding-right: 0.2rem !important;
        padding-left: 0.2rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_ml_models():
    base = "outputs"
    try:
        rm = joblib.load(os.path.join(base, "risk_model_improved.joblib"))
        lm = joblib.load(os.path.join(base, "limit_model_improved.joblib"))
        sc_clf = joblib.load(os.path.join(base, "scaler_clf_improved.joblib"))
        sc_reg = joblib.load(os.path.join(base, "scaler_reg_improved.joblib"))
        return rm, lm, sc_clf, sc_reg, True
    except Exception:
        return None, None, None, None, False

risk_model, limit_model, scaler_clf, scaler_reg, models_loaded = load_ml_models()


# ══════════════════════════════════════════════════════════════════════════════
#  DIALOGS (DATASET & EVALUATION & ABOUT)
# ══════════════════════════════════════════════════════════════════════════════

# 1. Dataset & Model Info Dialog
@st.dialog("📊 زانیاری داتاسێت و مۆدێلەکان", width="large")
def dataset_model_info_dialog():
    st.markdown("""
    <div class="info-box">
        <div class="info-title">📁 داتاسێت (Dataset)</div>
        <div class="info-text">
            ئەم داتاسێتە پێکهاتووە لە <b>1200 کۆرپوس (ڕیز)</b> کە تایبەتە بە بازرگانی (B2B) کۆگاکانی هەولێر، پاش چارەسەرکردنی داتا نابەرامبەرەکان (Imbalanced Data) بە تەکنیکی <b>SMOTE</b> ژمارەی داتاکان بووەتە نزیکەی 1800 بە هاوسەنگی تەواو.<br>
            داتاسێتەکە پێکهاتووە لە <b>8 فیچەر (تایبەتمەندی)</b>:
            <ul style="margin-top: 5px;">
                <li><b>Shop_Age_Years:</b> تەمەنی دوکانەکە بە ساڵ (متمانەی کڕیار).</li>
                <li><b>Days_Since_Last_Order:</b> پێوەری (Recency) - چەند ڕۆژە کاڵای نەبردووە.</li>
                <li><b>Order_Freq_Per_Month:</b> پێوەری (Frequency) - چەند جار لە مانگێکدا وەسڵی هەیە.</li>
                <li><b>Average_Invoice_Value:</b> تێکڕای بەهای مامەڵەکانی.</li>
                <li><b>Total_Trade_Volume:</b> پێوەری (Monetary) - کۆی قەبارەی بازرگانی.</li>
                <li><b>Unpaid_Invoice_Ratio:</b> ڕێژەی وەسڵە نەدراوەکان.</li>
                <li><b>Debt_To_Volume_Ratio:</b> ڕێژەی قەرز بەراورد بە قەبارەی مامەڵە.</li>
                <li><b>Late_Payment_History:</b> ژمارەی جارەکانی دواکەوتنی قەرز.</li>
            </ul>
        </div>
    </div>
    
    <div class="info-box" style="border-left-color: #34d399; background: rgba(52, 211, 153, 0.1);">
        <div class="info-title" style="color: #34d399;">⚙️ مۆدێلەکانی ڕاهێنان (XGBoost)</div>
        <div class="info-text">
            لەم پڕۆژەیەدا سوود لە ئەلگۆریتمی پێشکەوتووی <b>XGBoost</b> وەرگیراوە بۆ دروستکردنی دوو مۆدێلی جیاواز کە بە تەکنیکی <b>RandomizedSearchCV</b> باشترین پارامێتەرەکانیان بۆ دۆزراوەتەوە:<br>
            <b>1. مۆدێلی پۆلێنکردن (Classification):</b> بۆ جیاکردنەوەی کڕیارەکان بۆ دوو جۆر (High Risk و Low Risk).<br>
            <b>2. مۆدێلی پێشبینیکردن (Regression):</b> بۆ پێشبینیکردنی بڕی قەرزی گونجاو (Credit Limit) بە دۆلار بۆ هەر کڕیارێک بە پشتبەستن بە مەترسییەکەی.
        </div>
    </div>
    """, unsafe_allow_html=True)


# 2. Model Evaluation Dialog
@st.dialog("📈 هەڵسەنگاندن و ئاستی مۆدێلەکان", width="large")
def model_evaluation_dialog():
    if not metrics_data:
        st.warning("⚠️ داتای هەڵسەنگاندن نەدۆزرایەوە! تکایە سەرەتا نۆتبووکەکە ڕەن بکە.")
        return

    st.markdown("""
    <div class="info-box">
        <div class="info-title">📊 تێگەیشتن لە پێوەرەکانی هەڵسەنگاندن</div>
        <div class="info-text">
            <b>Accuracy:</b> ڕێژەی سەرکەوتنی مۆدێلەکە لە دیاریکردنی مەترسی (بەرز یان نزم).<br>
            <b>ROC-AUC:</b> توانای مۆدێلەکە بۆ جیاکردنەوەی دوو جۆرە کڕیارەکە بە دروستی.<br>
            <b>F1-Score:</b> هاوسەنگی نێوان وردبینی و دۆزینەوەی دروست.<br>
            <b>R² Score:</b> ڕێژەی دروستی پێشبینیکردنی بڕی قەرزەکە (چەند نزیکە لە ڕاستی).<br>
            <b>RMSE & MAE:</b> تێکڕای هەڵەی پێشبینی قەرز بە دۆلار (هەرچەند کەمتر بێت باشترە).
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_test, tab_train = st.tabs(["🧪 ئەنجامی تاقیکردنەوە (Test Data)", "📚 ئەنجامی ڕاهێنان (Train Data)"])
    
    with tab_test:
        clf = metrics_data.get("CLF", {})
        reg = metrics_data.get("REG", {})
        
        st.markdown("""
        <div class="sec-head" style="margin-top: 1rem;">
            <span class="sec-head-text">🎯 مۆدێلی پۆلێنکردنی مەترسی (XGBoost Classifier)</span>
            <span class="sec-head-line"></span>
        </div>""", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Accuracy (ڕێژەی ڕاستی)</div>
            <div class="metric-value">{clf.get('accuracy', 0)*100:.2f}%</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">ROC-AUC Score</div>
            <div class="metric-value">{clf.get('auc_roc', 0)*100:.2f}%</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">F1-Score</div>
            <div class="metric-value">{clf.get('f1', 0)*100:.2f}%</div></div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="sec-head" style="margin-top: 2.5rem;">
            <span class="sec-head-text">💰 مۆدێلی پێشبینیکردنی قەرز (XGBoost Regressor)</span>
            <span class="sec-head-line"></span>
        </div>""", unsafe_allow_html=True)
        
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">R² Score (ڕێژەی سەرکەوتن)</div>
            <div class="metric-value">{reg.get('r2', 0)*100:.2f}%</div></div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">RMSE (تێکڕای هەڵە)</div>
            <div class="metric-value">${reg.get('rmse', 0):,.2f}</div></div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">MAE (هەڵەی ڕەها)</div>
            <div class="metric-value">${reg.get('mae', 0):,.2f}</div></div>""", unsafe_allow_html=True)
            
        img_results_path = os.path.join("outputs", "credit_risk_scoring_results.png")
        if os.path.exists(img_results_path):
            st.markdown("<br>", unsafe_allow_html=True)
            st.image(img_results_path, use_container_width=True)

        img_feat_path = os.path.join("outputs", "feature_importance.png")
        if os.path.exists(img_feat_path):
            st.markdown("<br>", unsafe_allow_html=True)
            st.image(img_feat_path, use_container_width=True)

        st.markdown("""
        <div class="info-box" style="border-left-color: #34d399; background: rgba(52, 211, 153, 0.1);">
            <div class="info-title" style="color: #34d399;">💡 ئەنجامی کۆتایی</div>
            <div class="info-text">
                مۆدێلەکەمان بە سەرکەوتوویی توانای پێشبینیکردنی هەیە بە ڕێژەی زیاتر لە <b>%87</b> بۆ مەترسی و <b>%90</b> بۆ بڕی قەرز لەسەر داتای نەبینراو (Test). ئەنجامەکانی Test و Train زۆر نزیکن لە یەکەوە، ئەمەش دەریدەخات کە مۆدێلەکە زۆر جێگیرە و کێشەی (Overfitting)ی نییە.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_train:
        clf_train = metrics_data.get("CLF_TRAIN", {})
        reg_train = metrics_data.get("REG_TRAIN", {})
        
        st.markdown("""
        <div class="sec-head" style="margin-top: 1rem;">
            <span class="sec-head-text">🎯 مۆدێلی پۆلێنکردنی مەترسی (Train Data)</span>
            <span class="sec-head-line"></span>
        </div>""", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Accuracy (ڕێژەی ڕاستی)</div>
            <div class="metric-value">{clf_train.get('accuracy', 0)*100:.2f}%</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">ROC-AUC Score</div>
            <div class="metric-value">{clf_train.get('auc_roc', 0)*100:.2f}%</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">F1-Score</div>
            <div class="metric-value">{clf_train.get('f1', 0)*100:.2f}%</div></div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="sec-head" style="margin-top: 2.5rem;">
            <span class="sec-head-text">💰 مۆدێلی پێشبینیکردنی قەرز (Train Data)</span>
            <span class="sec-head-line"></span>
        </div>""", unsafe_allow_html=True)
        
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">R² Score (ڕێژەی سەرکەوتن)</div>
            <div class="metric-value">{reg_train.get('r2', 0)*100:.2f}%</div></div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">RMSE (تێکڕای هەڵە)</div>
            <div class="metric-value">${reg_train.get('rmse', 0):,.2f}</div></div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">MAE (هەڵەی ڕەها)</div>
            <div class="metric-value">${reg_train.get('mae', 0):,.2f}</div></div>""", unsafe_allow_html=True)

# 3. About Dialog
@st.dialog("ℹ️ دەربارەی پڕۆژە و گەشەپێدەر", width="large")
def project_info_dialog():
    st.markdown("""
    <div class="about-center">
        <div class="about-center-icon">📦</div>
        <div class="about-center-name">ERBIL WAREHOUSE RISK SYSTEM</div>
        <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 0.3rem;">v 3.0 · Advanced RFM XGBoost · 2025–2026</div>
    </div>
    """, unsafe_allow_html=True)

    info_col1, info_col2 = st.columns(2, gap="large")
    with info_col1:
        st.markdown("""
        <div class="about-card liquid-glass">
            <div class="about-card-title">👨‍💻 زانیاری گەشەپێدەر</div>
            <div class="about-card-body">
                <b>ئومێد جەمال نوری</b><br>
                ئەندازیاری کارەبا - قۆناغی سێیەم<br>
                ساڵی خوێندن: ٢٠٢٥ - ٢٠٢٦<br><br>
                🔗 <a href="https://github.com/UMEDJAMALA" target="_blank" style="color: #60a5fa; text-decoration: none;"><b>GitHub Profile (@UMEDJAMALA)</b></a><br>
                🔗 <a href="https://facebook.com/umedjamala" target="_blank" style="color: #60a5fa; text-decoration: none;"><b>Facebook Profile</b></a><br>
                📧 <a href="mailto:umedjamal2005@gmail.com" style="color: #60a5fa; text-decoration: none;"><b>Contact Email</b></a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with info_col2:
        st.markdown("""
        <div class="about-card liquid-glass">
            <div class="about-card-title">⚙️ تەکنەلۆژیاکانی بەکارهاتوو</div>
            <div class="about-card-body" style="margin-bottom:0.7rem;">ئەم پڕۆژەیە بەم تەکنەلۆژیایانە دروستکراوە:</div>
            <span class="tech-tag">Python 3</span>
            <span class="tech-tag">XGBoost</span>
            <span class="tech-tag">Scikit-learn</span>
            <span class="tech-tag">Streamlit</span>
            <span class="tech-tag">Pandas</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero liquid-glass">
    <div class="hero-icon">📦</div>
    <div class="hero-title">
        سیستەمی زیرەکی <span>نمرەدانی مەترسی</span> و سنووری قەرز
    </div>
    <div class="hero-sub">Erbil Warehouse B2B Credit Limit &amp; Advanced RFM Scoring</div>
    <div class="hero-pill">⚡  XGBoost ENGINE · REAL-TIME ANALYSIS</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ACTION BUTTONS (TOP)
# ══════════════════════════════════════════════════════════════════════════════
ab_col1, ab_col2, ab_col3, ab_col4, ab_col5 = st.columns([0.5, 1.5, 1.5, 1.5, 0.5])

with ab_col2:
    if st.button("ℹ️  دەربارەی پڕۆژە", use_container_width=True, type="secondary"):
        project_info_dialog()
with ab_col3:
    if st.button("📁  داتاسێت و مۆدێل", use_container_width=True, type="secondary"):
        dataset_model_info_dialog()
with ab_col4:
    if st.button("📈  هەڵسەنگاندن", use_container_width=True, type="secondary"):
        model_evaluation_dialog()

if not models_loaded:
    st.markdown("""
    <div class="liquid-glass" style="padding: 1rem; border-color: rgba(251,113,133,0.3); color: #fb7185; text-align: center; margin-bottom: 1.5rem; margin-top: 1.5rem;">
        ⚠️ &nbsp;مۆدێلەکان نەدۆزرانەوە. تکایە سەرەتا نۆتبووکەکە ڕەن بکە بۆ دروستکردنی مۆدێلەکان.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT SECTION (8 Features with Instructions)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-head" style="margin-top: 1.5rem;">
    <span class="sec-head-text">📝  زانیارییەکانی دوکان داخڵ بکە</span>
    <span class="sec-head-line"></span>
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown('<div class="input-card liquid-glass"><div class="card-title">💰 زانیاری دارایی و مامەڵەکان</div>', unsafe_allow_html=True)
    
    avg_invoice = st.number_input("تێکڕای بەهای یەک وەسڵ ($)", min_value=0.0, value=2500.0, step=100.0, format="%.0f")
    st.markdown('<div class="helper-text">زۆربەی وەسڵەکانی ئەم دوکانە چەند دۆلارە؟ (ئاسایی 500 بۆ 5000)</div>', unsafe_allow_html=True)
    
    freq_per_month = st.number_input("تێکڕای وەسڵەکان لە مانگێکدا (دانە)", min_value=0.0, value=12.0, step=1.0)
    st.markdown('<div class="helper-text">لە مانگێکدا نزیکەی چەند جار کاڵا دەبات؟ (ئاسایی 5 بۆ 30)</div>', unsafe_allow_html=True)
    
    total_volume = st.number_input("کۆی قەبارەی بازرگانی ($)", min_value=0.0, value=50000.0, step=1000.0, format="%.0f")
    st.markdown('<div class="helper-text">کۆی ئەو پارەیەی تا ئێستا کڕینی پێ کردووە چەندە؟</div>', unsafe_allow_html=True)
    
    unpaid_ratio_display = st.slider("ڕێژەی وەسڵە نەدراوەکان لەسەدا (%)", min_value=0, max_value=100, value=10, step=1)
    st.markdown('<div class="helper-text">چەند لەسەدای وەسڵەکانی هێشتا پارەیان نەدراوە؟ (0 باشترینە)</div>', unsafe_allow_html=True)
    unpaid_ratio = unpaid_ratio_display / 100.0
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="input-card liquid-glass"><div class="card-title">🏢 زانیاری دوکان و پێشینە</div>', unsafe_allow_html=True)
    
    shop_age = st.slider("تەمەنی دوکان (ساڵانی کارکردن)", min_value=0, max_value=50, value=5, step=1)
    st.markdown('<div class="helper-text">چەند ساڵە ئەم دوکانە لە بازاڕدا کار دەکات؟</div>', unsafe_allow_html=True)
    
    days_since_last = st.number_input("چەند ڕۆژ بەسەر کۆتا مامەڵە تێپەڕیوە", min_value=0, value=15, step=1)
    st.markdown('<div class="helper-text">دوایین جار کەی شتی لە کۆگاکەت کڕیوە؟ (ژمارەی ڕۆژەکان)</div>', unsafe_allow_html=True)
    
    debt_ratio_display = st.slider("ڕێژەی قەرز بۆ قەبارەی مامەڵە (%)", min_value=0, max_value=100, value=15, step=1)
    st.markdown('<div class="helper-text">ڕێژەی قەرزەکانی چەندە بەراورد بە کۆی مامەڵەکانی؟ (ئاسایی ژێر 30%)</div>', unsafe_allow_html=True)
    debt_ratio = debt_ratio_display / 100.0
    
    late_history = st.selectbox("پێشینەی دواکەوتنی پارەدان (جار)", options=list(range(31)), index=1, format_func=lambda x: "هیچ کات" if x == 0 else f"{x} جار دواکەوتووە")
    st.markdown('<div class="helper-text">تا ئێستا چەند جار لە کاتی دیاریکراو پارەی نەداوە؟</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYZE BUTTON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze = st.button("🔮  شیکردنەوە و بڕیاردان", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if analyze:
    st.divider()
    st.markdown("""
    <div class="sec-head">
        <span class="sec-head-text">📊  ئەنجامی شیکاری کۆگا</span>
        <span class="sec-head-line"></span>
    </div>""", unsafe_allow_html=True)

    if models_loaded:
        try:
            clf_features = np.array([[shop_age, days_since_last, freq_per_month, avg_invoice, total_volume, unpaid_ratio, debt_ratio, late_history]])
            fs_clf = scaler_clf.transform(clf_features)
            
            risk_pred = risk_model.predict(fs_clf)[0]
            is_high = int(risk_pred) == 1
            
            reg_features = np.array([[shop_age, days_since_last, freq_per_month, avg_invoice, total_volume, unpaid_ratio, debt_ratio, late_history, risk_pred]])
            fs_reg = scaler_reg.transform(reg_features)
            
            limit_pred = limit_model.predict(fs_reg)[0]
            credit_limit = float(limit_pred)
            
        except Exception as exc:
            st.error(f"⚠️ هەڵەیەک ڕوویدا لە کاتی هەژمارکردندا: {exc}")
            st.stop()
    else:
        is_high = debt_ratio > 0.4 or unpaid_ratio > 0.3 or late_history > 3
        credit_limit = max(500.0, avg_invoice * freq_per_month * 2)

    rc1, rc2 = st.columns(2, gap="large")
    with rc1:
        if is_high:
            st.markdown("""
            <div class="result-wrap"><div class="result-card liquid-glass rc-high">
                <div class="rc-eyebrow">⚠️  ئاستی مەترسی دوکان</div>
                <div class="rc-value">بەرز</div>
                <div class="rc-en">HIGH RISK</div>
                <span class="rc-badge badge-high">🔴 &nbsp;مەترسیدار</span>
            </div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-wrap"><div class="result-card liquid-glass rc-low">
                <div class="rc-eyebrow">✅  ئاستی مەترسی دوکان</div>
                <div class="rc-value">نزم</div>
                <div class="rc-en">LOW RISK</div>
                <span class="rc-badge badge-low">🟢 &nbsp;باوەڕپێکراو</span>
            </div></div>""", unsafe_allow_html=True)

    with rc2:
        st.markdown(f"""
        <div class="result-wrap" style="animation-delay:0.15s;">
        <div class="result-card liquid-glass rc-limit">
            <div class="rc-eyebrow">💳  سنووری قەرزی گونجاو (پەسەندکراو)</div>
            <div class="rc-value">${credit_limit:,.0f}</div>
            <div class="rc-en">Approved Credit Limit</div>
            <span class="rc-badge badge-limit">✅ &nbsp;پەسەندکراو</span>
        </div></div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer-section">
    &copy; 2026 Erbil Warehouse B2B Credit Limit System.<br>
    Developed by <strong>Umed Jamal Nouri</strong> | Advanced RFM Edition
</div>
""", unsafe_allow_html=True)

# --- END OF FILE app.py ---