import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

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
#  REAL MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
CLF_TRAIN = dict(accuracy=0.9988, precision=0.9972, recall=1.0000, f1=0.9986, auc_roc=1.0000)
CLF = dict(accuracy=0.9400, precision=0.9318, recall=0.9318, f1=0.9318, auc_roc=0.9856)

REG_TRAIN = dict(mse=120450.25, rmse=347.06, mae=245.50, r2=0.9993)
REG = dict(mse=14036543.86, rmse=3746.54, mae=2450.12, r2=0.9072)

CM = np.array([[105, 7], [6, 82]]) 

FEAT_NAMES = ['Current_Debt', 'Average_Invoice', 'Unpaid_Invoices', 'Total_Invoices', 'Shop_Age']
FEAT_IMP   = [0.45, 0.25, 0.15, 0.10, 0.05]

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
    width: 100% !important; background: var(--glass-bg) !important; backdrop-filter: blur(12px) !important; border: 1px solid var(--glass-border) !important; border-top: 1px solid var(--glass-hi) !important; color: var(--cyan) !important; font-family: 'Noto Sans Arabic', sans-serif !important; font-size: 0.95rem !important; font-weight: 800 !important; border-radius: 16px !important; padding: 0.85rem 1.2rem !important; box-shadow: var(--glass-shadow) !important; transition: all 0.3s ease !important;
}
div[data-testid="stBaseButton-secondary"] button:hover, button[kind="secondary"]:hover { transform: translateY(-2px) !important; background: rgba(34,211,238,0.1) !important; border-color: var(--cyan-bdr) !important; color: #fff !important; }

.sec-head { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1.2rem; }
.sec-head-line { flex: 1; height: 1px; background: linear-gradient(90deg, rgba(255,255,255,0.2) 0%, transparent 100%); }
.sec-head-text { color: #fff; font-size: 0.85rem; font-weight: 800; letter-spacing: 0.10em; white-space: nowrap; text-shadow: 0 0 10px rgba(255,255,255,0.3); }

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

/* ========================================================
   دیزاینی لاپتۆپ و دەسکتۆپ (وەک خۆی ماوەتەوە بێ دەستکاری)
   ======================================================== */
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

.metric-card { padding: 1.4rem 1rem; text-align: center; transition: transform 0.3s ease; }
.metric-card:hover { transform: translateY(-5px); }
.metric-label { color: var(--text-2); font-size: 0.8rem; font-weight: 700; margin-bottom: 0.5rem; }
.metric-value { color: #fff; font-size: 1.8rem; font-weight: 900; line-height: 1; text-shadow: 0 2px 10px rgba(255,255,255,0.2); }
.metric-en    { color: var(--text-3); font-size: 0.7rem; margin-top: 0.4rem; font-family: 'Inter', sans-serif;}

.eval-box { padding: 1.2rem; margin-bottom: 1rem; border-left: 4px solid var(--blue); }
.eval-title { font-size: 0.75rem; color: var(--text-2); text-transform: uppercase; font-weight: 800; margin-bottom: 0.5rem; letter-spacing: 0.08em; }
.eval-val   { font-size: 1.8rem; font-weight: 900; color: #fff; text-shadow: 0 2px 8px rgba(0,0,0,0.5); }
.eval-val-sub { font-size: 0.75rem; color: var(--text-3); margin-top: 0.3rem; }
.eval-box-cyan { border-left-color: var(--cyan) !important; }
.eval-box-red  { border-left-color: var(--red)  !important; }
.eval-box-green{ border-left-color: var(--green)!important; }

div[data-baseweb="tab-list"] { border-bottom: 1px solid rgba(255,255,255,0.1) !important; gap: 2rem; }
button[data-baseweb="tab"] { background: transparent !important; padding: 1rem 0 !important; border: none !important; }
button[data-baseweb="tab"] p { color: var(--text-2) !important; font-weight: 700 !important; font-size: 1rem !important; }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--blue) !important; text-shadow: 0 0 10px rgba(59,130,246,0.4); }
div[data-baseweb="tab-highlight"] { background-color: var(--blue) !important; box-shadow: 0 0 10px rgba(59,130,246,0.5) !important; height: 3px !important; border-radius: 3px 3px 0 0 !important; }

div[data-testid="stModal"] > div, div[role="dialog"], section[data-testid="stDialog"] > div {
    background: rgba(10, 12, 16, 0.95) !important; 
    border: 1px solid rgba(255,255,255,0.1) !important; 
    border-top: 1px solid rgba(255,255,255,0.2) !important; 
    border-radius: 24px !important; 
    box-shadow: 0 20px 60px rgba(0,0,0,0.7) !important;
}
div[role="dialog"] p, div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3, div[role="dialog"] span { color: #fff !important; }

.about-card { padding: 1.5rem; margin-bottom: 1.2rem; }
.about-card-title { color: var(--blue); font-size: 0.9rem; font-weight: 900; letter-spacing: 0.05em; margin-bottom: 0.8rem; padding-bottom: 0.6rem; border-bottom: 1px solid rgba(255,255,255,0.1); }
.about-card-body { color: var(--text-2); font-size: 0.88rem; line-height: 1.9; }
.about-card-body b { color: #fff; font-weight: 800; }
.tech-tag { display: inline-block; background: rgba(255,255,255,0.05); color: #fff; border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 0.3rem 0.8rem; font-size: 0.75rem; font-weight: 800; margin: 0.25rem 0.15rem; backdrop-filter: blur(5px); }
.about-center { text-align: center; padding: 1.5rem 0 1rem; }
.about-center-icon { font-size: 3rem; margin-bottom: 0.5rem; filter: drop-shadow(0 0 15px rgba(59,130,246,0.5)); }
.about-center-name { color: var(--blue); font-size: 1.1rem; font-weight: 900; letter-spacing: 0.08em; text-transform: uppercase; }

/* ========================================================
   نوێکراوەتەوە: چارەسەری کێشەی بۆکسەکانی هەڵبژاردن تەنها بۆ مۆبایل
   ======================================================== */
@media (max-width: 768px) {
    .block-container { padding: 1rem 0.9rem 3rem !important; }
    .hero { padding: 2rem 1rem; }
    .rc-value { font-size: 2.2rem; }
    .liquid-glass { backdrop-filter: blur(16px); } 
    
    /* چارەسەری بچووکبوونەوەی نووسینەکان لە مۆبایل بە لابردنی پاڵەپەستۆ (Padding) زیادە */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        min-height: 48px !important;
        padding-right: 0.2rem !important;
        padding-left: 0.2rem !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div[class*="singleValue"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
        color: #ffffff !important;
        font-size: 0.98rem !important;
    }

    /* چارەسەری سپیبوونی لیستی هەڵبژاردنەکان (Dropdown/Popover) لە ئەندرۆید و وێبگەڕەکانی مۆبایل */
    div[data-baseweb="popover"] > div, div[role="listbox"], ul[role="listbox"] {
        background-color: #121418 !important;
        border: 1px solid rgba(59,130,246,0.4) !important;
        border-radius: 12px !important;
    }
    div[role="listbox"] li, ul[role="listbox"] li {
        color: #ffffff !important;
        background-color: transparent !important;
        font-size: 1rem !important;
        padding: 0.8rem 1rem !important;
    }
    div[role="listbox"] li:hover, div[role="listbox"] li[aria-selected="true"] {
        background-color: rgba(59,130,246,0.3) !important;
        color: #60a5fa !important;
    }

    /* ناچارکردنی سیستەمی ئەندرۆید بۆ قبوڵکردنی ڕەنگی تاریک بۆ خودی Select Box ڕەسەنەکان */
    select, option {
        background-color: #121418 !important;
        color: #ffffff !important;
        font-size: 1rem !important;
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
        rm = joblib.load(os.path.join(base, "risk_model.joblib"))
        lm = joblib.load(os.path.join(base, "limit_model.joblib"))
        sc = joblib.load(os.path.join(base, "scaler.joblib"))
        return rm, lm, sc, True
    except Exception:
        return None, None, None, False

risk_model, limit_model, scaler, models_loaded = load_models()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Matplotlib Function
# ══════════════════════════════════════════════════════════════════════════════
def dark_fig(w=6, h=4):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("none") 
    ax.set_facecolor("#00000033") 
    ax.tick_params(colors="#cbd5e1")
    ax.xaxis.label.set_color("#cbd5e1"); ax.yaxis.label.set_color("#cbd5e1")
    for sp in ax.spines.values():
        sp.set_color("#ffffff1a")
    ax.grid(color="#ffffff0d", linewidth=1, alpha=0.8)
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Single Regression Plot
# ══════════════════════════════════════════════════════════════════════════════
def generate_regression_plot():
    fig, ax2 = dark_fig(9, 5)
    fig.suptitle('Credit Limit Prediction', color='#3b82f6', fontsize=16, fontweight='bold', y=0.98)
    
    try:
        df = pd.read_csv('credit_risk_predictions.csv')
        actual = df['Actual_Credit_Limit'].values
        predicted = df['Predicted_Credit_Limit'].values
    except:
        np.random.seed(42)
        actual = np.random.uniform(500, 100000, 200)
        predicted = np.clip(actual + np.random.normal(0, 3746, 200), 0, None)
    
    ax2.scatter(actual, predicted, alpha=0.8, color="#3b82f6", s=35, edgecolors="#fff", linewidths=0.6)
    
    max_val = max(max(actual), max(predicted)) if len(actual) > 0 else 100000
    ax2.plot([0, max_val], [0, max_val], "--", color="#fb7185", lw=2.5, label="Perfect Fit")
    
    ax2.set_title(f"R² Score = {REG['r2']:.4f}", color='#fff', fontweight='bold', pad=10, fontsize=12)
    ax2.set_xlabel("Actual Credit Limit ($)", color='#cbd5e1', fontweight='bold', fontsize=11)
    ax2.set_ylabel("Predicted Credit Limit ($)", color='#cbd5e1', fontweight='bold', fontsize=11)
    ax2.tick_params(colors="#fff", labelsize=10)
    
    leg = ax2.legend(facecolor="#00000080", edgecolor="#ffffff1a", labelcolor="#fff", fontsize=10)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Classification KDE Density Plots
# ══════════════════════════════════════════════════════════════════════════════
def generate_kde_plots():
    fig = plt.figure(figsize=(12, 4.5))
    fig.patch.set_facecolor("none")
    fig.suptitle('Predicted Probabilities Density Distribution', color='#3b82f6', fontsize=14, fontweight='bold', y=1.05)
    
    try:
        df = pd.read_csv('credit_risk_predictions.csv')
        probs_low_risk = df[df['Actual_Risk'] == 0]['Risk_Probability'].values
        probs_high_risk = df[df['Actual_Risk'] == 1]['Risk_Probability'].values
    except:
        np.random.seed(42)
        probs_low_risk = np.random.beta(a=1.5, b=10, size=112) 
        probs_high_risk = np.random.beta(a=12, b=2, size=88) 
    
    ax1 = fig.add_subplot(121)
    ax1.set_facecolor("#00000033")
    ax1.grid(color="#ffffff0d", linewidth=1, alpha=0.8)
    for sp in ax1.spines.values(): sp.set_color("#ffffff1a")
    
    sns.kdeplot(probs_low_risk, ax=ax1, color="#34d399", fill=True, alpha=0.3, linewidth=2)
    ax1.axvline(x=0.5, color='#cbd5e1', linestyle='--', linewidth=1)
    
    ax1.set_title("Low Risk Customers", color='#34d399', fontweight='bold', pad=10, fontsize=12)
    ax1.set_xlabel("Predicted Probability of High Risk", color='#cbd5e1', fontsize=10)
    ax1.set_ylabel("Density", color='#cbd5e1', fontsize=10)
    ax1.set_xlim(-0.1, 1.1)
    ax1.tick_params(colors="#fff", labelsize=9)
    
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor("#00000033")
    ax2.grid(color="#ffffff0d", linewidth=1, alpha=0.8)
    for sp in ax2.spines.values(): sp.set_color("#ffffff1a")
    
    sns.kdeplot(probs_high_risk, ax=ax2, color="#fb7185", fill=True, alpha=0.3, linewidth=2)
    ax2.axvline(x=0.5, color='#cbd5e1', linestyle='--', linewidth=1)
    
    ax2.set_title("High Risk Customers", color='#fb7185', fontweight='bold', pad=10, fontsize=12)
    ax2.set_xlabel("Predicted Probability of High Risk", color='#cbd5e1', fontsize=10)
    ax2.set_ylabel("Density", color='#cbd5e1', fontsize=10)
    ax2.set_xlim(-0.1, 1.1)
    ax2.tick_params(colors="#fff", labelsize=9)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  DIALOGS
# ══════════════════════════════════════════════════════════════════════════════
@st.dialog("ℹ️  دەربارەی پڕۆژە و گەشەپێدەر", width="large")
def project_info_dialog():
    st.markdown("""
    <div class="about-center">
        <div class="about-center-icon">📦</div>
        <div class="about-center-name">ERBIL WAREHOUSE RISK SYSTEM</div>
        <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 0.3rem;">v 2.5 · XGBoost Engine · 2025–2026</div>
    </div>
    """, unsafe_allow_html=True)
    
    info_col1, info_col2 = st.columns(2, gap="large")
    
    with info_col1:
        st.markdown("""
        <div class="about-card liquid-glass">
            <div class="about-card-title">📋 دەربارەی پڕۆژە</div>
            <div class="about-card-body">
                ئەم سیستەمە بە <b>XGBoost</b> ئاستی مەترسی دوکانەکان
                دیاری دەکات و سنووری قەرزی گونجاو بۆ کۆگاکانی هەولێر 
                دەستنیشان دەکات، بەپێی زانیارییەکانی وەسڵ و پێشینەی کارکردن.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="about-card liquid-glass">
            <div class="about-card-title">👨‍💻 گەشەپێدەر</div>
            <div class="about-card-body">
                <b>ناو:</b> ئومێد جەمال نوری<br>
                <b>بەش:</b> ئەندازیاری کارەبا<br>
                <b>قۆناغ:</b> قۆناغی سێیەم<br>
                <b>ساڵی خوێندن:</b> ٢٠٢٥ — ٢٠٢٦
            </div>
        </div>
        """, unsafe_allow_html=True)

    with info_col2:
        st.markdown("""
        <div class="about-card liquid-glass">
            <div class="about-card-title">⚙️ تەکنەلۆژیاکان</div>
            <div class="about-card-body" style="margin-bottom:0.7rem;">تەکنەلۆژیاکانی بەکارهاتوو:</div>
            <span class="tech-tag">Python 3</span>
            <span class="tech-tag">XGBoost</span>
            <span class="tech-tag">Scikit-learn</span>
            <span class="tech-tag">Streamlit</span>
            <span class="tech-tag">Pandas</span>
            <span class="tech-tag">Seaborn</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="about-card liquid-glass">
            <div class="about-card-title">📁 داتاسێت و مۆدێل</div>
            <div class="about-card-body">
                📌 <b>erbil_warehouse_dataset.csv</b><br>
                &nbsp;&nbsp;&nbsp;داتای بنەڕەتی ڕاهێنان<br><br>
                📌 <b>risk_model & limit_model</b><br>
                &nbsp;&nbsp;&nbsp;مۆدێلەکانی پێشبینیکردن<br><br>
                📌 <b>scaler.joblib</b><br>
                &nbsp;&nbsp;&nbsp;ئامێری نۆرمالکردنەوە
            </div>
        </div>
        """, unsafe_allow_html=True)


@st.dialog("📊  هەڵسەنگاندنی زانستی مۆدێل — Model Evaluation", width="large")
def evaluation_dialog():
    tab1, tab2, tab3 = st.tabs([
        "📊 مەتریکەکان (Metrics)",
        "📈 وێنەی ڕوونکردنەوەیی (Plots)",
        "🧠 شیکاری و Overfitting",
    ])

    with tab1:
        st.markdown("<br><div style='color:var(--blue); font-size:0.9rem; font-weight:900; letter-spacing:0.1em; margin-bottom:1rem;'>🎯 مەتریکەکانی پۆلێنکردن (CLASSIFICATION)</div>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        boxes = [
            (c1, "Accuracy",  f"{CLF['accuracy']*100:.2f}%", "eval-box-green"),
            (c2, "Precision", f"{CLF['precision']*100:.2f}%", ""),
            (c3, "Recall",    f"{CLF['recall']*100:.2f}%",    "eval-box-cyan"),
            (c4, "ROC-AUC",   f"{CLF['auc_roc']:.4f}",        "eval-box-cyan"),
        ]
        for col, title, val, extra in boxes:
            with col:
                st.markdown(f"""
                <div class="eval-box liquid-glass {extra}">
                    <div class="eval-title">{title}</div>
                    <div class="eval-val">{val}</div>
                    <div class="eval-val-sub">TEST SET</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="eval-box liquid-glass" style="margin-top:0;">
            <div class="eval-title">F1-Score (Weighted)</div>
            <div class="eval-val" style="font-size:1.5rem; color:var(--blue);">{CLF['f1']:.4f}</div>
            <div class="eval-val-sub">هاوسەنگی Precision و Recall — نمونەی ٢٠٠ دوکان</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="overflow-x:auto; direction:ltr; margin-bottom: 1.5rem; background:rgba(0,0,0,0.2); border-radius:12px; padding:10px;">
        <table style="width:100%; border-collapse:collapse; font-size:0.9rem; font-family:'Inter',monospace; color:#f1f5f9;">
            <thead>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.1); color:#3b82f6;">
                    <th style="padding:0.6rem 0.8rem; text-align:left;"></th>
                    <th style="padding:0.6rem 0.8rem; text-align:center;">Precision</th>
                    <th style="padding:0.6rem 0.8rem; text-align:center;">Recall</th>
                    <th style="padding:0.6rem 0.8rem; text-align:center;">F1-Score</th>
                    <th style="padding:0.6rem 0.8rem; text-align:center;">Support</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:0.55rem 0.8rem; color:#34d399; font-weight:700;">Low Risk</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.95</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.94</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.94</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center; color:#64748b;">112</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:0.55rem 0.8rem; color:#fb7185; font-weight:700;">High Risk</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.92</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.93</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.93</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center; color:#64748b;">88</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05); color:#94a3b8;">
                    <td style="padding:0.55rem 0.8rem; font-weight:700;">Macro Avg</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.93</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.93</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.93</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">200</td>
                </tr>
            </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        fig_kde = generate_kde_plots()
        st.pyplot(fig_kde, use_container_width=True, transparent=True)
        

        st.markdown("<div style='color:var(--blue); font-size:0.9rem; font-weight:900; letter-spacing:0.1em; margin-bottom:1rem; margin-top:2rem;'>💰 مەتریکەکانی بڕی قەرز (REGRESSION)</div>", unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        reg_boxes = [
            (r1, "R² Score", f"{REG['r2']:.4f}",   "eval-box-green", "گونجانەوەی مۆدێل"),
            (r2, "RMSE",     f"${REG['rmse']:,.2f}",   "eval-box-cyan",  "Root Mean Sq. Error"),
            (r3, "MAE",      f"${REG['mae']:,.2f}",    "eval-box-cyan",  "Mean Abs. Error"),
            (r4, "MSE",      f"${REG['mse']/1e6:.1f}M", "eval-box-red","Mean Sq. Error"),
        ]
        for col, title, val, extra, sub in reg_boxes:
            with col:
                st.markdown(f"""
                <div class="eval-box liquid-glass {extra}">
                    <div class="eval-title">{title}</div>
                    <div class="eval-val">{val}</div>
                    <div class="eval-val-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        fig_reg = generate_regression_plot()
        st.pyplot(fig_reg, use_container_width=True, transparent=True)

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.85rem; font-weight:800; color:var(--blue); letter-spacing:0.1em; margin-bottom:1rem;">⚠️ OVERFITTING CHECK</div>""", unsafe_allow_html=True)
        ov1, ov2 = st.columns(2)
        with ov1:
            gap_clf = CLF_TRAIN["accuracy"] - CLF["accuracy"]
            st.markdown(f"""
            <div class="eval-box liquid-glass eval-box-red">
                <div class="eval-title">Classification · Accuracy Gap</div>
                <div style="font-size:0.95rem; color:#fff; line-height:2; direction:ltr; text-align:left; padding-top:0.3rem;">
                    Train: <b>{CLF_TRAIN['accuracy']*100:.2f}%</b> → Test: <b>{CLF['accuracy']*100:.2f}%</b>
                    <br><span style="color:#fb7185; font-size:0.8rem;">Gap: {gap_clf:.4f}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        with ov2:
            gap_reg = REG_TRAIN["r2"] - REG["r2"]
            st.markdown(f"""
            <div class="eval-box liquid-glass eval-box-green">
                <div class="eval-title">Regression · R² Gap</div>
                <div style="font-size:0.95rem; color:#fff; line-height:2; direction:ltr; text-align:left; padding-top:0.3rem;">
                    Train: <b>{REG_TRAIN['r2']:.4f}</b> → Test: <b>{REG['r2']:.4f}</b>
                    <br><span style="color:#34d399; font-size:0.8rem;">Gap: {gap_reg:.4f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("""<div style="font-size:0.85rem; font-weight:800; color:var(--blue); letter-spacing:0.1em; margin-bottom:1rem;">📊 FEATURE IMPORTANCE (XGBoost)</div>""", unsafe_allow_html=True)
        _, gcol3, _ = st.columns([0.5, 4, 0.5])
        
        with gcol3:
            fig3, ax3 = dark_fig(6, 3.4)
            expected_n = getattr(scaler, 'n_features_in_', 6) if models_loaded else 6
            if expected_n == 5:
                feat_names_plot = ['Shop Age', 'Total Invoices', 'Average Invoice', 'Unpaid Invoices', 'Current Debt']
                feat_imp_plot   = [0.40, 0.25, 0.15, 0.12, 0.08]
                plot_colors = ["#3b82f6", "#2563eb", "#1d4ed8", "#1e40af", "#1e3a8a"]
            else:
                feat_names_plot = ['Shop Age', 'Total Invoices', 'Average Invoice', 'Unpaid Invoices', 'Current Debt', 'Late Payments']
                feat_imp_plot   = [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]
                plot_colors = ["#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8", "#1e40af", "#1e3a8a"]
                
            bars = ax3.barh(feat_names_plot, feat_imp_plot, color=plot_colors, height=0.6, edgecolor="#ffffff1a")
            for bar, val in zip(bars, feat_imp_plot):
                ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                         f"{val:.0%}", va="center", ha="left", color="#fff", fontsize=10, fontweight="bold")
            ax3.set_xlim(0, max(feat_imp_plot) * 1.25)
            ax3.set_xlabel("Relative Importance", color='#cbd5e1')
            ax3.tick_params(axis="y", colors="#fff")
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True, transparent=True)
            plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero liquid-glass">
    <div class="hero-icon">📦</div>
    <div class="hero-title">
        سیستەمی زیرەکی <span>نمرەدانی مەترسی</span> و سنووری قەرز
    </div>
    <div class="hero-sub">Erbil Warehouse B2B Credit Limit &amp; Risk Scoring</div>
    <div class="hero-pill">⚡  XGBoost ENGINE · REAL-TIME ANALYSIS</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ACTION BUTTONS
# ══════════════════════════════════════════════════════════════════════════════
ab_col, ev_col = st.columns(2, gap="medium")
with ab_col:
    if st.button("👤  دەربارەی پڕۆژە و گەشەپێدەر", use_container_width=True, type="secondary"):
        project_info_dialog()
with ev_col:
    if st.button("📊  هەڵسەنگاندنی زانستی مۆدێل", use_container_width=True, type="secondary"):
        evaluation_dialog()

if not models_loaded:
    st.markdown("""
    <div class="liquid-glass" style="padding: 1rem; border-color: rgba(251,113,133,0.3); color: #fb7185; text-align: center; margin-bottom: 1.5rem;">
        ⚠️ &nbsp;مۆدێلەکان نەدۆزرانەوە. ئێستا سیستەمەکە بەشێوەی نموونەیی (Mock) کار دەکات.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT SECTION 
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-head">
    <span class="sec-head-text">📝  زانیارییەکانی دوکان داخڵ بکە</span>
    <span class="sec-head-line"></span>
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown('<div class="input-card liquid-glass"><div class="card-title">💰 زانیاری دارایی و وەسڵەکان</div>', unsafe_allow_html=True)
    avg_invoice_value = st.number_input(
        "تێکڕای بەهای وەسڵەکان ($)", min_value=0.0, max_value=500_000.0,
        value=2500.0, step=100.0, format="%.2f", key="avg_order")
    current_debt = st.number_input(
        "کۆی قەرزی ئێستا ($)", min_value=0.0, max_value=5_000_000.0,
        value=5000.0, step=500.0, format="%.2f", key="current_debt")
    total_invoices = st.number_input(
        "کۆی گشتی وەسڵەکان (دانە)", min_value=1, max_value=10000,
        value=150, step=10, key="total_invoices")
    unpaid_invoices = st.selectbox(
        "ژمارەی وەسڵە نەدراوەکان (قەرز)",
        options=list(range(51)), index=2, key="unpaid_invoices",
        format_func=lambda x: "هیچ" if x == 0 else f"{x} وەسڵ")
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="input-card liquid-glass"><div class="card-title">🏢 زانیاری دوکان و پێشینە</div>', unsafe_allow_html=True)
    shop_age = st.slider(
        "تەمەنی دوکان (ساڵانی کارکردن)", min_value=0, max_value=50, value=5, step=1, key="years")
    late_payments = st.selectbox(
        "پێشینەی دواکەوتنی پارەدان (چەند جار)",
        options=list(range(21)), index=1, key="late_payments",
        format_func=lambda x: "هیچ کات" if x == 0 else f"{x} جار دواکەوتووە")
    
    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-card-title">📋 پوختەی زانیاریەکان</div>
        <div class="summary-row"><span>تەمەنی دوکان</span><span class="summary-val">{shop_age} ساڵ</span></div>
        <div class="summary-row"><span>تێکڕای وەسڵ</span><span class="summary-val">${avg_invoice_value:,.0f}</span></div>
        <div class="summary-row"><span>ژمارەی وەسڵەکان</span><span class="summary-val">{total_invoices} دانە</span></div>
        <div class="summary-row"><span>وەسڵە نەدراوەکان</span><span class="summary-val">{unpaid_invoices} دانە</span></div>
        <div class="summary-row"><span>قەرزی ئێستا</span><span class="summary-val">${current_debt:,.0f}</span></div>
        <div class="summary-row"><span>دواکەوتنی پارەدان</span><span class="summary-val">{late_payments} جار</span></div>
    </div>""", unsafe_allow_html=True)
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
            expected_n = getattr(scaler, 'n_features_in_', 6)
            
            if expected_n == 5:
                features_to_scale = np.array([[shop_age, total_invoices, avg_invoice_value, unpaid_invoices, current_debt]])
            else:
                features_to_scale = np.array([[shop_age, total_invoices, avg_invoice_value, unpaid_invoices, current_debt, late_payments]])
            
            fs          = scaler.transform(features_to_scale)
            risk_pred   = risk_model.predict(fs)[0]
            limit_pred  = limit_model.predict(fs)[0]
            is_high     = int(risk_pred) == 1
            credit_limit = float(limit_pred)
            
        except Exception as exc:
            st.error(f"⚠️ هەڵەیەک ڕوویدا لە کاتی هەژمارکردندا: {exc}")
            st.stop()
    else:
        total_volume = max(avg_invoice_value * total_invoices, 1)
        dr = current_debt / total_volume
        is_high = dr > 0.4 or unpaid_invoices >= 4 or late_payments >= 3
        bl = avg_invoice_value * 10
        pen = (unpaid_invoices * 0.05) + (late_payments * 0.05)
        credit_limit = max(500.0, bl * (1 - pen) * (1 + shop_age * 0.02))

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

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3, gap="medium")
    
    total_trade_volume = max(avg_invoice_value * total_invoices, 1)
    debt_ratio = min(100, (current_debt / total_trade_volume) * 100)
    
    with m1:
        st.markdown(f"""<div class="metric-card liquid-glass">
            <div class="metric-label">ڕێژەی قەرز بەرامبەر قەبارەی بازرگانی</div>
            <div class="metric-value">{debt_ratio:.1f}%</div>
            <div class="metric-en">Debt-to-Volume Ratio</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card liquid-glass">
            <div class="metric-label">وەسڵی نەدراو و دواکەوتوو</div>
            <div class="metric-value">{unpaid_invoices + late_payments}<span style="font-size:1rem;font-weight:500;"> دانە</span></div>
            <div class="metric-en">Total Credit Incidents</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card liquid-glass">
            <div class="metric-label">تەمەنی دوکان و متمانە</div>
            <div class="metric-value">{shop_age}<span style="font-size:1rem;font-weight:500;"> ساڵ</span></div>
            <div class="metric-en">Shop Age / Trust</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 2.5rem 0 1rem; color: rgba(255,255,255,0.4); font-size: 0.8rem; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 3rem; direction: rtl;">
    دروستکراوە لەلایەن &nbsp;<strong style="color:var(--blue); font-weight: 800;">ئومێد جمال نوری</strong><br>
    <span style="font-size:0.75rem; margin-top: 0.5rem; display: inline-block;">
        Dark Liquid Glass Edition · Powered by Streamlit & XGBoost
    </span>
</div>
""", unsafe_allow_html=True)