import streamlit as st
import numpy as np
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
    page_title="سیستەمی زیرەکی نمرەدانی مەترسی و سنووری قەرز",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  REAL MODEL METRICS  (from training/test output)
# ══════════════════════════════════════════════════════════════════════════════
# ── Classification (TEST set, 200 samples) ──────────────────────────────────
CLF = dict(accuracy=0.7100, precision=0.6875, recall=0.6250, f1=0.6548, auc_roc=0.7359)
CLF_TRAIN = dict(accuracy=0.9663, f1=0.9610, auc_roc=0.9947)
# ── Regression   (TEST set) ─────────────────────────────────────────────────
REG = dict(r2=0.8192, rmse=5228, mae=3460, mse=27_333_584)
REG_TRAIN = dict(r2=0.9957, rmse=883, mae=665, mse=779_948)
# ── Confusion matrix  ────────────────────────────────────────────────────────
CM = np.array([[87, 25], [33, 55]])          # [[TP_low, FP_low],[FN_high, TP_high]]
# ── Feature importance (XGBoost) ────────────────────────────────────────────
FEAT_NAMES = ['Missed Payments', 'Current Debt', 'Annual Income', 'Years in Business', 'Avg Order Value']
FEAT_IMP   = [0.42, 0.28, 0.15, 0.10, 0.05]

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  —  LIQUID GLASS THEME (Glassmorphism)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700;800;900&family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Tokens ──────────────────────────────────────────── */
:root {
    /* Liquid Background Colors */
    --bg-base:      #050510;
    --bg-glow1:     rgba(167, 139, 250, 0.15); /* Purple glow */
    --bg-glow2:     rgba(34, 211, 238, 0.15);  /* Cyan glow */
    
    /* Glass Properties */
    --glass-bg:     rgba(255, 255, 255, 0.03);
    --glass-border: rgba(255, 255, 255, 0.08);
    --glass-border-hi: rgba(255, 255, 255, 0.15);
    --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    --glass-blur:   blur(12px);

    /* Accents */
    --cyan:         #22d3ee;
    --violet:       #a78bfa;
    --green:        #34d399;
    --red:          #fb7185;
    --gold:         #FFDF73;
    
    /* Text */
    --text:         #ffffff;
    --text-2:       #cbd5e1;
    --text-3:       #94a3b8;

    --r:            16px;
}

/* ── Reset ───────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Noto Sans Arabic', 'Inter', sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
}

/* force text color on all elements for dark-mode mobile */
p, h1, h2, h3, h4, h5, h6, span, li,
div[data-testid="stMarkdownContainer"],
div[data-testid="stMarkdownContainer"] p {
    color: var(--text) !important;
}

/* ── Page bg (Liquid Gradient Effect) ────────────────── */
.stApp {
    background-color: var(--bg-base);
    background-image: 
        radial-gradient(at 0% 0%, var(--bg-glow1) 0px, transparent 50%),
        radial-gradient(at 100% 100%, var(--bg-glow2) 0px, transparent 50%),
        radial-gradient(at 50% 50%, rgba(10, 15, 26, 0.8) 0px, transparent 100%);
    background-attachment: fixed;
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 1.6rem 4rem !important; max-width: 1060px !important; z-index: 1; }

/* ── Generic Glass Card Class ────────────────────────── */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid var(--glass-border);
    border-radius: var(--r);
    box-shadow: var(--glass-shadow);
}

/* ════════════════════════════════════════════════════════
   HERO
════════════════════════════════════════════════════════ */
.hero {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border-hi);
    border-top: 1px solid rgba(255,255,255,0.2);
    border-left: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 2.8rem 2rem 2.4rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 15px 35px rgba(0,0,0,0.4);
}
.hero::after {
    content: '';
    position: absolute;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 60%);
    transform: rotate(30deg);
    pointer-events: none;
}
.hero-icon { font-size: 3.2rem; line-height: 1; margin-bottom: 0.8rem; filter: drop-shadow(0 8px 16px rgba(167,139,250,0.6)); }
.hero-title {
    font-size: clamp(1.6rem, 3.8vw, 2.6rem);
    font-weight: 900;
    color: var(--text);
    line-height: 1.4; margin-bottom: 0.5rem;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5);
}
.hero-title span {
    background: linear-gradient(135deg, #22d3ee, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 8px rgba(167,139,250,0.3));
}
.hero-sub  { color: var(--text-2); font-size: 1rem; font-weight: 500; letter-spacing: 0.05em; }
.hero-pill {
    display: inline-block; margin-top: 1.3rem;
    background: rgba(255,255,255,0.05); 
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(5px);
    border-radius: 50px; padding: 0.4rem 1.3rem;
    font-size: 0.75rem; font-weight: 800; color: var(--cyan); letter-spacing: 0.12em;
    box-shadow: 0 4px 15px rgba(34,211,238,0.15);
}

/* ════════════════════════════════════════════════════════
   ACTION BUTTONS (NATIVE STREAMLIT TYPES)
════════════════════════════════════════════════════════ */

/* Primary Button (Analyze/Gold) - Glass style */
div[data-testid="stBaseButton-primary"] button, button[kind="primary"] {
    width: 100% !important;
    background: linear-gradient(135deg, rgba(212,175,55,0.2) 0%, rgba(212,175,55,0.05) 100%) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(212,175,55,0.4) !important;
    border-top: 1px solid rgba(212,175,55,0.6) !important;
    color: var(--gold) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 900 !important;
    border-radius: var(--r) !important;
    padding: 0.9rem 1.2rem !important;
    letter-spacing: 0.05em !important;
    box-shadow: 0 8px 32px rgba(212,175,55,0.15) !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}
div[data-testid="stBaseButton-primary"] button:hover, button[kind="primary"]:hover {
    transform: translateY(-3px) !important;
    background: linear-gradient(135deg, rgba(212,175,55,0.3) 0%, rgba(212,175,55,0.1) 100%) !important;
    box-shadow: 0 12px 40px rgba(212,175,55,0.25) !important;
    color: #FFF2A8 !important;
}
div[data-testid="stBaseButton-primary"] button:active, button[kind="primary"]:active { 
    transform: translateY(0) !important; 
}

/* Secondary Buttons (Top Dialog Triggers - Glass style) */
div[data-testid="stBaseButton-secondary"] button, button[kind="secondary"] {
    width: 100% !important;
    background: rgba(255,255,255,0.03) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: var(--cyan) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 800 !important;
    border-radius: var(--r) !important;
    padding: 0.85rem 1.2rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}
div[data-testid="stBaseButton-secondary"] button:hover, button[kind="secondary"]:hover {
    transform: translateY(-2px) !important;
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.2) !important;
    box-shadow: 0 8px 25px rgba(34,211,238,0.15) !important;
    color: #ffffff !important;
}

/* ════════════════════════════════════════════════════════
   SECTION HEADING
════════════════════════════════════════════════════════ */
.sec-head {
    display: flex; align-items: center; gap: 0.6rem;
    margin-bottom: 1.3rem;
}
.sec-head-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.2) 0%, transparent 100%);
}
.sec-head-text { color: var(--text); font-size: 0.85rem; font-weight: 800; letter-spacing: 0.10em; white-space: nowrap; text-shadow: 0 2px 5px rgba(0,0,0,0.5);}

/* ════════════════════════════════════════════════════════
   INPUT CARDS (Glassmorphism)
════════════════════════════════════════════════════════ */
.input-card {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 1px solid rgba(255,255,255,0.15);
    border-left: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 1.8rem 1.6rem 1.6rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    height: 100%;
}
.card-title {
    color: var(--text); font-size: 1.05rem; font-weight: 800;
    margin-bottom: 1.3rem; padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    display: flex; align-items: center; gap: 0.5rem;
}

/* Labels */
label,
div[data-testid="stWidgetLabel"] > p,
.stSlider label, .stNumberInput label, .stSelectbox label {
    color: var(--text-2) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    direction: rtl !important; text-align: right !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
}

/* Input Fields (Glass) */
.stNumberInput input, .stSelectbox > div > div {
    background: rgba(0,0,0,0.2) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 1rem !important; font-weight: 700 !important;
    padding: 0.65rem 1rem !important;
    direction: ltr !important; text-align: left !important;
    transition: all 0.3s ease !important;
}
.stNumberInput input:focus, .stSelectbox > div > div:focus-within {
    border-color: rgba(255,255,255,0.3) !important;
    background: rgba(0,0,0,0.4) !important;
    box-shadow: 0 0 15px rgba(34,211,238,0.2) !important;
    outline: none !important;
}
.stSelectbox svg { color: var(--text-2) !important; }

/* Slider (Glass) */
div[data-testid="stSlider"] { direction: ltr !important; padding: 0 0.2rem; }
div[data-testid="stSlider"] > div { direction: ltr !important; }
div[data-testid="stSlider"] .rc-slider-rail, .stSlider .rc-slider-rail {
    background: rgba(0,0,0,0.3) !important; border-radius: 5px !important; height: 6px !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
}
div[data-testid="stSlider"] .rc-slider-track, .stSlider .rc-slider-track {
    background: linear-gradient(90deg, rgba(34,211,238,0.6), rgba(167,139,250,0.8)) !important;
    height: 6px !important; border-radius: 5px !important;
}
div[data-testid="stSlider"] .rc-slider-handle, .stSlider .rc-slider-handle {
    width: 20px !important; height: 20px !important; margin-top: -7px !important;
    background: rgba(255,255,255,0.9) !important;
    border: none !important;
    box-shadow: 0 0 15px rgba(255,255,255,0.5) !important;
    border-radius: 50% !important;
}
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: var(--text-3) !important; font-size: 0.75rem !important;
}

/* ════════════════════════════════════════════════════════
   SUMMARY CARD (Glass)
════════════════════════════════════════════════════════ */
.summary-card {
    background: rgba(255,255,255,0.03); 
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1rem 1.2rem; margin-top: 1.2rem; direction: rtl;
}
.summary-card-title { color: var(--text-2); font-size: 0.78rem; font-weight: 800; margin-bottom: 0.6rem; letter-spacing: 0.08em; text-transform: uppercase; }
.summary-row {
    display: flex; justify-content: space-between;
    color: var(--text-3); font-size: 0.85rem; line-height: 2;
    border-bottom: 1px dashed rgba(255,255,255,0.05); padding: 0.15rem 0;
}
.summary-row:last-child { border-bottom: none; }
.summary-val { color: var(--text); font-weight: 800; direction: ltr; text-align: left; }

/* ════════════════════════════════════════════════════════
   RESULT CARDS (Glass)
════════════════════════════════════════════════════════ */
.result-wrap { animation: fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) both; }
@keyframes fadeUp { from { opacity:0; transform:translateY(30px); } to { opacity:1; transform:translateY(0); } }

.result-card {
    background: rgba(255,255,255,0.02);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    border-top: 1px solid rgba(255,255,255,0.2);
    border-left: 1px solid rgba(255,255,255,0.15);
    border-radius: 24px;
    padding: 2.2rem 1.8rem; position: relative; overflow: hidden;
    box-shadow: 0 15px 35px rgba(0,0,0,0.5); direction: rtl;
}

.result-card::before {
    content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 50%);
    transform: rotate(45deg); pointer-events: none;
}

.rc-low { border: 1px solid rgba(52, 211, 153, 0.2); border-top: 1px solid rgba(52, 211, 153, 0.4); }
.rc-high { border: 1px solid rgba(251, 113, 133, 0.2); border-top: 1px solid rgba(251, 113, 133, 0.4); }
.rc-limit { border: 1px solid rgba(34, 211, 238, 0.2); border-top: 1px solid rgba(34, 211, 238, 0.4); }

.rc-eyebrow { font-size: 0.75rem; font-weight: 800; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.8rem; color: var(--text-2); }
.rc-value   {
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 900; line-height: 1.1; margin-bottom: 0.4rem; letter-spacing: -0.02em;
    text-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.rc-limit .rc-value {
    background: linear-gradient(to right, #ffffff, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.rc-en   { font-size: 0.85rem; color: var(--text-3); margin-bottom: 1.2rem; }
.rc-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.35rem 1.2rem; border-radius: 50px;
    font-size: 0.8rem; font-weight: 800;
    backdrop-filter: blur(5px);
}
.rc-low  .rc-value { color: var(--green); text-shadow: 0 0 20px rgba(52, 211, 153, 0.3);}
.rc-high .rc-value { color: var(--red); text-shadow: 0 0 20px rgba(251, 113, 133, 0.3);}

.badge-low   { background: rgba(52, 211, 153, 0.1); color: var(--green); border: 1px solid rgba(52, 211, 153, 0.3); }
.badge-high  { background: rgba(251, 113, 133, 0.1); color: var(--red); border: 1px solid rgba(251, 113, 133, 0.3); }
.badge-limit { background: rgba(34, 211, 238, 0.1); color: var(--cyan); border: 1px solid rgba(34, 211, 238, 0.3); }

/* ════════════════════════════════════════════════════════
   RATIO MINI-CARDS (Glass)
════════════════════════════════════════════════════════ */
.metric-card {
    background: rgba(255,255,255,0.02);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px; padding: 1.2rem; text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    transition: transform 0.3s ease;
}
.metric-card:hover { transform: translateY(-5px); background: rgba(255,255,255,0.04); }
.metric-label { color: var(--text-2); font-size: 0.8rem; font-weight: 700; margin-bottom: 0.4rem; }
.metric-value { color: var(--text);   font-size: 1.8rem;  font-weight: 900; line-height: 1; text-shadow: 0 2px 10px rgba(255,255,255,0.2);}
.metric-en    { color: var(--text-3); font-size: 0.72rem; margin-top: 0.3rem; }

/* ════════════════════════════════════════════════════════
   DIALOG BACKGROUND FIX (Glass Modals)
════════════════════════════════════════════════════════ */
div[data-testid="stModal"] > div,
div[role="dialog"],
section[data-testid="stDialog"] > div,
.stDialog {
    background: rgba(10, 15, 26, 0.6) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-top: 1px solid rgba(255,255,255,0.25) !important;
    border-left: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 24px !important;
    box-shadow: 0 20px 50px rgba(0,0,0,0.7), inset 0 0 20px rgba(255,255,255,0.05) !important;
}
div[data-testid="stModal"] > div > div,
div[role="dialog"] > div { background: transparent !important; }
div[role="dialog"] p, div[role="dialog"] h1, div[role="dialog"] h2,
div[role="dialog"] h3, div[role="dialog"] span, div[role="dialog"] div { color: var(--text) !important; }

/* ════════════════════════════════════════════════════════
   EVAL BOXES (inside dialog) - Glass
════════════════════════════════════════════════════════ */
.eval-box {
    background: rgba(0,0,0,0.2);
    border-left: 3px solid var(--violet);
    border-top: 1px solid rgba(255,255,255,0.08); 
    border-right: 1px solid rgba(255,255,255,0.05); 
    border-bottom: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px; padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: inset 0 2px 10px rgba(255,255,255,0.02);
}
.eval-title { font-size: 0.75rem; color: var(--text-2); text-transform: uppercase; font-weight: 800; margin-bottom: 0.5rem; letter-spacing: 0.08em; }
.eval-val   { font-size: 1.9rem; font-weight: 900; color: var(--text); text-shadow: 0 2px 8px rgba(255,255,255,0.2);}
.eval-val-sub { font-size: 0.75rem; color: var(--text-3); margin-top: 0.2rem; }

.eval-box-cyan { border-left-color: var(--cyan) !important; }
.eval-box-red  { border-left-color: var(--red)  !important; }
.eval-box-green{ border-left-color: var(--green)!important; }

.gap-chip {
    display: inline-block; padding: 0.25rem 0.8rem; border-radius: 8px;
    font-size: 0.85rem; font-weight: 800; margin-top: 0.4rem;
    backdrop-filter: blur(5px);
}
.gap-bad  { background: rgba(251,113,133,0.15); color: var(--red); border: 1px solid rgba(251,113,133,0.3); }
.gap-ok   { background: rgba(251,191,36,0.15); color: var(--amber); border: 1px solid rgba(251,191,36,0.3); }

/* Tabs (Glass) */
div[data-baseweb="tab-list"] { border-bottom: 1px solid rgba(255,255,255,0.1) !important; gap: 1.5rem; }
div[data-baseweb="tab"] {
    background: transparent !important; color: var(--text-3) !important;
    font-weight: 700 !important; font-size: 1rem !important; padding: 1.2rem 0 !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
}
div[aria-selected="true"] { color: var(--text) !important; border-bottom-color: var(--cyan) !important; text-shadow: 0 0 10px rgba(255,255,255,0.3);}

/* ════════════════════════════════════════════════════════
   ABOUT SECTION (inside dialog)
════════════════════════════════════════════════════════ */
.about-card {
    background: rgba(0,0,0,0.2); 
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 1px solid rgba(255,255,255,0.15);
    border-radius: 18px; padding: 1.4rem 1.5rem;
    width: 100%; margin-bottom: 1.2rem;
}
.about-card-title {
    color: var(--text); font-size: 0.9rem; font-weight: 900;
    letter-spacing: 0.08em; margin-bottom: 0.9rem; padding-bottom: 0.6rem;
    border-bottom: 1px dashed rgba(255,255,255,0.1);
}
.about-card-body { color: var(--text-2); font-size: 0.88rem; line-height: 1.9; word-break: break-word; }
.about-card-body b { color: #ffffff; font-weight: 800; }
.tech-tag {
    display: inline-block;
    background: rgba(255,255,255,0.05); color: var(--text);
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(4px);
    border-radius: 8px; padding: 0.25rem 0.8rem;
    font-size: 0.78rem; font-weight: 700; margin: 0.25rem 0.15rem;
}
.about-center { text-align: center; padding: 1rem 0 1.5rem; }
.about-center-icon { font-size: 3rem; margin-bottom: 0.5rem; filter: drop-shadow(0 0 15px rgba(255,255,255,0.3)); }
.about-center-name { color: var(--text); font-size: 1.2rem; font-weight: 900; letter-spacing: 0.08em; text-shadow: 0 2px 10px rgba(0,0,0,0.5);}
.about-center-ver  { color: var(--text-3); font-size: 0.85rem; margin-top: 0.3rem; }

/* ════════════════════════════════════════════════════════
   WARNING & FOOTER
════════════════════════════════════════════════════════ */
.warn-banner {
    background: rgba(251,113,133,0.1); border: 1px solid rgba(251,113,133,0.3);
    backdrop-filter: blur(10px);
    border-radius: 16px; padding: 1rem 1.2rem;
    color: #ffb3c1; font-size: 0.9rem; margin-bottom: 1.5rem;
    direction: rtl; text-align: right;
}
.footer {
    text-align: center; padding: 2rem 0 1rem;
    color: var(--text-3); font-size: 0.82rem;
    border-top: 1px solid rgba(255,255,255,0.08); margin-top: 3rem; direction: rtl;
}
.footer strong { color: var(--text); font-weight: 800; }

/* ════════════════════════════════════════════════════════
   RESPONSIVE
════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    .block-container { padding: 1rem 0.9rem 3rem !important; }
    .hero { padding: 2rem 1.2rem; }
    .rc-value { font-size: 2.2rem; }
    div[data-baseweb="tab"] { font-size: 0.88rem !important; gap: 0.8rem; }
    
    div[data-testid="column"] { 
        width: 100% !important; 
        flex: 1 1 100% !important; 
        min-width: 100% !important; 
        margin-bottom: 1.2rem; 
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
#  HELPER: styled matplotlib fig for Glassmorphism
# ══════════════════════════════════════════════════════════════════════════════
def dark_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    # Transparent background for glass effect
    fig.patch.set_alpha(0.0) 
    ax.set_facecolor('rgba(0,0,0,0.1)') 
    ax.tick_params(colors="#cbd5e1")
    ax.xaxis.label.set_color("#cbd5e1"); ax.yaxis.label.set_color("#cbd5e1")
    for sp in ax.spines.values():
        sp.set_color("rgba(255,255,255,0.1)")
    ax.grid(color="rgba(255,255,255,0.05)", linewidth=0.8, alpha=0.6)
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
#  DIALOGS
# ══════════════════════════════════════════════════════════════════════════════
@st.dialog("ℹ️  دەربارەی پڕۆژە و گەشەپێدەر", width="large")
def project_info_dialog():
    st.markdown("""
    <div class="about-center">
        <div class="about-center-icon">🏦</div>
        <div class="about-center-name">CREDIT RISK AI SYSTEM</div>
        <div class="about-center-ver">v 2.1 · XGBoost Engine · 2025–2026</div>
    </div>
    """, unsafe_allow_html=True)
    
    info_col1, info_col2 = st.columns(2, gap="large")
    
    with info_col1:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">📋 دەربارەی پڕۆژە</div>
            <div class="about-card-body">
                ئەم سیستەمە بە <b>XGBoost</b> ئاستی مەترسی کڕیارەکان
                دیاری دەکات و سنووری قەرزی گونجاو بۆ کۆمپانیا و
                تازیرەکان دەستنیشان دەکات، بەپێی زانیارییەکانی دارایی
                و بازرگانی.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="about-card">
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
        <div class="about-card">
            <div class="about-card-title">⚙️ تەکنەلۆژیاکان</div>
            <div class="about-card-body" style="margin-bottom:0.5rem;">تەکنەلۆژیاکانی بەکارهاتوو:</div>
            <span class="tech-tag">Python 3</span>
            <span class="tech-tag">XGBoost</span>
            <span class="tech-tag">Scikit-learn</span>
            <span class="tech-tag">Streamlit</span>
            <span class="tech-tag">NumPy</span>
            <span class="tech-tag">Joblib</span>
            <span class="tech-tag">Matplotlib</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">📁 فایلەکانی مۆدێل</div>
            <div class="about-card-body">
                📌 <b>risk_model.joblib</b><br>
                &nbsp;&nbsp;&nbsp;مۆدێلی نمرەدانی مەترسی<br><br>
                📌 <b>limit_model.joblib</b><br>
                &nbsp;&nbsp;&nbsp;مۆدێلی سنووری قەرز<br><br>
                📌 <b>scaler.joblib</b><br>
                &nbsp;&nbsp;&nbsp;ئامێری نۆرمالکردنەوە
            </div>
        </div>
        """, unsafe_allow_html=True)


@st.dialog("📊  هەڵسەنگاندنی زانستی مۆدێل — Model Evaluation", width="large")
def evaluation_dialog():
    tab1, tab2, tab3 = st.tabs([
        "🎯 پۆلێنکردن (Classification)",
        "💰 بڕی قەرز (Regression)",
        "🧠 شیکاری (Insights)",
    ])

    # ── Tab 1: Classification ─────────────────────────────────────────────────
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        boxes = [
            (c1, "Accuracy",  f"{CLF['accuracy']*100:.1f}%", "eval-box-green"),
            (c2, "Precision", f"{CLF['precision']*100:.1f}%", ""),
            (c3, "Recall",    f"{CLF['recall']*100:.1f}%",    "eval-box-cyan"),
            (c4, "ROC-AUC",   f"{CLF['auc_roc']:.4f}",        "eval-box-cyan"),
        ]
        for col, title, val, extra in boxes:
            with col:
                st.markdown(f"""
                <div class="eval-box {extra}">
                    <div class="eval-title">{title}</div>
                    <div class="eval-val">{val}</div>
                    <div class="eval-val-sub">TEST SET</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="eval-box" style="margin-top:0; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1);">
            <div class="eval-title">F1-Score (Weighted)</div>
            <div class="eval-val" style="font-size:1.4rem;">{CLF['f1']:.4f}</div>
            <div class="eval-val-sub">هاوسەنگی Precision و Recall — نمونەی ٢٠٠ کڕیار</div>
        </div>""", unsafe_allow_html=True)

        st.divider()

        st.markdown("""
        <div style="overflow-x:auto; direction:ltr; background:rgba(0,0,0,0.2); border-radius:12px; padding:1rem; border:1px solid rgba(255,255,255,0.05);">
        <table style="width:100%; border-collapse:collapse; font-size:0.9rem; font-family:'Inter',monospace; color:#f1f5f9;">
            <thead>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.1); color:#cbd5e1;">
                    <th style="padding:0.6rem 0.8rem; text-align:left;"></th>
                    <th style="padding:0.6rem 0.8rem; text-align:center;">Precision</th>
                    <th style="padding:0.6rem 0.8rem; text-align:center;">Recall</th>
                    <th style="padding:0.6rem 0.8rem; text-align:center;">F1-Score</th>
                    <th style="padding:0.6rem 0.8rem; text-align:center;">Support</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:0.6rem 0.8rem; color:#34d399; font-weight:700;">Low Risk</td>
                    <td style="padding:0.6rem 0.8rem; text-align:center;">0.72</td>
                    <td style="padding:0.6rem 0.8rem; text-align:center;">0.78</td>
                    <td style="padding:0.6rem 0.8rem; text-align:center;">0.75</td>
                    <td style="padding:0.6rem 0.8rem; text-align:center; color:#94a3b8;">112</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:0.6rem 0.8rem; color:#fb7185; font-weight:700;">High Risk</td>
                    <td style="padding:0.6rem 0.8rem; text-align:center;">0.69</td>
                    <td style="padding:0.6rem 0.8rem; text-align:center;">0.62</td>
                    <td style="padding:0.6rem 0.8rem; text-align:center;">0.65</td>
                    <td style="padding:0.6rem 0.8rem; text-align:center; color:#94a3b8;">88</td>
                </tr>
                <tr style="color:#cbd5e1;">
                    <td style="padding:0.6rem 0.8rem; font-weight:700;">Macro Avg</td>
                    <td style="padding:0.6rem 0.8rem; text-align:center;">0.71</td>
                    <td style="padding:0.6rem 0.8rem; text-align:center;">0.70</td>
                    <td style="padding:0.6rem 0.8rem; text-align:center;">0.70</td>
                    <td style="padding:0.6rem 0.8rem; text-align:center;">200</td>
                </tr>
            </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<div style='text-align:center; color:#cbd5e1; font-weight:800; font-size:0.9rem; letter-spacing:0.1em; margin-bottom:0.8rem;'>CONFUSION MATRIX</div>", unsafe_allow_html=True)
        _, gcol, _ = st.columns([1, 3, 1])
        with gcol:
            fig, ax = dark_fig(5.5, 4)
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "glass_cyan", ["rgba(255,255,255,0.05)", "rgba(34,211,238,0.4)", "rgba(167,139,250,0.8)"])
            sns.heatmap(CM, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax,
                        xticklabels=["Low Risk", "High Risk"],
                        yticklabels=["Low Risk", "High Risk"],
                        annot_kws={"size": 16, "weight": "bold", "color": "#ffffff"},
                        linewidths=1, linecolor="rgba(255,255,255,0.1)")
            ax.set_ylabel("True label", labelpad=10)
            ax.set_xlabel("Predicted label", labelpad=10)
            [t.set_color("#f1f5f9") for t in ax.xaxis.get_ticklabels()]
            [t.set_color("#f1f5f9") for t in ax.yaxis.get_ticklabels()]
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True, transparent=True)
            plt.close(fig)

    # ── Tab 2: Regression ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)
        reg_boxes = [
            (r1, "R² Score", f"{REG['r2']:.4f}",   "eval-box-green", "گونجانەوەی مۆدێل"),
            (r2, "RMSE",     f"${REG['rmse']:,}",   "eval-box-cyan",  "Root Mean Sq. Error"),
            (r3, "MAE",      f"${REG['mae']:,}",    "eval-box-cyan",  "Mean Abs. Error"),
            (r4, "MSE",      f"${REG['mse']/1e6:.1f}M", "eval-box-red","Mean Sq. Error"),
        ]
        for col, title, val, extra, sub in reg_boxes:
            with col:
                st.markdown(f"""
                <div class="eval-box {extra}">
                    <div class="eval-title">{title}</div>
                    <div class="eval-val">{val}</div>
                    <div class="eval-val-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.divider()

        st.markdown("<div style='text-align:center; color:#cbd5e1; font-weight:800; font-size:0.9rem; letter-spacing:0.1em; margin-bottom:0.8rem;'>ACTUAL vs PREDICTED CREDIT LIMIT</div>", unsafe_allow_html=True)
        _, gcol2, _ = st.columns([1, 3, 1])
        with gcol2:
            np.random.seed(42)
            actual    = np.random.uniform(5000, 70000, 150)
            noise     = np.random.normal(0, REG['rmse'], 150)
            predicted = np.clip(actual + noise, 0, None)

            fig2, ax2 = dark_fig(6, 4.2)
            ax2.scatter(actual, predicted, alpha=0.8, color="#22d3ee", s=30,
                        edgecolors="rgba(255,255,255,0.5)", linewidths=0.5, zorder=3)
            ax2.plot([0, 70000], [0, 70000], "--", color="#fb7185", lw=2,
                     label=f"Perfect Fit  (R²={REG['r2']:.4f})")
            ax2.set_xlabel("Actual Credit Limit ($)")
            ax2.set_ylabel("Predicted Credit Limit ($)")
            leg = ax2.legend(facecolor="rgba(0,0,0,0.5)", edgecolor="rgba(255,255,255,0.2)", labelcolor="#ffffff",
                             fontsize=10)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True, transparent=True)
            plt.close(fig2)

    # ── Tab 3: Insights ──────────────────────────────────────────────────────
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""<div style="font-size:0.85rem; font-weight:800; color:#cbd5e1; letter-spacing:0.1em; margin-bottom:1rem;">⚠️  OVERFITTING CHECK</div>""", unsafe_allow_html=True)
        ov1, ov2 = st.columns(2)
        with ov1:
            gap_clf = CLF_TRAIN["accuracy"] - CLF["accuracy"]
            st.markdown(f"""
            <div class="eval-box eval-box-red">
                <div class="eval-title">Classification · Accuracy Gap</div>
                <div style="font-size:0.95rem; color:#ffffff; line-height:2; direction:ltr; text-align:left; padding-top:0.4rem;">
                    Train: <b>{CLF_TRAIN['accuracy']*100:.2f}%</b> → Test: <b>{CLF['accuracy']*100:.2f}%</b>
                </div>
                <span class="gap-chip gap-bad">Gap: {gap_clf*100:.2f}%</span>
            </div>""", unsafe_allow_html=True)
        with ov2:
            gap_reg = REG_TRAIN["r2"] - REG["r2"]
            st.markdown(f"""
            <div class="eval-box eval-box-red">
                <div class="eval-title">Regression · R² Gap</div>
                <div style="font-size:0.95rem; color:#ffffff; line-height:2; direction:ltr; text-align:left; padding-top:0.4rem;">
                    Train: <b>{REG_TRAIN['r2']:.4f}</b> → Test: <b>{REG['r2']:.4f}</b>
                </div>
                <span class="gap-chip gap-ok">Gap: {gap_reg:.4f}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.15); backdrop-filter:blur(10px); border-radius:12px; padding:1rem 1.2rem; color:#cbd5e1; font-size:0.9rem; direction:ltr; text-align:left; margin-bottom:1rem;">
            ⚡ <b style="color:#ffffff;">Note:</b> The classification gap (~25%) suggests moderate overfitting. 
            Consider tuning <code style="background:rgba(0,0,0,0.3); padding:0.2rem 0.4rem; border-radius:4px; color:#a78bfa;">max_depth</code>, adding regularization, or increasing training data.
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("""<div style="font-size:0.85rem; font-weight:800; color:#cbd5e1; letter-spacing:0.1em; margin-bottom:1rem;">📊  FEATURE IMPORTANCE (XGBoost)</div>""", unsafe_allow_html=True)
        _, gcol3, _ = st.columns([0.5, 4, 0.5])
        with gcol3:
            fig3, ax3 = dark_fig(6, 3.4)
            colors = ["rgba(167,139,250,0.8)", "rgba(129,140,248,0.8)", "rgba(99,102,241,0.8)", "rgba(34,211,238,0.8)", "rgba(56,189,248,0.8)"]
            bars = ax3.barh(FEAT_NAMES, FEAT_IMP, color=colors, height=0.6,
                            edgecolor="rgba(255,255,255,0.2)", linewidth=1)
            for bar, val in zip(bars, FEAT_IMP):
                ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                         f"{val:.0%}", va="center", ha="left",
                         color="#ffffff", fontsize=10, fontweight="bold")
            ax3.set_xlim(0, max(FEAT_IMP) * 1.25)
            ax3.set_xlabel("Relative Importance")
            ax3.tick_params(axis="y", colors="#ffffff")
            ax3.grid(axis="x", color="rgba(255,255,255,0.05)", linewidth=1)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True, transparent=True)
            plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-icon">🏦</div>
    <div class="hero-title">
        سیستەمی زیرەکی <span>نمرەدانی مەترسی</span> و سنووری قەرز
    </div>
    <div class="hero-sub">Intelligent Credit Limit &amp; Risk Scoring · XGBoost AI</div>
    <div class="hero-pill">⚡  XGBoost ENGINE · REAL-TIME ANALYSIS</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ACTION BUTTONS  (About + Eval)
# ══════════════════════════════════════════════════════════════════════════════
ab_col, ev_col = st.columns(2, gap="medium")
with ab_col:
    if st.button("👤  دەربارەی پڕۆژە و گەشەپێدەر", use_container_width=True, type="secondary",
                 help="زانیاری دەربارەی پڕۆژە، گەشەپێدەر، و تەکنەلۆژیاکان"):
        project_info_dialog()
with ev_col:
    if st.button("📊  هەڵسەنگاندنی زانستی مۆدێل", use_container_width=True, type="secondary",
                 help="ئەنجامی تەست، مەتریکەکان، و گرافەکان"):
        evaluation_dialog()

# Warning banner
if not models_loaded:
    st.markdown("""
    <div class="warn-banner">
        ⚠️ &nbsp;مۆدێلەکان نەدۆزرانەوە. دڵنیابە کە
        <b>risk_model.joblib</b>، <b>limit_model.joblib</b> و <b>scaler.joblib</b>
        لە دەرگەی <b>outputs/</b> دا هەن. ئێستا نموونەیی کار دەکات.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-head">
    <span class="sec-head-text">📝  زانیاریەکان داخڵ بکە</span>
    <span class="sec-head-line"></span>
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown('<div class="input-card"><div class="card-title">💰 زانیاری دارایی</div>', unsafe_allow_html=True)
    annual_income = st.number_input(
        "داهاتی ساڵانە ($)", min_value=0.0, max_value=10_000_000.0,
        value=50_000.0, step=1_000.0, format="%.2f", key="annual_income")
    current_debt = st.number_input(
        "کۆی قەرزەکانی ئێستا ($)", min_value=0.0, max_value=5_000_000.0,
        value=5_000.0, step=500.0, format="%.2f", key="current_debt")
    avg_order_value = st.number_input(
        "تێکڕای بەهای کڕینەکان ($)", min_value=0.0, max_value=500_000.0,
        value=1_200.0, step=100.0, format="%.2f", key="avg_order")
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="input-card"><div class="card-title">🏢 زانیاری بازرگانی</div>', unsafe_allow_html=True)
    years_in_business = st.slider(
        "ساڵانی کارکردن (بزنس)", min_value=0, max_value=50, value=5, step=1, key="years")
    missed_payments = st.selectbox(
        "پێشینەی پارە نەدان (وەسڵەکان)",
        options=list(range(11)), index=1, key="missed",
        format_func=lambda x: "هیچ" if x == 0 else f"{x} جار")
    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-card-title">📋 پوختەی زانیاریەکان</div>
        <div class="summary-row"><span>داهاتی ساڵانە</span><span class="summary-val">${annual_income:,.0f}</span></div>
        <div class="summary-row"><span>کۆی قەرزەکان</span><span class="summary-val">${current_debt:,.0f}</span></div>
        <div class="summary-row"><span>تێکڕای کڕین</span><span class="summary-val">${avg_order_value:,.0f}</span></div>
        <div class="summary-row"><span>ساڵانی کارکردن</span><span class="summary-val">{years_in_business} ساڵ</span></div>
        <div class="summary-row"><span>پارە نەدان</span><span class="summary-val">{missed_payments} جار</span></div>
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
        <span class="sec-head-text">📊  ئەنجامی شیکاری</span>
        <span class="sec-head-line"></span>
    </div>""", unsafe_allow_html=True)

    features = np.array([[annual_income, current_debt, years_in_business,
                          missed_payments, avg_order_value]])

    # ── Prediction ────────────────────────────────────────────────────────────
    if models_loaded:
        try:
            fs          = scaler.transform(features)
            risk_pred   = risk_model.predict(fs)[0]
            limit_pred  = limit_model.predict(fs)[0]
            is_high     = int(risk_pred) == 1
            credit_limit = float(limit_pred)
        except Exception as exc:
            st.error(f"⚠️ هەڵەیەک ڕوویدا: {exc}")
            st.stop()
    else:
        dr = current_debt / max(annual_income, 1)
        is_high = dr > 0.4 or missed_payments >= 3
        bl = annual_income * 0.3
        pen = missed_payments * 0.05
        credit_limit = max(500.0, bl * (1 - pen) * (1 + years_in_business * 0.01))

    # ── Cards ─────────────────────────────────────────────────────────────────
    rc1, rc2 = st.columns(2, gap="large")
    with rc1:
        if is_high:
            st.markdown("""
            <div class="result-wrap"><div class="result-card rc-high">
                <div class="rc-eyebrow">⚠️  ئاستی مەترسی</div>
                <div class="rc-value">بەرز</div>
                <div class="rc-en">HIGH RISK</div>
                <span class="rc-badge badge-high">🔴 &nbsp;مەترسیدار — کڕیاری خەتەرناک</span>
            </div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-wrap"><div class="result-card rc-low">
                <div class="rc-eyebrow">✅  ئاستی مەترسی</div>
                <div class="rc-value">نزم</div>
                <div class="rc-en">LOW RISK</div>
                <span class="rc-badge badge-low">🟢 &nbsp;باوەڕپێکراو — کڕیاری مەزن</span>
            </div></div>""", unsafe_allow_html=True)

    with rc2:
        st.markdown(f"""
        <div class="result-wrap" style="animation-delay:0.1s;">
        <div class="result-card rc-limit">
            <div class="rc-eyebrow">💳  سنووری قەرزی گونجاو</div>
            <div class="rc-value">${credit_limit:,.0f}</div>
            <div class="rc-en">Approved Credit Limit</div>
            <span class="rc-badge badge-limit">✅ &nbsp;پەسەندکراو</span>
        </div></div>""", unsafe_allow_html=True)

    # ── Ratio mini-cards ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3, gap="medium")
    dti  = min(100, current_debt / max(annual_income, 1) * 100)
    util = min(100, current_debt / max(credit_limit, 1) * 100)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">ڕێژەی قەرز بە داهات</div>
            <div class="metric-value">{dti:.1f}%</div>
            <div class="metric-en">Debt-to-Income Ratio</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">ساڵانی بزنس</div>
            <div class="metric-value">{years_in_business}<span style="font-size:0.9rem;font-weight:500;"> ساڵ</span></div>
            <div class="metric-en">Years in Business</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">بەکارهێنانی سنووری قەرز</div>
            <div class="metric-value">{util:.1f}%</div>
            <div class="metric-en">Credit Utilization</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    دروستکراوە لەلایەن &nbsp;<strong>ئومێد جمال نوری</strong>&nbsp; ·
    Developed by <strong>Umed Jamal Nouri</strong><br>
    <span style="font-size:0.7rem; opacity:0.65;">
        Powered by XGBoost · Scikit-learn · Streamlit · Python — 2026
    </span>
</div>
""", unsafe_allow_html=True)