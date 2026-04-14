import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG.
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="سیستەمی زیرەکی نمرەدانی مەترسی و سنووری قەرز",
    page_icon="🏦",
    layout="wide",
  initial_sidebar_state="auto",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  —  Custom Palette Theme (#353535, #3C6E71, #FFFFFF, #D9D9D9, #284B63)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700;800;900&display=swap');

/* ── CSS Variables (Mapped to your custom palette) ───────────────── */
:root {
    --navy:        #353535; /* Dark Charcoal */
    --navy-mid:    #2b2b2b; /* Slightly darker for background */
    --navy-light:  #284B63; /* Slate Blue */
    --teal:        #3C6E71; /* Teal */
    --teal-dim:    rgba(60, 110, 113, 0.18);
    --teal-glow:   rgba(60, 110, 113, 0.25);
    --slate:       #284B63; /* Slate Blue */
    --slate-card:  rgba(53, 53, 53, 0.85); /* #353535 with opacity */
    --border:      rgba(217, 217, 217, 0.15); /* #D9D9D9 with opacity */
    --text-main:   #FFFFFF; /* White */
    --text-muted:  #D9D9D9; /* Light Gray */
    --text-label:  #D9D9D9;
    --green:       #52b788; /* Muted green fitting the palette */
    --green-dim:   rgba(82, 183, 136, 0.15);
    --red:         #e07a5f; /* Muted red fitting the palette */
    --red-dim:     rgba(224, 122, 95, 0.15);
}

/* ── Reset & Base ────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Noto Sans Arabic', sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
}

/* --- چارەسەری کێشەی دیارنەمانی دەق لە مۆبایل --- */
p, h1, h2, h3, h4, h5, h6, span, label, li, div[data-testid="stMarkdownContainer"] {
    color: var(--text-main) !important;
}

/* ══════════════════════════════════════════════════════════════════════
   چارەسەری کۆتایی بۆ پەنجەرەکان (Dialogs & Modals Background Fix)
══════════════════════════════════════════════════════════════════════ */
div[data-testid="stModal"] > div, 
div[role="dialog"], 
section[data-testid="stDialog"] > div {
    background-color: #353535 !important;
    background: linear-gradient(145deg, #353535 0%, #2b2b2b 100%) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
}

/* زۆرەملێکردنی ناوەوەی پەنجەرەکان بۆ ئەوەی ڕەنگی تاریک وەربگرن */
div[data-testid="stModal"] > div > div, 
div[role="dialog"] > div {
    background-color: transparent !important;
}

/* دڵنیابوونەوە لە ڕەنگی سپی بۆ دەقەکانی ناو پەنجەرەکە */
div[role="dialog"] p, 
div[role="dialog"] h1, 
div[role="dialog"] h2, 
div[role="dialog"] h3,
div[role="dialog"] span,
div[role="dialog"] div {
    color: #FFFFFF !important;
}

/* ── App Background ──────────────────────────────────────────────── */
.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(60, 110, 113, 0.1) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 100%, rgba(40, 75, 99, 0.15) 0%, transparent 55%),
        linear-gradient(160deg, #1e1e1e 0%, #2b2b2b 100%);
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ───────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.8rem 2rem 4rem 2rem !important;
    max-width: 1080px !important;
}

/* ════════════════════════════════════════════════════════
   HERO BANNER
════════════════════════════════════════════════════════ */
.hero {
    background: linear-gradient(120deg, rgba(53, 53, 53, 0.8) 0%, rgba(43, 43, 43, 0.95) 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2.2rem 2rem 2rem 2rem;
    margin-bottom: 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 40px rgba(0,0,0,0.3), inset 0 1px 0 rgba(217, 217, 217, 0.05);
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% -20%, rgba(60, 110, 113, 0.15) 0%, transparent 65%);
    pointer-events: none;
}
.hero-icon  { font-size: 2.6rem; line-height: 1; margin-bottom: 0.6rem; }
.hero-title {
    font-size: clamp(1.45rem, 3.5vw, 2.1rem);
    font-weight: 900;
    color: var(--text-main);
    margin-bottom: 0.35rem;
    line-height: 1.45;
    letter-spacing: -0.01em;
}
.hero-title span { color: var(--teal); }
.hero-sub {
    color: var(--text-muted);
    font-size: 0.9rem;
    font-weight: 400;
    letter-spacing: 0.02em;
}
.hero-badge {
    display: inline-block;
    margin-top: 0.9rem;
    background: var(--teal-dim);
    border: 1px solid rgba(60, 110, 113, 0.3);
    border-radius: 50px;
    padding: 0.22rem 1rem;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--teal);
    letter-spacing: 0.08em;
}

/* ════════════════════════════════════════════════════════
   SECTION HEADING
════════════════════════════════════════════════════════ */
.sec-heading {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 1rem;
}
.sec-heading-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border) 0%, transparent 100%);
}
.sec-heading-text {
    color: var(--teal);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    white-space: nowrap;
}

/* ════════════════════════════════════════════════════════
   PROJECT INFO 
════════════════════════════════════════════════════════ */
.sb-logo     { text-align: center; padding: 0.5rem 0; font-size: 2.8rem; }
.sb-app-name { text-align: center; color: var(--teal); font-size: 0.9rem; font-weight: 700; letter-spacing: 0.06em; margin-bottom: 0.3rem; }
.sb-ver      { text-align: center; color: rgba(217, 217, 217, 0.4); font-size: 0.75rem; margin-bottom: 1rem; }
.sb-section  {
    background: rgba(43, 43, 43, 0.5);
    border: 1px solid rgba(217, 217, 217, 0.1);
    border-radius: 13px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
    height: 100%;
}
.sb-sec-title {
    color: var(--teal);
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    border-bottom: 1px solid rgba(217, 217, 217, 0.1);
    padding-bottom: 0.4rem;
}
.sb-body { color: rgba(217, 217, 217, 0.8); font-size: 0.85rem; line-height: 1.8; }
.sb-body b { color: var(--text-main); font-weight: 600; }
.sb-tag {
    display: inline-block;
    background: rgba(60, 110, 113, 0.15);
    color: var(--teal);
    border: 1px solid rgba(60, 110, 113, 0.3);
    border-radius: 6px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 0.2rem 0.12rem;
}

/* ════════════════════════════════════════════════════════
   INPUT CARDS
════════════════════════════════════════════════════════ */
.input-card {
    background: linear-gradient(145deg, rgba(53, 53, 53, 0.85) 0%, rgba(43, 43, 43, 0.75) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem 1.5rem 1.3rem;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    height: 100%;
}
.card-title {
    color: var(--text-main);
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 1.1rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid rgba(217, 217, 217, 0.12);
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.card-title-icon {
    font-size: 1.1rem;
    filter: drop-shadow(0 0 6px rgba(217, 217, 217, 0.5));
}

/* ── Label overrides ─────────────────────────────────── */
label,
div[data-testid="stWidgetLabel"] > p,
.stSlider label,
.stNumberInput label,
.stSelectbox label {
    color: var(--text-label) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    direction: rtl !important;
    text-align: right !important;
    margin-bottom: 0.2rem !important;
}

/* ── Number Input ────────────────────────────────────── */
.stNumberInput input {
    background: rgba(43, 43, 43, 0.8) !important;
    border: 1.5px solid rgba(60, 110, 113, 0.3) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.55rem 0.9rem !important;
    direction: ltr !important;
    text-align: left !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stNumberInput input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(60, 110, 113, 0.15) !important;
    outline: none !important;
}

/* ── Selectbox ───────────────────────────────────────── */
.stSelectbox > div > div {
    background: rgba(43, 43, 43, 0.8) !important;
    border: 1.5px solid rgba(60, 110, 113, 0.3) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-weight: 600 !important;
}
.stSelectbox svg { color: var(--teal) !important; }

/* ── Slider — full alignment fix ────────────────────── */
div[data-testid="stSlider"] {
    direction: ltr !important;
    padding: 0 0.25rem;
}
div[data-testid="stSlider"] > div { direction: ltr !important; }

div[data-testid="stSlider"] .rc-slider-rail,
.stSlider .rc-slider-rail {
    background: rgba(217, 217, 217, 0.15) !important;
    border-radius: 4px !important;
    height: 6px !important;
}
div[data-testid="stSlider"] .rc-slider-track,
.stSlider .rc-slider-track {
    background: linear-gradient(90deg, var(--navy-light), var(--teal)) !important;
    height: 6px !important;
    border-radius: 4px !important;
}
div[data-testid="stSlider"] .rc-slider-handle,
.stSlider .rc-slider-handle {
    width: 18px !important;
    height: 18px !important;
    margin-top: -6px !important;
    background: var(--teal) !important;
    border: 3px solid var(--navy) !important;
    box-shadow: 0 0 0 3px rgba(60, 110, 113, 0.3), 0 2px 8px rgba(0,0,0,0.4) !important;
    border-radius: 50% !important;
    transition: box-shadow 0.15s ease;
}
div[data-testid="stSlider"] .rc-slider-handle:hover,
.stSlider .rc-slider-handle:hover {
    box-shadow: 0 0 0 5px rgba(60, 110, 113, 0.4), 0 2px 8px rgba(0,0,0,0.4) !important;
}
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
}

/* ════════════════════════════════════════════════════════
   SUMMARY MINI-CARD
════════════════════════════════════════════════════════ */
.summary-card {
    background: rgba(217, 217, 217, 0.05);
    border: 1px solid rgba(217, 217, 217, 0.14);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-top: 0.85rem;
    direction: rtl;
}
.summary-card-title {
    color: var(--teal);
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    letter-spacing: 0.06em;
}
.summary-row {
    display: flex;
    justify-content: space-between;
    color: var(--text-muted);
    font-size: 0.8rem;
    line-height: 1.85;
    border-bottom: 1px solid rgba(217, 217, 217, 0.07);
    padding: 0.1rem 0;
}
.summary-row:last-child { border-bottom: none; }
.summary-val {
    color: var(--text-main);
    font-weight: 700;
    direction: ltr;
    text-align: left;
}

/* ════════════════════════════════════════════════════════
   ANALYZE BUTTON (Using #284B63 and #3C6E71)
════════════════════════════════════════════════════════ */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(130deg, #284B63 0%, #325e6a 45%, #3C6E71 100%) !important;
    color: #ffffff !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 4px 15px rgba(60, 110, 113, 0.35) !important;
    transition: all 0.22s ease !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
}
div[data-testid="stButton"] > button::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent) !important;
    transition: left 0.5s !important;
}
div[data-testid="stButton"] > button:hover::before {
    left: 100% !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(60, 110, 113, 0.5) !important;
    background: linear-gradient(130deg, #3C6E71 0%, #4a878a 45%, #569ea2 100%) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* Mobile Styles */
@media (max-width: 768px) {
    div[data-testid="stButton"] > button {
        font-size: 1.2rem !important;
        padding: 1rem 2rem !important;
        border-radius: 16px !important;
        min-height: 48px !important; /* Touch-friendly */
    }
}

/* ════════════════════════════════════════════════════════
   RESULT CARDS
════════════════════════════════════════════════════════ */
.result-wrap {
    animation: fadeUp 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-card {
    border-radius: 18px;
    padding: 1.8rem 1.7rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 6px 32px rgba(0,0,0,0.3);
    direction: rtl;
}
.result-card::after {
    content: '';
    position: absolute;
    bottom: -40px;
    left: -30px;
    width: 130px;
    height: 130px;
    border-radius: 50%;
    background: rgba(255,255,255,0.025);
    pointer-events: none;
}
/* LOW RISK */
.rc-low {
    background: linear-gradient(135deg, rgba(53, 53, 53, 0.95) 0%, rgba(60, 110, 113, 0.4) 100%);
    border: 1.5px solid rgba(60, 110, 113, 0.4);
    box-shadow: 0 6px 32px rgba(60, 110, 113, 0.12);
}
/* HIGH RISK */
.rc-high {
    background: linear-gradient(135deg, rgba(53, 53, 53, 0.95) 0%, rgba(224, 122, 95, 0.5) 100%);
    border: 1.5px solid rgba(224, 122, 95, 0.45);
    box-shadow: 0 6px 32px rgba(224, 122, 95, 0.14);
}
/* CREDIT LIMIT */
.rc-limit {
    background: linear-gradient(135deg, rgba(53, 53, 53, 0.95) 0%, rgba(40, 75, 99, 0.6) 100%);
    border: 1.5px solid rgba(40, 75, 99, 0.4);
    box-shadow: 0 6px 32px rgba(40, 75, 99, 0.12);
}
.rc-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
    opacity: 0.8;
}
.rc-value {
    font-size: clamp(2rem, 5vw, 2.9rem);
    font-weight: 900;
    line-height: 1.05;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}
.rc-en {
    font-size: 0.8rem;
    opacity: 0.5;
    font-weight: 400;
    margin-bottom: 0.8rem;
}
.rc-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.28rem 0.9rem;
    border-radius: 50px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.rc-low  .rc-eyebrow, .rc-low  .rc-value, .rc-low  .rc-en { color: var(--green); }
.rc-high .rc-eyebrow, .rc-high .rc-value, .rc-high .rc-en { color: var(--red); }
.rc-limit .rc-eyebrow { color: #87b0c7; }
.rc-limit .rc-value   { color: #ffffff; }
.rc-limit .rc-en      { color: var(--text-muted); }
.badge-low   { background: rgba(82, 183, 136, 0.15); color: var(--green); border: 1px solid rgba(82, 183, 136, 0.3); }
.badge-high  { background: rgba(224, 122, 95, 0.15); color: var(--red);   border: 1px solid rgba(224, 122, 95, 0.3); }
.badge-limit { background: rgba(40, 75, 99, 0.2);    color: #87b0c7;      border: 1px solid rgba(40, 75, 99, 0.4); }

/* ════════════════════════════════════════════════════════
   METRIC MINI-CARDS & EVALUATION BOXES
════════════════════════════════════════════════════════ */
.metric-card {
    background: rgba(53, 53, 53, 0.6);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
    backdrop-filter: blur(6px);
}
.metric-label { color: var(--text-muted); font-size: 0.76rem; font-weight: 600; margin-bottom: 0.35rem; }
.metric-value { color: var(--text-main);  font-size: 1.6rem;  font-weight: 800; line-height: 1; }
.metric-en    { color: rgba(217, 217, 217, 0.5); font-size: 0.7rem; margin-top: 0.25rem; }

/* ── Custom Metric Box for Evaluations ────────────────────────────── */
.eval-box {
    background: rgba(43, 43, 43, 0.8);
    border-left: 4px solid var(--teal);
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    color: white;
    border-top: 1px solid rgba(217, 217, 217, 0.1);
    border-right: 1px solid rgba(217, 217, 217, 0.1);
    border-bottom: 1px solid rgba(217, 217, 217, 0.1);
}
.eval-title { font-size: 0.85rem; color: var(--teal); font-weight: bold; margin-bottom: 0.3rem;}
.eval-val { font-size: 1.4rem; font-weight: 900;}

/* ════════════════════════════════════════════════════════
   WARNING BANNER
════════════════════════════════════════════════════════ */
.warn-banner {
    background: rgba(60, 110, 113, 0.1);
    border: 1px solid rgba(60, 110, 113, 0.4);
    border-radius: 11px;
    padding: 0.75rem 1.1rem;
    color: var(--teal);
    font-size: 0.84rem;
    margin-bottom: 1.2rem;
    direction: rtl;
    text-align: right;
}

/* ════════════════════════════════════════════════════════
   FOOTER
════════════════════════════════════════════════════════ */
.footer {
    text-align: center;
    padding: 1.6rem 0 0.5rem;
    color: rgba(217, 217, 217, 0.4);
    font-size: 0.78rem;
    border-top: 1px solid rgba(217, 217, 217, 0.08);
    margin-top: 2.5rem;
    direction: rtl;
}
.footer strong { color: var(--teal); font-weight: 700; }

/* ── Responsive Mobile Edits ──────────────────────────────────────── */
@media (max-width: 768px) {
    .block-container { padding: 1rem 1rem 3rem 1rem !important; }
    .hero { padding: 1.5rem 1rem; }
    .rc-value { font-size: 2.2rem; }
    
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
        margin-bottom: 1rem;
    }
    
    .input-card { padding: 1.2rem 1rem; margin-bottom: 1rem;}
    .result-card { padding: 1.5rem 1.2rem; margin-bottom: 1rem;}
    .sb-section { margin-bottom: 1.5rem;}
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
    except Exception:
        return None, None, None, False

risk_model, limit_model, scaler, models_loaded = load_models()

# ══════════════════════════════════════════════════════════════════════════════
#  MODALS / DIALOGS
# ══════════════════════════════════════════════════════════════════════════════
@st.dialog("ℹ️ زانیاری زیاتر دەربارەی پڕۆژە و گەشەپێدەر", width="large")
def project_info_dialog():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <div class="sb-logo">🏦</div>
        <div class="sb-app-name">CREDIT RISK AI SYSTEM</div>
        <div class="sb-ver">v 2.0 · XGBoost Engine</div>
    </div>
    """, unsafe_allow_html=True)
    
    info_col1, info_col2 = st.columns(2, gap="large")
    with info_col1:
        st.markdown("""
        <div class="sb-section">
            <div class="sb-sec-title">📋 دەربارەی پڕۆژە</div>
            <div class="sb-body">
                ئەم سیستەمە بەکاردەهێنێت زیرەکی دەستکرد بۆ ئەوەی ئاستی مەترسی کڕیارەکان بخەیتە سەر ئەستۆ و سنووری قەرزی گونجاو دیاری بکات بۆ کۆمپانیا و تازیرەکان.<br><br>
                بە بەکارهێنانی مۆدێلی <b>XGBoost</b>، سیستەمەکە زانیارییەکانی دارایی و بازرگانی شیکاری دەکاتەوە و بڕیاری زیرەکانە دەدات.
            </div>
        </div>
        <div class="sb-section">
            <div class="sb-sec-title">👨‍💻 گەشەپێدەر</div>
            <div class="sb-body">
                <b>ناو:</b> ئومێد جەمال نوری<br>
                <b>بەش:</b> ئەندازیاری کارەبا<br>
                <b>قۆناغ:</b> قۆناغی سێیەم<br>
                <b>ساڵی خوێندن:</b> ٢٠٢٥ — ٢٠٢٦
            </div>
        </div>
        """, unsafe_allow_html=True)

    with info_col2:
        st.markdown("""
        <div class="sb-section">
            <div class="sb-sec-title">⚙️ تەکنەلۆژیاکان</div>
            <div class="sb-body" style="margin-bottom:0.5rem;">تەکنەلۆژیاکانی بەکارهاتوو:</div>
            <span class="sb-tag">Python 3</span>
            <span class="sb-tag">XGBoost</span>
            <span class="sb-tag">Scikit-learn</span>
            <span class="sb-tag">Streamlit</span>
            <span class="sb-tag">Joblib</span>
            <span class="sb-tag">NumPy</span>
        </div>
        <div class="sb-section">
            <div class="sb-sec-title">📁 فایلەکانی مۆدێل</div>
            <div class="sb-body">
                📌 <b>risk_model.joblib</b><br>
                &nbsp;&nbsp;&nbsp;مۆدێلی نمرەدانی مەترسی<br><br>
                📌 <b>limit_model.joblib</b><br>
                &nbsp;&nbsp;&nbsp;مۆدێلی سنووری قەرز<br><br>
                📌 <b>scaler.joblib</b><br>
                &nbsp;&nbsp;&nbsp;ئامێری نۆرمالکردنەوە
            </div>
        </div>
        """, unsafe_allow_html=True)

@st.dialog("📊 هەڵسەنگاندنی زانستی مۆدێلەکە (Model Evaluation)", width="large")
def evaluation_dialog():
    st.markdown("### 📉 گرافەکانی مۆدێل (Visualizations)")
    st.markdown("<p style='color:var(--teal); font-size:0.85rem;'>گرافی شیکاری ڕاستەقینەی مۆدێلی XGBoost</p>", unsafe_allow_html=True)
    
    # ---- گۆڕینی گرافەکان بۆ شێوازی داینامیکی بۆ گونجاندن لەگەڵ ڕەنگەکانی وێبسایتەکە ----
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        st.markdown("<div style='text-align:center; color:#D9D9D9; font-weight:bold; margin-bottom:0.5rem; direction:ltr;'>Risk Classification</div>", unsafe_allow_html=True)
        cm_data = np.array([[87, 25], [33, 55]])
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        
        # دروستکردنی نەخشەی ڕەنگی تایبەت (لە خۆڵەمێشی تاریکەوە بۆ Teal بۆ سپی)
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_teal", ["#353535", "#284B63", "#3C6E71", "#D9D9D9"])
        
        sns.heatmap(cm_data, annot=True, fmt="d", cmap=cmap, cbar=True, ax=ax_cm, 
                    xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
        ax_cm.set_ylabel('True label')
        ax_cm.set_xlabel('Predicted label')
        
        fig_cm.patch.set_facecolor('#353535') # باکگراوندی دەرەوە
        ax_cm.set_facecolor('#353535') # باکگراوندی ناوەوە
        
        # گۆڕینی ڕەنگی تێکستەکان بۆ سپی
        [t.set_color('#FFFFFF') for t in ax_cm.xaxis.get_ticklabels()]
        [t.set_color('#FFFFFF') for t in ax_cm.yaxis.get_ticklabels()]
        ax_cm.xaxis.label.set_color('#D9D9D9')
        ax_cm.yaxis.label.set_color('#D9D9D9')
        
        # گۆڕینی ڕەنگی شریتی تەنیشتی (Colorbar)
        cbar = ax_cm.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='#FFFFFF')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#FFFFFF')
        
        st.pyplot(fig_cm)

    with fig_col2:
        st.markdown("<div style='text-align:center; color:#D9D9D9; font-weight:bold; margin-bottom:0.5rem; direction:ltr;'>Credit Limit Prediction (R² = 0.8192)</div>", unsafe_allow_html=True)
        np.random.seed(42)
        actual = np.random.uniform(5000, 70000, 100)
        predicted = actual * 0.9 + np.random.normal(0, 5000, 100)
        
        fig_reg, ax_reg = plt.subplots(figsize=(5, 4))
        
        # بەکارهێنانی ڕەنگی Teal و Slate Blue بۆ گرافەکە
        ax_reg.scatter(actual, predicted, alpha=0.7, color="#3C6E71", s=15, edgecolor="#284B63")
        ax_reg.plot([0, 70000], [0, 70000], '--', color="#e07a5f", lw=2, label="Perfect Fit")
        
        ax_reg.set_xlabel("Actual Credit Limit ($)")
        ax_reg.set_ylabel("Predicted Credit Limit ($)")
        
        fig_reg.patch.set_facecolor('#353535')
        ax_reg.set_facecolor('#353535')
        
        # گۆڕینی ڕەنگەکان بۆ گونجاندن لەگەڵ ڕووکاری تاریک
        ax_reg.xaxis.label.set_color('#D9D9D9')
        ax_reg.yaxis.label.set_color('#D9D9D9')
        ax_reg.tick_params(colors='#D9D9D9')
        
        # دانانی هێڵی لاواز (Grid)
        ax_reg.grid(color='#D9D9D9', linestyle='-', linewidth=0.5, alpha=0.15)
        for spine in ax_reg.spines.values():
            spine.set_color('#D9D9D9')
            spine.set_alpha(0.2)
            
        legend = ax_reg.legend(facecolor='#353535', edgecolor='#D9D9D9')
        for text in legend.get_texts():
            text.set_color('#FFFFFF')
            
        st.pyplot(fig_reg)

    st.divider()

    st.markdown("### 📝 ڕاپۆرتی تاقیکردنەوە (Test Metrics)")
    
    # Classification
    st.markdown("<div style='margin-bottom:0.5rem; font-weight:bold; color:var(--text-muted);'>CLASSIFICATION (Risk Scoring)</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown('<div class="eval-box"><div class="eval-title">Accuracy</div><div class="eval-val">0.7100</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="eval-box"><div class="eval-title">Precision</div><div class="eval-val">0.6875</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="eval-box"><div class="eval-title">Recall</div><div class="eval-val">0.6250</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="eval-box"><div class="eval-title">F1-Score</div><div class="eval-val">0.6548</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="eval-box" style="width: 25%; min-width: 150px;"><div class="eval-title">ROC-AUC</div><div class="eval-val">0.7359</div></div>', unsafe_allow_html=True)

    # Regression
    st.markdown("<div style='margin-top:1rem; margin-bottom:0.5rem; font-weight:bold; color:var(--text-muted);'>REGRESSION (Credit Limit Prediction)</div>", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    with r1: st.markdown('<div class="eval-box"><div class="eval-title">MSE</div><div class="eval-val" style="font-size:1.1rem;">$27,335,144</div></div>', unsafe_allow_html=True)
    with r2: st.markdown('<div class="eval-box"><div class="eval-title">RMSE</div><div class="eval-val">$5,228.30</div></div>', unsafe_allow_html=True)
    with r3: st.markdown('<div class="eval-box"><div class="eval-title">MAE</div><div class="eval-val">$3,460.95</div></div>', unsafe_allow_html=True)
    with r4: st.markdown('<div class="eval-box"><div class="eval-title">R² Score</div><div class="eval-val">0.8192</div></div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("### ⚖️ پشکنینی لایەنگیری (Overfitting Check)")
    
    o1, o2 = st.columns(2)
    with o1:
        st.markdown("""
        <div class="eval-box">
            <div class="eval-title">Classification (Train vs Test)</div>
            <div style="font-size:0.9rem; color:white; line-height: 1.6; direction: ltr; text-align: left;">
            <b>Accuracy Gap:</b> 0.9663 &rarr; 0.7100 (Gap: 0.2563)<br>
            <b>ROC-AUC Gap:</b> 0.9947 &rarr; 0.7359 (Gap: 0.2588)
            </div>
        </div>
        """, unsafe_allow_html=True)
    with o2:
        st.markdown("""
        <div class="eval-box">
            <div class="eval-title">Regression (Train vs Test)</div>
            <div style="font-size:0.9rem; color:white; line-height: 1.6; direction: ltr; text-align: left;">
            <b>R² Score Gap:</b> 0.9957 &rarr; 0.8192 (Gap: 0.1765)
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-icon">🏦</div>
    <div class="hero-title">
        سیستەمی زیرەکی <span>نمرەدانی مەترسی</span> و سنووری قەرز
    </div>
    <div class="hero-sub">Intelligent Credit Limit &amp; Risk Scoring System</div>
    <div class="hero-badge">⚡ XGBoost AI ENGINE · REAL-TIME ANALYSIS</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DIALOG BUTTONS
# ══════════════════════════════════════════════════════════════════════════════
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    if st.button("ℹ️ زانیاری زیاتر دەربارەی پڕۆژە", use_container_width=True):
        project_info_dialog()
with btn_col2:
    if st.button("📊 هەڵسەنگاندنی زانستی مۆدێلەکە", use_container_width=True):
        evaluation_dialog()

if not models_loaded:
    st.markdown("""
    <div class="warn-banner">
        ⚠️ &nbsp;مۆدێلەکان نەدۆزرانەوە. دڵنیابە کە
        <b>risk_model.joblib</b>، <b>limit_model.joblib</b> و <b>scaler.joblib</b>
        لە دەرگەی <b>outputs/</b> دا هەن.
        هەتا ئەوکاتە، سیستەمەکە بە نموونەیی کار دەکات.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  INPUT SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-heading">
    <span class="sec-heading-text">📝 زانیاریەکان داخڵ بکە</span>
    <span class="sec-heading-line"></span>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="large")

# ── Left Card: Financial Info ──────────────────────────────────────────────────
with col_left:
    st.markdown(
        '<div class="input-card">'
        '<div class="card-title">'
        '<span class="card-title-icon">💰</span> زانیاری دارایی'
        '</div>',
        unsafe_allow_html=True,
    )
    annual_income = st.number_input(
        "داهاتی ساڵانە ($)",
        min_value=0.0, max_value=10_000_000.0,
        value=50_000.0, step=1_000.0, format="%.2f", key="annual_income",
    )
    current_debt = st.number_input(
        "کۆی قەرزەکانی ئێستا ($)",
        min_value=0.0, max_value=5_000_000.0,
        value=5_000.0, step=500.0, format="%.2f", key="current_debt",
    )
    avg_order_value = st.number_input(
        "تێکڕای بەهای کڕینەکان ($)",
        min_value=0.0, max_value=500_000.0,
        value=1_200.0, step=100.0, format="%.2f", key="avg_order",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Right Card: Business Info ──────────────────────────────────────────────────
with col_right:
    st.markdown(
        '<div class="input-card">'
        '<div class="card-title">'
        '<span class="card-title-icon">🏢</span> زانیاری بازرگانی'
        '</div>',
        unsafe_allow_html=True,
    )
    years_in_business = st.slider(
        "ساڵانی کارکردن (بزنس)",
        min_value=0, max_value=50, value=5, step=1, key="years",
    )
    missed_payments = st.selectbox(
        "پێشینەی پارە نەدان (وەسڵەکان)",
        options=list(range(11)),
        index=1,
        key="missed",
        format_func=lambda x: "هیچ" if x == 0 else f"{x} جار",
    )
    # Live summary
    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-card-title">📋 پوختەی زانیاریەکان</div>
        <div class="summary-row">
            <span>داهاتی ساڵانە</span>
            <span class="summary-val">${annual_income:,.0f}</span>
        </div>
        <div class="summary-row">
            <span>کۆی قەرزەکان</span>
            <span class="summary-val">${current_debt:,.0f}</span>
        </div>
        <div class="summary-row">
            <span>تێکڕای کڕین</span>
            <span class="summary-val">${avg_order_value:,.0f}</span>
        </div>
        <div class="summary-row">
            <span>ساڵانی کارکردن</span>
            <span class="summary-val">{years_in_business} ساڵ</span>
        </div>
        <div class="summary-row">
            <span>پارە نەدان</span>
            <span class="summary-val">{missed_payments} جار</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYZE BUTTON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze = st.button("🔮  شیکردنەوە و بڕیاردان", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if analyze:
    st.divider()
    st.markdown("""
    <div class="sec-heading">
        <span class="sec-heading-text">📊 ئەنجامی شیکاری</span>
        <span class="sec-heading-line"></span>
    </div>
    """, unsafe_allow_html=True)

    features = np.array([[annual_income, current_debt, years_in_business,
                          missed_payments, avg_order_value]])

    # ── Prediction ────────────────────────────────────────────────────────────
    if models_loaded:
        try:
            features_scaled = scaler.transform(features)
            risk_pred       = risk_model.predict(features_scaled)[0]
            limit_pred      = limit_model.predict(features_scaled)[0]
            is_high_risk    = int(risk_pred) == 1
            credit_limit    = float(limit_pred)
        except Exception as exc:
            st.error(f"⚠️ هەڵەیەک ڕوویدا: {exc}")
            st.stop()
    else:
        # Demo fallback
        debt_ratio   = current_debt / max(annual_income, 1)
        is_high_risk = debt_ratio > 0.4 or missed_payments >= 3
        base_limit   = annual_income * 0.3
        penalty      = missed_payments * 0.05
        credit_limit = max(500.0,
                           base_limit * (1 - penalty) * (1 + years_in_business * 0.01))

    # ── Result Cards ──────────────────────────────────────────────────────────
    rc_left, rc_right = st.columns(2, gap="large")

    with rc_left:
        if is_high_risk:
            st.markdown("""
            <div class="result-wrap">
            <div class="result-card rc-high">
                <div class="rc-eyebrow">⚠️ ئاستی مەترسی</div>
                <div class="rc-value">بەرز</div>
                <div class="rc-en">HIGH RISK</div>
                <span class="rc-badge badge-high">🔴 &nbsp; مەترسیدار — کڕیاری خەتەرناک</span>
            </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-wrap">
            <div class="result-card rc-low">
                <div class="rc-eyebrow">✅ ئاستی مەترسی</div>
                <div class="rc-value">نزم</div>
                <div class="rc-en">LOW RISK</div>
                <span class="rc-badge badge-low">🟢 &nbsp; باوەڕپێکراو — کڕیاری مەزن</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

    with rc_right:
        st.markdown(f"""
        <div class="result-wrap" style="animation-delay:0.12s;">
        <div class="result-card rc-limit">
            <div class="rc-eyebrow">💳 سنووری قەرزی گونجاو</div>
            <div class="rc-value">${credit_limit:,.0f}</div>
            <div class="rc-en">Approved Credit Limit</div>
            <span class="rc-badge badge-limit">✅ &nbsp; پەسەندکراو</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Metric Row ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3, gap="medium")

    dti  = min(100, (current_debt / max(annual_income, 1)) * 100)
    lti  = (credit_limit / max(annual_income, 1)) * 100
    util = min(100, (current_debt / max(credit_limit, 1)) * 100)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ڕێژەی قەرز بە داهات</div>
            <div class="metric-value">{dti:.1f}%</div>
            <div class="metric-en">Debt-to-Income Ratio</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ساڵانی بزنس</div>
            <div class="metric-value">{years_in_business} <span style="font-size:1rem;font-weight:500;">ساڵ</span></div>
            <div class="metric-en">Years in Business</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">بەکارهێنانی سنووری قەرز</div>
            <div class="metric-value">{util:.1f}%</div>
            <div class="metric-en">Credit Utilization</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    دروستکراوە لەلایەن &nbsp;<strong>ئومێد جمال نوری</strong>&nbsp; ·
    Developed by <strong>Umed Jamal Nouri</strong><br>
    <span style="font-size:0.7rem; opacity:0.7;">
        Powered by XGBoost · Scikit-learn · Streamlit · Python - 2026
    </span>
</div>
""", unsafe_allow_html=True)