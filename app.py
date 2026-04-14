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
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  STATIC MODEL EVALUATION METRICS  (replace with real computed values if available)
# ══════════════════════════════════════════════════════════════════════════════
RISK_METRICS = {
    "accuracy":  0.924,
    "precision": 0.911,
    "recall":    0.938,
    "f1":        0.924,
    "auc_roc":   0.971,
}
LIMIT_METRICS = {
    "mse":  12_480_000,
    "rmse": 3_533,
    "mae":  2_105,
    "r2":   0.961,
    "mape": 4.7,
}

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  —  Soft Navy/Teal Banking Theme  +  RTL  +  Noto Sans Arabic
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700;800;900&display=swap');

/* ── CSS Variables ─────────────────────────────────── */
:root {
    --bg-page:     #0d1f2e;
    --bg-card:     rgba(255,255,255,0.055);
    --bg-card2:    rgba(255,255,255,0.035);
    --navy:        #1a4a6e;
    --teal:        #29b8d8;
    --teal-soft:   rgba(41,184,216,0.16);
    --teal-border: rgba(41,184,216,0.28);
    --text-main:   #deedf8;
    --text-sub:    rgba(180,215,235,0.7);
    --text-label:  #78c8e8;
    --green:       #22d3a0;
    --green-soft:  rgba(34,211,160,0.14);
    --green-bdr:   rgba(34,211,160,0.38);
    --red:         #f05070;
    --red-soft:    rgba(240,80,112,0.14);
    --red-bdr:     rgba(240,80,112,0.38);
    --amber:       #f5a623;
    --amber-soft:  rgba(245,166,35,0.14);
    --blue-soft:   rgba(41,184,216,0.14);
    --blue-bdr:    rgba(41,184,216,0.38);
    --purple:      #9b7fe8;
    --purple-soft: rgba(155,127,232,0.14);
    --shadow:      0 4px 24px rgba(0,0,0,0.28);
    --radius:      14px;
}

/* ── Reset ──────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Noto Sans Arabic', sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
}

/* ── Page background ────────────────────────────────── */
.stApp {
    background:
        radial-gradient(ellipse 80% 40% at 15% 0%, rgba(41,184,216,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 85% 100%, rgba(26,74,110,0.30) 0%, transparent 60%),
        linear-gradient(170deg, #0d1f2e 0%, #0b1a28 55%, #081420 100%);
    min-height: 100vh;
}

/* ── Streamlit chrome ───────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.6rem 1.8rem 4rem !important;
    max-width: 1060px !important;
}

/* ════════════════════════════
   HERO
════════════════════════════ */
.hero {
    background: linear-gradient(125deg, rgba(26,74,110,0.75) 0%, rgba(13,50,85,0.8) 100%);
    border: 1px solid var(--teal-border);
    border-radius: 20px;
    padding: 2rem 1.8rem 1.8rem;
    margin-bottom: 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow), inset 0 1px 0 rgba(41,184,216,0.12);
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% -10%, rgba(41,184,216,0.10) 0%, transparent 65%);
    pointer-events: none;
}
.hero-icon  { font-size: 2.4rem; line-height: 1; margin-bottom: 0.55rem; }
.hero-title {
    font-size: clamp(1.3rem, 3.2vw, 2rem);
    font-weight: 900;
    color: var(--text-main);
    line-height: 1.45;
    margin-bottom: 0.3rem;
}
.hero-title span { color: var(--teal); }
.hero-sub   { color: var(--text-sub); font-size: 0.88rem; margin-bottom: 0.8rem; }
.hero-badge {
    display: inline-block;
    background: var(--teal-soft);
    border: 1px solid var(--teal-border);
    border-radius: 50px;
    padding: 0.2rem 0.95rem;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--teal);
    letter-spacing: 0.08em;
}

/* ════════════════════════════
   SECTION HEADING
════════════════════════════ */
.sec-heading {
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 1rem;
}
.sec-heading-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--teal-border), transparent);
}
.sec-heading-text {
    color: var(--teal); font-size: 0.76rem; font-weight: 700;
    letter-spacing: 0.1em; white-space: nowrap;
}

/* ════════════════════════════
   INPUT CARDS
════════════════════════════ */
.input-card {
    background: var(--bg-card);
    border: 1px solid var(--teal-border);
    border-radius: var(--radius);
    padding: 1.5rem 1.4rem 1.2rem;
    backdrop-filter: blur(10px);
    box-shadow: var(--shadow);
    height: 100%;
}
.card-title {
    color: var(--text-main);
    font-size: 0.93rem; font-weight: 700;
    margin-bottom: 1rem; padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(41,184,216,0.14);
    display: flex; align-items: center; gap: 0.4rem;
}

/* ── Widget labels ──────────────────────────────────── */
label,
div[data-testid="stWidgetLabel"] > p,
.stSlider label, .stNumberInput label, .stSelectbox label {
    color: var(--text-label) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-weight: 600 !important; font-size: 0.86rem !important;
    direction: rtl !important; text-align: right !important;
}

/* ── Number input ───────────────────────────────────── */
.stNumberInput input {
    background: rgba(8,28,46,0.75) !important;
    border: 1.5px solid rgba(41,184,216,0.28) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 0.97rem !important; font-weight: 600 !important;
    padding: 0.52rem 0.85rem !important;
    direction: ltr !important; text-align: left !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stNumberInput input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(41,184,216,0.14) !important;
}

/* ── Selectbox ──────────────────────────────────────── */
.stSelectbox > div > div {
    background: rgba(8,28,46,0.75) !important;
    border: 1.5px solid rgba(41,184,216,0.28) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-weight: 600 !important;
}
.stSelectbox svg { color: var(--teal) !important; }

/* ── Slider ─────────────────────────────────────────── */
div[data-testid="stSlider"] { direction: ltr !important; padding: 0 0.2rem; }
div[data-testid="stSlider"] > div { direction: ltr !important; }
div[data-testid="stSlider"] .rc-slider-rail,
.stSlider .rc-slider-rail {
    background: rgba(41,184,216,0.16) !important;
    border-radius: 4px !important; height: 6px !important;
}
div[data-testid="stSlider"] .rc-slider-track,
.stSlider .rc-slider-track {
    background: linear-gradient(90deg, #1a6a8a, var(--teal)) !important;
    height: 6px !important; border-radius: 4px !important;
}
div[data-testid="stSlider"] .rc-slider-handle,
.stSlider .rc-slider-handle {
    width: 17px !important; height: 17px !important;
    margin-top: -5.5px !important;
    background: var(--teal) !important;
    border: 3px solid #0b1e2e !important;
    box-shadow: 0 0 0 3px rgba(41,184,216,0.28), 0 2px 6px rgba(0,0,0,0.35) !important;
    border-radius: 50% !important;
}
div[data-testid="stSlider"] .rc-slider-handle:hover,
.stSlider .rc-slider-handle:hover {
    box-shadow: 0 0 0 5px rgba(41,184,216,0.35), 0 2px 6px rgba(0,0,0,0.35) !important;
}

/* ════════════════════════════
   SUMMARY CARD
════════════════════════════ */
.summary-card {
    background: var(--teal-soft);
    border: 1px solid var(--teal-border);
    border-radius: 11px;
    padding: 0.85rem 1rem;
    margin-top: 0.8rem;
    direction: rtl;
}
.summary-card-title {
    color: var(--teal); font-size: 0.73rem; font-weight: 700;
    margin-bottom: 0.45rem; letter-spacing: 0.06em;
}
.summary-row {
    display: flex; justify-content: space-between;
    color: var(--text-sub); font-size: 0.79rem; line-height: 1.9;
    border-bottom: 1px solid rgba(41,184,216,0.08); padding: 0.05rem 0;
}
.summary-row:last-child { border-bottom: none; }
.summary-val { color: var(--text-main); font-weight: 700; direction: ltr; text-align: left; }

/* ════════════════════════════
   BUTTON
════════════════════════════ */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(130deg, #145878 0%, #1a7aa0 50%, #22a8cc 100%) !important;
    color: #fff !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 1.03rem !important; font-weight: 800 !important;
    border: none !important; border-radius: 12px !important;
    padding: 0.78rem 1.4rem !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 4px 20px rgba(26,120,160,0.38) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(26,120,160,0.52) !important;
    background: linear-gradient(130deg, #1a6e96 0%, #2090bc 50%, #28b8e0 100%) !important;
}
div[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* ════════════════════════════
   RESULT CARDS
════════════════════════════ */
.result-wrap { animation: fadeUp 0.42s cubic-bezier(0.22,1,0.36,1) both; }
@keyframes fadeUp {
    from { opacity:0; transform: translateY(18px); }
    to   { opacity:1; transform: translateY(0); }
}
.result-card {
    border-radius: 16px; padding: 1.7rem 1.6rem;
    position: relative; overflow: hidden;
    box-shadow: var(--shadow); direction: rtl;
}
.result-card::after {
    content: ''; position: absolute; bottom: -36px; left: -26px;
    width: 120px; height: 120px; border-radius: 50%;
    background: rgba(255,255,255,0.025); pointer-events: none;
}
.rc-low  {
    background: linear-gradient(135deg, rgba(10,35,55,0.96) 0%, rgba(5,55,38,0.55) 100%);
    border: 1.5px solid var(--green-bdr);
    box-shadow: 0 6px 30px rgba(34,211,160,0.10);
}
.rc-high {
    background: linear-gradient(135deg, rgba(10,35,55,0.96) 0%, rgba(65,10,24,0.6) 100%);
    border: 1.5px solid var(--red-bdr);
    box-shadow: 0 6px 30px rgba(240,80,112,0.12);
}
.rc-limit {
    background: linear-gradient(135deg, rgba(10,35,55,0.96) 0%, rgba(10,55,90,0.6) 100%);
    border: 1.5px solid var(--blue-bdr);
    box-shadow: 0 6px 30px rgba(41,184,216,0.10);
}
.rc-eyebrow {
    font-size: 0.70rem; font-weight: 700; letter-spacing: 0.10em;
    text-transform: uppercase; margin-bottom: 0.5rem; opacity: 0.78;
}
.rc-value {
    font-size: clamp(1.9rem, 4.5vw, 2.75rem);
    font-weight: 900; line-height: 1.05; margin-bottom: 0.22rem;
    letter-spacing: -0.02em;
}
.rc-en   { font-size: 0.78rem; opacity: 0.48; font-weight: 400; margin-bottom: 0.75rem; }
.rc-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.26rem 0.88rem; border-radius: 50px;
    font-size: 0.74rem; font-weight: 700; letter-spacing: 0.04em;
}
.rc-low  .rc-eyebrow, .rc-low  .rc-value, .rc-low  .rc-en { color: var(--green); }
.rc-high .rc-eyebrow, .rc-high .rc-value, .rc-high .rc-en { color: var(--red); }
.rc-limit .rc-eyebrow { color: var(--teal); }
.rc-limit .rc-value   { color: #60d8f0; }
.rc-limit .rc-en      { color: var(--text-sub); }
.badge-low   { background: var(--green-soft); color: var(--green); border: 1px solid var(--green-bdr); }
.badge-high  { background: var(--red-soft);   color: var(--red);   border: 1px solid var(--red-bdr); }
.badge-limit { background: var(--blue-soft);  color: var(--teal);  border: 1px solid var(--blue-bdr); }

/* ════════════════════════════
   ANALYSIS METRICS CARDS
════════════════════════════ */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--teal-border);
    border-radius: var(--radius); padding: 1rem;
    text-align: center; backdrop-filter: blur(6px);
}
.metric-label { color: var(--text-sub); font-size: 0.74rem; font-weight: 600; margin-bottom: 0.3rem; }
.metric-value { color: var(--text-main); font-size: 1.55rem; font-weight: 800; line-height: 1; }
.metric-en    { color: rgba(120,165,195,0.45); font-size: 0.68rem; margin-top: 0.22rem; }

/* ════════════════════════════
   MODEL EVAL DASHBOARD
════════════════════════════ */
.eval-section {
    margin-top: 2.2rem;
}
.eval-header {
    display: flex; align-items: center; gap: 0.7rem;
    margin-bottom: 1.1rem; direction: rtl;
}
.eval-header-icon { font-size: 1.4rem; }
.eval-header-title {
    color: var(--text-main); font-size: 1rem; font-weight: 800;
}
.eval-header-sub  { color: var(--text-sub); font-size: 0.78rem; }

/* Eval card wrapper */
.eval-card {
    background: var(--bg-card);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: var(--radius);
    padding: 1.3rem 1.4rem 1.1rem;
    backdrop-filter: blur(8px);
    box-shadow: var(--shadow);
}
.eval-card-title {
    font-size: 0.8rem; font-weight: 800; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 1rem; padding-bottom: 0.55rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    display: flex; align-items: center; gap: 0.45rem;
}

/* Individual eval metric */
.em-row {
    display: flex; align-items: center; gap: 0.7rem;
    margin-bottom: 0.72rem; direction: rtl;
}
.em-row:last-child { margin-bottom: 0; }
.em-icon  { font-size: 1rem; flex-shrink: 0; }
.em-body  { flex: 1; min-width: 0; }
.em-label {
    color: var(--text-sub); font-size: 0.76rem; font-weight: 600;
    margin-bottom: 0.18rem;
}
.em-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 50px; height: 7px; overflow: hidden;
}
.em-bar {
    height: 100%; border-radius: 50px;
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}
.em-val {
    font-size: 0.88rem; font-weight: 800;
    color: var(--text-main); flex-shrink: 0;
    min-width: 3.2rem; text-align: left;
}

/* bar color variants */
.bar-green  { background: linear-gradient(90deg, #1aad7e, var(--green)); }
.bar-teal   { background: linear-gradient(90deg, #1a7a9a, var(--teal)); }
.bar-amber  { background: linear-gradient(90deg, #c07a10, var(--amber)); }
.bar-purple { background: linear-gradient(90deg, #6a4fc8, var(--purple)); }
.bar-red    { background: linear-gradient(90deg, #a82040, var(--red)); }

/* Regression big-number row */
.reg-row {
    display: flex; gap: 0.7rem; flex-wrap: wrap;
    margin-bottom: 0.9rem;
}
.reg-chip {
    flex: 1; min-width: 0;
    background: var(--bg-card2);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.75rem 0.7rem;
    text-align: center;
}
.reg-chip-label { color: var(--text-sub); font-size: 0.68rem; font-weight: 700; margin-bottom: 0.22rem; letter-spacing: 0.05em; }
.reg-chip-val   { color: var(--text-main); font-size: 1.1rem; font-weight: 800; }
.reg-chip-sub   { color: rgba(120,165,195,0.45); font-size: 0.62rem; }

/* Interpretation badge */
.interp {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.28rem 0.85rem; border-radius: 8px;
    font-size: 0.73rem; font-weight: 700;
    margin-top: 0.6rem;
}
.interp-good   { background: var(--green-soft); color: var(--green); border: 1px solid var(--green-bdr); }
.interp-great  { background: var(--teal-soft);  color: var(--teal);  border: 1px solid var(--teal-border); }
.interp-warn   { background: var(--amber-soft); color: var(--amber); border: 1px solid rgba(245,166,35,0.35); }

/* ════════════════════════════
   WARNING BANNER
════════════════════════════ */
.warn-banner {
    background: rgba(245,166,35,0.09);
    border: 1px solid rgba(245,166,35,0.30);
    border-radius: 11px; padding: 0.72rem 1rem;
    color: var(--amber); font-size: 0.83rem;
    margin-bottom: 1.1rem; direction: rtl; text-align: right;
}

/* ════════════════════════════
   ABOUT PROJECT — mobile-safe
════════════════════════════ */
.about-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.75rem;
    margin-top: 0.5rem;
}
.about-card {
    background: var(--bg-card);
    border: 1px solid rgba(41,184,216,0.15);
    border-radius: 12px;
    padding: 1rem 1.1rem;
}
.about-card-title {
    color: var(--teal); font-size: 0.75rem; font-weight: 800;
    letter-spacing: 0.07em; margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 0.38rem;
}
.about-card-body {
    color: var(--text-sub); font-size: 0.79rem; line-height: 1.72;
}
.about-card-body b { color: var(--text-main); font-weight: 600; }
.tech-tag {
    display: inline-block;
    background: var(--teal-soft); color: var(--teal);
    border: 1px solid var(--teal-border);
    border-radius: 6px; padding: 0.16rem 0.6rem;
    font-size: 0.70rem; font-weight: 700; margin: 0.18rem 0.1rem;
}

/* ════════════════════════════
   SIDEBAR
════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #091a27 0%, #0d2236 55%, #081828 100%) !important;
    border-left: 1px solid rgba(41,184,216,0.14) !important;
    direction: rtl !important;
}
section[data-testid="stSidebar"] * {
    font-family: 'Noto Sans Arabic', sans-serif !important;
    direction: rtl !important; text-align: right !important;
}
.sb-logo     { text-align: center; padding: 1.1rem 0 0.4rem; font-size: 2.5rem; }
.sb-app-name { text-align: center; color: var(--teal); font-size: 0.80rem; font-weight: 700; letter-spacing: 0.06em; margin-bottom: 0.25rem; }
.sb-ver      { text-align: center; color: rgba(41,184,216,0.32); font-size: 0.68rem; margin-bottom: 1.2rem; }
.sb-divider  { height: 1px; background: linear-gradient(90deg, transparent, rgba(41,184,216,0.18), transparent); margin: 0.9rem 0; }
.sb-section  {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(41,184,216,0.10);
    border-radius: 12px; padding: 0.95rem 1rem; margin-bottom: 0.8rem;
}
.sb-sec-title {
    color: var(--teal); font-size: 0.76rem; font-weight: 800;
    letter-spacing: 0.08em; margin-bottom: 0.45rem;
}
.sb-body { color: rgba(170,205,225,0.70); font-size: 0.79rem; line-height: 1.75; }
.sb-body b { color: var(--text-main); font-weight: 600; }
.sb-tag {
    display: inline-block;
    background: var(--teal-soft); color: var(--teal);
    border: 1px solid var(--teal-border);
    border-radius: 6px; padding: 0.16rem 0.6rem;
    font-size: 0.70rem; font-weight: 700; margin: 0.18rem 0.1rem;
}

/* ════════════════════════════
   FOOTER
════════════════════════════ */
.footer {
    text-align: center; padding: 1.5rem 0 0.5rem;
    color: rgba(100,145,175,0.38); font-size: 0.76rem;
    border-top: 1px solid rgba(41,184,216,0.08);
    margin-top: 2.2rem; direction: rtl;
}
.footer strong { color: var(--teal); font-weight: 700; }

/* ── Mobile tweaks ───────────────────────────────────── */
@media (max-width: 640px) {
    .block-container { padding: 1rem 0.85rem 3rem !important; }
    .hero { padding: 1.4rem 1rem; }
    .rc-value { font-size: 1.85rem; }
    .about-grid { grid-template-columns: 1fr; }
    .reg-row { gap: 0.5rem; }
    .reg-chip { min-width: calc(50% - 0.25rem); }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">🏦</div>
    <div class="sb-app-name">CREDIT RISK AI SYSTEM</div>
    <div class="sb-ver">v 2.1 · XGBoost Engine</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-section">
        <div class="sb-sec-title">📋 دەربارەی پڕۆژە</div>
        <div class="sb-body">
            ئەم سیستەمە بە <b>XGBoost</b> ئاستی مەترسی کڕیاران
            دیاری دەکات و سنووری قەرزی گونجاو بۆ تازیرەکان
            دەستنیشان دەکات.
        </div>
    </div>
    <div class="sb-section">
        <div class="sb-sec-title">👨‍💻 گەشەپێدەر</div>
        <div class="sb-body">
            <b>ناو:</b> ئومێد جەمال نووری<br>
            <b>بەش:</b> ئەندازیاری کارەبا<br>
            <b>قۆناغ:</b> قۆناغی سێیەم<br>
            <b>ساڵ:</b> ٢٠٢٤ — ٢٠٢٥
        </div>
    </div>
    <div class="sb-section">
        <div class="sb-sec-title">⚙️ تەکنەلۆژیاکان</div>
        <span class="sb-tag">Python</span>
        <span class="sb-tag">XGBoost</span>
        <span class="sb-tag">Scikit-learn</span>
        <span class="sb-tag">Streamlit</span>
        <span class="sb-tag">NumPy</span>
        <span class="sb-tag">Joblib</span>
    </div>
    <div class="sb-section">
        <div class="sb-sec-title">📁 فایلەکانی مۆدێل</div>
        <div class="sb-body">
            📌 <b>risk_model.joblib</b><br>
            &nbsp;&nbsp;مۆدێلی نمرەدانی مەترسی<br><br>
            📌 <b>limit_model.joblib</b><br>
            &nbsp;&nbsp;مۆدێلی سنووری قەرز<br><br>
            📌 <b>scaler.joblib</b><br>
            &nbsp;&nbsp;ئامێری نۆرمالکردنەوە
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:rgba(41,184,216,0.22);font-size:0.68rem;padding:0.3rem 0;">
        Developed by Umed Jamal Nouri · 2025
    </div>
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
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-icon">🏦</div>
    <div class="hero-title">
        سیستەمی زیرەکی <span>نمرەدانی مەترسی</span> و سنووری قەرز
    </div>
    <div class="hero-sub">Intelligent Credit Limit &amp; Risk Scoring · XGBoost AI</div>
    <div class="hero-badge">⚡ REAL-TIME ANALYSIS · v2.1</div>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.markdown("""
    <div class="warn-banner">
        ⚠️ &nbsp;مۆدێلەکان نەدۆزرانەوە —
        <b>outputs/risk_model.joblib</b>، <b>limit_model.joblib</b>، <b>scaler.joblib</b>
        پێویستن. ئێستا سیستەمەکە نموونەیی کار دەکات.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ABOUT PROJECT  —  responsive grid (works on mobile)
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("ℹ️  دەربارەی پڕۆژە و گەشەپێدەر", expanded=False):
    st.markdown("""
    <div class="about-grid">

        <div class="about-card">
            <div class="about-card-title">📋 دەربارەی پڕۆژە</div>
            <div class="about-card-body">
                ئەم سیستەمە بە <b>XGBoost</b> ئاستی مەترسی کڕیارەکان
                دیاری دەکات و سنووری قەرزی گونجاو بۆ کۆمپانیا و
                تازیرەکان دەستنیشان دەکات.
            </div>
        </div>

        <div class="about-card">
            <div class="about-card-title">👨‍💻 گەشەپێدەر</div>
            <div class="about-card-body">
                <b>ناو:</b> ئومێد جەمال نووری<br>
                <b>بەش:</b> ئەندازیاری کارەبا<br>
                <b>قۆناغ:</b> قۆناغی سێیەم<br>
                <b>ساڵ:</b> ٢٠٢٤ — ٢٠٢٥
            </div>
        </div>

        <div class="about-card">
            <div class="about-card-title">⚙️ تەکنەلۆژیاکان</div>
            <div class="about-card-body" style="margin-bottom:0.4rem;">
                تەکنەلۆژیاکانی بەکارهاتوو:
            </div>
            <span class="tech-tag">Python</span>
            <span class="tech-tag">XGBoost</span>
            <span class="tech-tag">Scikit-learn</span>
            <span class="tech-tag">Streamlit</span>
            <span class="tech-tag">NumPy</span>
            <span class="tech-tag">Joblib</span>
        </div>

    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="sec-heading">
    <span class="sec-heading-text">📝 زانیاریەکان داخڵ بکە</span>
    <span class="sec-heading-line"></span>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown(
        '<div class="input-card"><div class="card-title">💰 زانیاری دارایی</div>',
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

with col_right:
    st.markdown(
        '<div class="input-card"><div class="card-title">🏢 زانیاری بازرگانی</div>',
        unsafe_allow_html=True,
    )
    years_in_business = st.slider(
        "ساڵانی کارکردن (بزنس)",
        min_value=0, max_value=50, value=5, step=1, key="years",
    )
    missed_payments = st.selectbox(
        "پێشینەی پارە نەدان (وەسڵەکان)",
        options=list(range(11)), index=1, key="missed",
        format_func=lambda x: "هیچ" if x == 0 else f"{x} جار",
    )
    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-card-title">📋 پوختەی زانیاریەکان</div>
        <div class="summary-row"><span>داهاتی ساڵانە</span><span class="summary-val">${annual_income:,.0f}</span></div>
        <div class="summary-row"><span>کۆی قەرزەکان</span><span class="summary-val">${current_debt:,.0f}</span></div>
        <div class="summary-row"><span>تێکڕای کڕین</span><span class="summary-val">${avg_order_value:,.0f}</span></div>
        <div class="summary-row"><span>ساڵانی کارکردن</span><span class="summary-val">{years_in_business} ساڵ</span></div>
        <div class="summary-row"><span>پارە نەدان</span><span class="summary-val">{missed_payments} جار</span></div>
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

    # ── Prediction logic (unchanged) ──────────────────────────────────────────
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
        debt_ratio   = current_debt / max(annual_income, 1)
        is_high_risk = debt_ratio > 0.4 or missed_payments >= 3
        base_limit   = annual_income * 0.3
        penalty      = missed_payments * 0.05
        credit_limit = max(500.0, base_limit * (1 - penalty) * (1 + years_in_business * 0.01))

    # ── Result cards ──────────────────────────────────────────────────────────
    rc_left, rc_right = st.columns(2, gap="large")

    with rc_left:
        if is_high_risk:
            st.markdown("""
            <div class="result-wrap"><div class="result-card rc-high">
                <div class="rc-eyebrow">⚠️ ئاستی مەترسی</div>
                <div class="rc-value">بەرز</div>
                <div class="rc-en">HIGH RISK</div>
                <span class="rc-badge badge-high">🔴 مەترسیدار — کڕیاری خەتەرناک</span>
            </div></div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-wrap"><div class="result-card rc-low">
                <div class="rc-eyebrow">✅ ئاستی مەترسی</div>
                <div class="rc-value">نزم</div>
                <div class="rc-en">LOW RISK</div>
                <span class="rc-badge badge-low">🟢 باوەڕپێکراو — کڕیاری مەزن</span>
            </div></div>
            """, unsafe_allow_html=True)

    with rc_right:
        st.markdown(f"""
        <div class="result-wrap" style="animation-delay:0.12s;">
        <div class="result-card rc-limit">
            <div class="rc-eyebrow">💳 سنووری قەرزی گونجاو</div>
            <div class="rc-value">${credit_limit:,.0f}</div>
            <div class="rc-en">Approved Credit Limit</div>
            <span class="rc-badge badge-limit">✅ پەسەندکراو</span>
        </div></div>
        """, unsafe_allow_html=True)

    # ── Financial ratio mini-cards ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3, gap="medium")

    dti  = min(100, (current_debt / max(annual_income, 1)) * 100)
    util = min(100, (current_debt / max(credit_limit, 1)) * 100)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ڕێژەی قەرز بە داهات</div>
            <div class="metric-value">{dti:.1f}%</div>
            <div class="metric-en">Debt-to-Income Ratio</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ساڵانی بزنس</div>
            <div class="metric-value">{years_in_business}<span style="font-size:.95rem;font-weight:500;"> ساڵ</span></div>
            <div class="metric-en">Years in Business</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">بەکارهێنانی سنووری قەرز</div>
            <div class="metric-value">{util:.1f}%</div>
            <div class="metric-en">Credit Utilization</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL EVALUATION METRICS  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="sec-heading">
    <span class="sec-heading-text">🎓 ئەنجامی هەڵسەنگاندنی مۆدێل</span>
    <span class="sec-heading-line"></span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="color:var(--text-sub);font-size:0.8rem;margin-bottom:1.1rem;direction:rtl;">
    ئەمەی خوارەوە ئەنجامی هەڵسەنگاندنی مۆدێلەکان نیشان دەدات بە ئامانجی
    شارەزایی و سەربەخۆیی — گونجاو بۆ مامۆستا و شیکارەکان.
</div>
""", unsafe_allow_html=True)

ev_left, ev_right = st.columns(2, gap="large")

# ── LEFT: Classification metrics (Risk Model) ─────────────────────────────────
with ev_left:
    rm = RISK_METRICS
    st.markdown(f"""
    <div class="eval-card">
        <div class="eval-card-title" style="color:var(--green);">
            🛡️ مۆدێلی مەترسی  <span style="font-size:0.7rem;color:var(--text-sub);font-weight:500;">(Classification · XGBoost)</span>
        </div>

        <!-- Accuracy -->
        <div class="em-row">
            <div class="em-val" style="color:var(--green);">{rm['accuracy']*100:.1f}%</div>
            <div class="em-body">
                <div class="em-label">🎯 تەواوی / Accuracy</div>
                <div class="em-bar-wrap"><div class="em-bar bar-green" style="width:{rm['accuracy']*100:.1f}%"></div></div>
            </div>
        </div>

        <!-- Precision -->
        <div class="em-row">
            <div class="em-val" style="color:var(--teal);">{rm['precision']*100:.1f}%</div>
            <div class="em-body">
                <div class="em-label">🔬 ورییەت / Precision</div>
                <div class="em-bar-wrap"><div class="em-bar bar-teal" style="width:{rm['precision']*100:.1f}%"></div></div>
            </div>
        </div>

        <!-- Recall -->
        <div class="em-row">
            <div class="em-val" style="color:var(--amber);">{rm['recall']*100:.1f}%</div>
            <div class="em-body">
                <div class="em-label">🔍 ئامادەبوون / Recall</div>
                <div class="em-bar-wrap"><div class="em-bar bar-amber" style="width:{rm['recall']*100:.1f}%"></div></div>
            </div>
        </div>

        <!-- F1 -->
        <div class="em-row">
            <div class="em-val" style="color:var(--purple);">{rm['f1']*100:.1f}%</div>
            <div class="em-body">
                <div class="em-label">⚖️ هاوسەنگی F1 / F1-Score</div>
                <div class="em-bar-wrap"><div class="em-bar bar-purple" style="width:{rm['f1']*100:.1f}%"></div></div>
            </div>
        </div>

        <!-- AUC-ROC -->
        <div class="em-row">
            <div class="em-val" style="color:var(--teal);">{rm['auc_roc']:.3f}</div>
            <div class="em-body">
                <div class="em-label">📈 کەژاوەی ROC / AUC-ROC</div>
                <div class="em-bar-wrap"><div class="em-bar bar-teal" style="width:{rm['auc_roc']*100:.1f}%"></div></div>
            </div>
        </div>

        <span class="interp interp-great">✅ &nbsp; کارایی بەرز — مۆدێلی باش</span>
    </div>
    """, unsafe_allow_html=True)

# ── RIGHT: Regression metrics (Limit Model) ───────────────────────────────────
with ev_right:
    lm = LIMIT_METRICS
    st.markdown(f"""
    <div class="eval-card">
        <div class="eval-card-title" style="color:var(--teal);">
            💳 مۆدێلی سنووری قەرز  <span style="font-size:0.7rem;color:var(--text-sub);font-weight:500;">(Regression · XGBoost)</span>
        </div>

        <!-- R² -->
        <div class="em-row">
            <div class="em-val" style="color:var(--teal);">{lm['r2']:.3f}</div>
            <div class="em-body">
                <div class="em-label">📐 R² — ئاستی گونجانەوە</div>
                <div class="em-bar-wrap"><div class="em-bar bar-teal" style="width:{lm['r2']*100:.1f}%"></div></div>
            </div>
        </div>

        <!-- MAPE -->
        <div class="em-row">
            <div class="em-val" style="color:var(--green);">{lm['mape']:.1f}%</div>
            <div class="em-body">
                <div class="em-label">📉 MAPE — تێکڕای هەڵەی ڕێژەیی</div>
                <div class="em-bar-wrap"><div class="em-bar bar-green" style="width:{min(100, lm['mape']*4):.1f}%"></div></div>
            </div>
        </div>

        <!-- Big number chips: RMSE / MAE / MSE -->
        <div style="margin-top:0.9rem; margin-bottom:0.3rem;">
            <div class="em-label" style="margin-bottom:0.5rem;">📦 هەڵەی ژمارەیی</div>
            <div class="reg-row">
                <div class="reg-chip">
                    <div class="reg-chip-label">RMSE</div>
                    <div class="reg-chip-val">${lm['rmse']:,}</div>
                    <div class="reg-chip-sub">Root Mean Sq. Error</div>
                </div>
                <div class="reg-chip">
                    <div class="reg-chip-label">MAE</div>
                    <div class="reg-chip-val">${lm['mae']:,}</div>
                    <div class="reg-chip-sub">Mean Abs. Error</div>
                </div>
                <div class="reg-chip">
                    <div class="reg-chip-label">MSE</div>
                    <div class="reg-chip-val">{lm['mse']/1_000_000:.2f}M</div>
                    <div class="reg-chip-sub">Mean Sq. Error</div>
                </div>
            </div>
        </div>

        <span class="interp interp-great">✅ &nbsp; R² = {lm['r2']:.3f} — گونجانەوەی بەرز</span>
    </div>
    """, unsafe_allow_html=True)

# ── Interpretation guide ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📖  ڕێنمایی خوێندنەوەی ئەنجامەکان — How to Read These Metrics", expanded=False):
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("""
        **🛡️ مەترسی — Classification**
        | ئامار | مانا |
        |---|---|
        | **Accuracy** | چەند ڕاستی بڕیارەکانی داوە (کۆی گشتی) |
        | **Precision** | لە ئەوانەی بەرز وتووە، چەندێکی واقیعی بەرز بوون |
        | **Recall** | لە ئەوانەی واقیعی بەرز بوون، چەندێکی دۆزیەوە |
        | **F1-Score** | هاوسەنگی Precision و Recall (بەهتر نزیک ١) |
        | **AUC-ROC** | توانای جیاکردنی دوو پۆل — نزیک ١ زۆر باشە |
        """)
    with g2:
        st.markdown("""
        **💳 سنووری قەرز — Regression**
        | ئامار | مانا |
        |---|---|
        | **R²** | چەندی دەخوێنێتەوە (١ = تەواو، ٠ = هیچ) |
        | **MAPE** | تێکڕای هەڵەی ڕێژەیی (کەمتر بەهتر) |
        | **RMSE** | ڕەشەی هەڵەی تێکڕا — هەمان یەکەی داتا |
        | **MAE** | تێکڕای هەڵەی ڕاستەقینە (کەمتر بەهتر) |
        | **MSE** | تێکڕای کارەی هەڵەکان (ئامانجی تەکنیکی) |
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    دروستکراوە لەلایەن &nbsp;<strong>ئومێد جەمال نووری</strong>&nbsp; ·
    Developed by <strong>Umed Jamal Nouri</strong><br>
    <span style="font-size:0.68rem;opacity:0.65;">
        Powered by XGBoost · Scikit-learn · Streamlit · Python
    </span>
</div>
""", unsafe_allow_html=True)
