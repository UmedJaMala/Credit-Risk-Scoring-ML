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
    initial_sidebar_state="auto", # ئەمە وادەکات لەسەر مۆبایل داخراو بێت و لەسەر کۆمپیوتەر کراوە بێت
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  —  Navy / Teal Banking Theme  +  RTL  +  Noto Sans Arabic
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700;800;900&display=swap');

/* ── CSS Variables ───────────────────────────────────────────────── */
:root {
    --navy:        #002d4b;
    --navy-mid:    #003d66;
    --navy-light:  #004e80;
    --teal:        #00d4ff;
    --teal-dim:    rgba(0, 212, 255, 0.18);
    --teal-glow:   rgba(0, 212, 255, 0.08);
    --slate:       #0a1f35;
    --slate-card:  rgba(0, 45, 75, 0.55);
    --border:      rgba(0, 212, 255, 0.18);
    --text-main:   #e8f4fd;
    --text-muted:  rgba(180, 210, 235, 0.65);
    --text-label:  #7ecfee;
    --green:       #00e5a0;
    --green-dim:   rgba(0, 229, 160, 0.12);
    --red:         #ff4d6d;
    --red-dim:     rgba(255, 77, 109, 0.12);
    --gold:        #ffc857;
}

/* ── Reset & Base ────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,[class*="css"] {
    font-family: 'Noto Sans Arabic', sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
}

/* ── App Background ──────────────────────────────────────────────── */
.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(0,212,255,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 100%, rgba(0,78,128,0.25) 0%, transparent 55%),
        linear-gradient(160deg, #011928 0%, #001e38 40%, #00111f 100%);
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ───────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.8rem 1rem 4rem 1rem !important; /* پادینگی کەمکرایەوە بۆ مۆبایل */
    max-width: 1080px !important;
}

/* ════════════════════════════════════════════════════════
   HERO BANNER
════════════════════════════════════════════════════════ */
.hero {
    background: linear-gradient(120deg, rgba(0,45,75,0.9) 0%, rgba(0,62,102,0.85) 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2rem 1rem; /* پادینگی کەمکرایەوە */
    margin-bottom: 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(0,212,255,0.1);
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% -20%, rgba(0,212,255,0.12) 0%, transparent 65%);
    pointer-events: none;
}
.hero-icon  { font-size: 2rem; line-height: 1; margin-bottom: 0.5rem; }
.hero-title {
    font-size: clamp(1.2rem, 3.5vw, 2.1rem); /* بچووککرایەوە بۆ مۆبایل */
    font-weight: 900;
    color: var(--text-main);
    margin-bottom: 0.35rem;
    line-height: 1.45;
    letter-spacing: -0.01em;
}
.hero-title span { color: var(--teal); }
.hero-sub {
    color: var(--text-muted);
    font-size: 0.8rem;
    font-weight: 400;
    letter-spacing: 0.02em;
}
.hero-badge {
    display: inline-block;
    margin-top: 0.9rem;
    background: var(--teal-dim);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 50px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
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
   INPUT CARDS
════════════════════════════════════════════════════════ */
.input-card {
    background: linear-gradient(145deg, rgba(0,45,75,0.6) 0%, rgba(0,30,56,0.5) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem 1rem; /* پادینگی کەمکرایەوە بۆ مۆبایل */
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
    border-bottom: 1px solid rgba(0,212,255,0.12);
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.card-title-icon {
    font-size: 1.1rem;
    filter: drop-shadow(0 0 6px rgba(0,212,255,0.5));
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
    background: rgba(0,30,56,0.8) !important;
    border: 1.5px solid rgba(0,212,255,0.25) !important;
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
    box-shadow: 0 0 0 3px rgba(0,212,255,0.12) !important;
    outline: none !important;
}

/* ── Selectbox ───────────────────────────────────────── */
.stSelectbox > div > div {
    background: rgba(0,30,56,0.8) !important;
    border: 1.5px solid rgba(0,212,255,0.25) !important;
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
    background: rgba(0,212,255,0.15) !important;
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
    border: 3px solid #001e38 !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.3), 0 2px 8px rgba(0,0,0,0.4) !important;
    border-radius: 50% !important;
    transition: box-shadow 0.15s ease;
}
div[data-testid="stSlider"] .rc-slider-handle:hover,
.stSlider .rc-slider-handle:hover {
    box-shadow: 0 0 0 5px rgba(0,212,255,0.4), 0 2px 8px rgba(0,0,0,0.4) !important;
}
div[data-testid="stSlider"][data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
}

/* ════════════════════════════════════════════════════════
   SUMMARY MINI-CARD
════════════════════════════════════════════════════════ */
.summary-card {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.14);
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
    border-bottom: 1px solid rgba(0,212,255,0.07);
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
   ANALYZE BUTTON
════════════════════════════════════════════════════════ */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(130deg, #005580 0%, #007aab 45%, #00aad4 100%) !important;
    color: #ffffff !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 4px 22px rgba(0,170,212,0.35) !important;
    transition: all 0.22s ease !important;
    cursor: pointer !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,170,212,0.5) !important;
    background: linear-gradient(130deg, #006699 0%, #0090c0 45%, #00c0e8 100%) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
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
    padding: 1.5rem; /* بچووککرایەوە بۆ مۆبایل */
    position: relative;
    overflow: hidden;
    box-shadow: 0 6px 32px rgba(0,0,0,0.3);
    direction: rtl;
    margin-bottom: 1rem; /* زیادکرا بۆ ئەوەی لەسەر مۆبایل لێک جیابنەوە */
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
    background: linear-gradient(135deg, rgba(0,30,56,0.95) 0%, rgba(0,60,40,0.6) 100%);
    border: 1.5px solid rgba(0,229,160,0.4);
    box-shadow: 0 6px 32px rgba(0,229,160,0.12);
}
/* HIGH RISK */
.rc-high {
    background: linear-gradient(135deg, rgba(0,30,56,0.95) 0%, rgba(70,10,25,0.65) 100%);
    border: 1.5px solid rgba(255,77,109,0.45);
    box-shadow: 0 6px 32px rgba(255,77,109,0.14);
}
/* CREDIT LIMIT */
.rc-limit {
    background: linear-gradient(135deg, rgba(0,30,56,0.95) 0%, rgba(0,60,100,0.65) 100%);
    border: 1.5px solid rgba(0,212,255,0.4);
    box-shadow: 0 6px 32px rgba(0,212,255,0.12);
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
    font-size: clamp(1.8rem, 5vw, 2.9rem); /* بچووککرایەوە بۆ مۆبایل */
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
.rc-limit .rc-eyebrow { color: var(--teal); }
.rc-limit .rc-value   { color: #7de8ff; }
.rc-limit .rc-en      { color: var(--text-muted); }
.badge-low   { background: rgba(0,229,160,0.15);  color: var(--green); border: 1px solid rgba(0,229,160,0.3); }
.badge-high  { background: rgba(255,77,109,0.15); color: var(--red);   border: 1px solid rgba(255,77,109,0.3); }
.badge-limit { background: var(--teal-dim);       color: var(--teal);  border: 1px solid rgba(0,212,255,0.3); }

/* ════════════════════════════════════════════════════════
   METRIC MINI-CARDS
════════════════════════════════════════════════════════ */
.metric-card {
    background: rgba(0,45,75,0.45);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
    backdrop-filter: blur(6px);
    margin-bottom: 10px; /* زیادکرا بۆ مۆبایل */
}
.metric-label { color: var(--text-muted); font-size: 0.76rem; font-weight: 600; margin-bottom: 0.35rem; }
.metric-value { color: var(--text-main);  font-size: 1.4rem;  font-weight: 800; line-height: 1; }
.metric-en    { color: rgba(130,170,200,0.45); font-size: 0.7rem; margin-top: 0.25rem; }

/* ════════════════════════════════════════════════════════
   WARNING BANNER
════════════════════════════════════════════════════════ */
.warn-banner {
    background: rgba(255,200,87,0.08);
    border: 1px solid rgba(255,200,87,0.3);
    border-radius: 11px;
    padding: 0.75rem 1.1rem;
    color: var(--gold);
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
    color: rgba(120,160,190,0.4);
    font-size: 0.78rem;
    border-top: 1px solid rgba(0,212,255,0.08);
    margin-top: 2.5rem;
    direction: rtl;
}
.footer strong { color: var(--teal); font-weight: 700; }

/* ════════════════════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #001626 0%, #002038 60%, #001828 100%) !important;
    border-left: 1px solid rgba(0,212,255,0.15) !important;
    direction: rtl !important;
}
section[data-testid="stSidebar"] * {
    font-family: 'Noto Sans Arabic', sans-serif !important;
    direction: rtl !important;
    text-align: right !important;
}
.sb-logo     { text-align: center; padding: 1.2rem 0 0.5rem; font-size: 2.8rem; }
.sb-app-name { text-align: center; color: var(--teal); font-size: 0.82rem; font-weight: 700; letter-spacing: 0.06em; margin-bottom: 0.3rem; }
.sb-ver      { text-align: center; color: rgba(0,212,255,0.35); font-size: 0.7rem; margin-bottom: 1.4rem; }
.sb-divider  { height: 1px; background: linear-gradient(90deg, transparent, rgba(0,212,255,0.2), transparent); margin: 1rem 0; }
.sb-section  {
    background: rgba(0,45,75,0.4);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 13px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
}
.sb-sec-title {
    color: var(--teal);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}
.sb-body { color: rgba(180,210,235,0.72); font-size: 0.8rem; line-height: 1.75; }
.sb-body b { color: var(--text-main); font-weight: 600; }
.sb-tag {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    color: var(--teal);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 6px;
    padding: 0.18rem 0.65rem;
    font-size: 0.72rem;
    font-weight: 700;
    margin: 0.2rem 0.12rem;
}

/* ── Responsive ──────────────────────────────────────── */
@media (max-width: 640px) {
    .block-container { padding: 1rem 0.9rem 3rem !important; }
    .hero { padding: 1.5rem 1rem; }
    .rc-value { font-size: 2rem; }
    
    /* ڕێکخستنی ئینپوتەکان بۆ مۆبایل */
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    
    .input-card { margin-bottom: 1rem; }
}
</style>
""", unsafe_allow_html=True)
# باقی کۆدەکە وەک خۆیەتی بەبێ گۆڕانکاری