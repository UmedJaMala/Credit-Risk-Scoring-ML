import streamlit as st
import numpy as np
import joblib
import os

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG.
#  PAGE CONFIG.
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="سیستەمی زیرەکی نمرەدانی مەترسی و سنووری قەرز",
    page_icon="🏦",
    layout="wide"
    layout="wide"
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

html, body, [class*="css"] {
html, body, [class*="css"] {
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
    padding: 1.8rem 2rem 4rem 2rem !important;
    padding: 1.8rem 2rem 4rem 2rem !important;
    max-width: 1080px !important;
}
            
        /* چاککردنی ڕەنگی تێکستی ناو ئێکسپاندەر */
    .streamlit-expanderHeader p {
        color: #00d4ff !important; /* ڕەنگی تێڵ */
        font-weight: bold !important;
        font-size: 1.1rem !important;
    }
    /* بۆ ئەوەی لەسەر مۆبایل ئایکۆنەکەش ڕەنگی جوان بێت */
    .streamlit-expanderHeader svg {
        fill: #00d4ff !important;
    }

/* ════════════════════════════════════════════════════════
   HERO BANNER
════════════════════════════════════════════════════════ */
.hero {
    background: linear-gradient(120deg, rgba(0,45,75,0.9) 0%, rgba(0,62,102,0.85) 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2.2rem 2rem 2rem 2rem;
    padding: 2.2rem 2rem 2rem 2rem;
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
.hero-icon  { font-size: 2.6rem; line-height: 1; margin-bottom: 0.6rem; }
.hero-icon  { font-size: 2.6rem; line-height: 1; margin-bottom: 0.6rem; }
.hero-title {
    font-size: clamp(1.45rem, 3.5vw, 2.1rem);
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
    font-size: 0.9rem;
    font-weight: 400;
    letter-spacing: 0.02em;
}
.hero-badge {
    display: inline-block;
    margin-top: 0.9rem;
    background: var(--teal-dim);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 50px;
    padding: 0.22rem 1rem;
    font-size: 0.75rem;
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
   PROJECT INFO (Replaced Sidebar)
════════════════════════════════════════════════════════ */
.sb-logo     { text-align: center; padding: 0.5rem 0; font-size: 2.8rem; }
.sb-app-name { text-align: center; color: var(--teal); font-size: 0.9rem; font-weight: 700; letter-spacing: 0.06em; margin-bottom: 0.3rem; }
.sb-ver      { text-align: center; color: rgba(0,212,255,0.4); font-size: 0.75rem; margin-bottom: 1rem; }
.sb-section  {
    background: rgba(0,45,75,0.4);
    border: 1px solid rgba(0,212,255,0.1);
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
    border-bottom: 1px solid rgba(0,212,255,0.1);
    padding-bottom: 0.4rem;
}
.sb-body { color: rgba(180,210,235,0.8); font-size: 0.85rem; line-height: 1.8; }
.sb-body b { color: var(--text-main); font-weight: 600; }
.sb-tag {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    color: var(--teal);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 6px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 0.2rem 0.12rem;
}

/* ════════════════════════════════════════════════════════
   PROJECT INFO (Replaced Sidebar)
════════════════════════════════════════════════════════ */
.sb-logo     { text-align: center; padding: 0.5rem 0; font-size: 2.8rem; }
.sb-app-name { text-align: center; color: var(--teal); font-size: 0.9rem; font-weight: 700; letter-spacing: 0.06em; margin-bottom: 0.3rem; }
.sb-ver      { text-align: center; color: rgba(0,212,255,0.4); font-size: 0.75rem; margin-bottom: 1rem; }
.sb-section  {
    background: rgba(0,45,75,0.4);
    border: 1px solid rgba(0,212,255,0.1);
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
    border-bottom: 1px solid rgba(0,212,255,0.1);
    padding-bottom: 0.4rem;
}
.sb-body { color: rgba(180,210,235,0.8); font-size: 0.85rem; line-height: 1.8; }
.sb-body b { color: var(--text-main); font-weight: 600; }
.sb-tag {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    color: var(--teal);
    border: 1px solid rgba(0,212,255,0.2);
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
    background: linear-gradient(145deg, rgba(0,45,75,0.6) 0%, rgba(0,30,56,0.5) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem 1.5rem 1.3rem;
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
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
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
    padding: 1.8rem 1.7rem;
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
    font-size: clamp(2rem, 5vw, 2.9rem);
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
    padding: 1.1rem;
    padding: 1.1rem;
    text-align: center;
    backdrop-filter: blur(6px);
}
.metric-label { color: var(--text-muted); font-size: 0.76rem; font-weight: 600; margin-bottom: 0.35rem; }
.metric-value { color: var(--text-main);  font-size: 1.6rem;  font-weight: 800; line-height: 1; }
.metric-value { color: var(--text-main);  font-size: 1.6rem;  font-weight: 800; line-height: 1; }
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

/* ── Responsive Mobile Edits ──────────────────────────────────────── */
@media (max-width: 768px) {
    .block-container { padding: 1rem 1rem 3rem 1rem !important; }
/* ── Responsive Mobile Edits ──────────────────────────────────────── */
@media (max-width: 768px) {
    .block-container { padding: 1rem 1rem 3rem 1rem !important; }
    .hero { padding: 1.5rem 1rem; }
    .rc-value { font-size: 2.2rem; }
    .rc-value { font-size: 2.2rem; }
    
    /* مۆبایل: دوو کۆڵۆمەکە بکە بەیەک */
    /* مۆبایل: دوو کۆڵۆمەکە بکە بەیەک */
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
        margin-bottom: 1rem;
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
#  PROJECT INFO (Replaced Sidebar with Expander)
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("ℹ️ زانیاری زیاتر دەربارەی پڕۆژە و گەشەپێدەر", expanded=False):
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
                ئەم سیستەمە بەکاردەهێنێت زیرەکی دەستکرد بۆ ئەوەی ئاستی مەترسی
                کڕیارەکان بخەیتە سەر ئەستۆ و سنووری قەرزی گونجاو دیاری بکات
                بۆ کۆمپانیا و تازیرەکان.<br><br>
                بە بەکارهێنانی مۆدێلی <b>XGBoost</b>، سیستەمەکە
                زانیارییەکانی دارایی و بازرگانی شیکاری دەکاتەوە
                و بڕیاری زیرەکانە دەدات.
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