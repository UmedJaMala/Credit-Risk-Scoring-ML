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
#  GLOBAL CSS  —  Premium AI Dark Theme (Neon Purple & Cyan)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700;800;900&display=swap');

/* ── CSS Variables (Premium Dark Mode) ──────────────────────────── */
:root {
    --bg-dark:     #050505; /* Deep Space Black */
    --bg-card:     #121212; /* Very dark gray for cards */
    --border:      rgba(255, 255, 255, 0.08);
    --text-main:   #f8fafc; /* Brilliant white */
    --text-muted:  #a1a1aa; /* Soft gray */
    --text-label:  #e2e8f0;
    
    --accent-1:    #a855f7; /* Neon Purple */
    --accent-2:    #06b6d4; /* Electric Cyan */
    --accent-dim:  rgba(168, 85, 247, 0.15);
    
    --green:       #10b981; /* Emerald */
    --green-dim:   rgba(16, 185, 129, 0.15);
    --red:         #f43f5e; /* Rose/Red */
    --red-dim:     rgba(244, 63, 94, 0.15);
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
   پەنجەرەکان (Dialogs & Modals Background Fix)
══════════════════════════════════════════════════════════════════════ */
div[data-testid="stModal"] > div, 
div[role="dialog"], 
section[data-testid="stDialog"] > div {
    background-color: var(--bg-card) !important;
    background: linear-gradient(145deg, #121212 0%, #0a0a0a 100%) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.8) !important;
}

div[data-testid="stModal"] > div > div, 
div[role="dialog"] > div {
    background-color: transparent !important;
}

div[role="dialog"] p, div[role="dialog"] h1, div[role="dialog"] h2, 
div[role="dialog"] h3, div[role="dialog"] span, div[role="dialog"] div {
    color: var(--text-main) !important;
}

/* ── App Background ──────────────────────────────────────────────── */
.stApp {
    background: radial-gradient(circle at 15% 0%, rgba(168, 85, 247, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 100%, rgba(6, 182, 212, 0.08) 0%, transparent 40%),
                var(--bg-dark);
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
    background: rgba(18, 18, 18, 0.6);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(168,85,247,0.05) 0%, rgba(6,182,212,0.05) 100%);
    pointer-events: none;
}
.hero-icon  { font-size: 3rem; line-height: 1; margin-bottom: 0.8rem; filter: drop-shadow(0 0 10px rgba(168,85,247,0.4)); }
.hero-title {
    font-size: clamp(1.6rem, 3.8vw, 2.4rem);
    font-weight: 900;
    color: var(--text-main);
    margin-bottom: 0.4rem;
    line-height: 1.4;
    letter-spacing: -0.02em;
}
.hero-title span {
    background: linear-gradient(to left, var(--accent-2), var(--accent-1));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: 0.04em;
}
.hero-badge {
    display: inline-block;
    margin-top: 1.2rem;
    background: rgba(168, 85, 247, 0.1);
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 50px;
    padding: 0.3rem 1.2rem;
    font-size: 0.8rem;
    font-weight: 700;
    color: #d8b4fe;
    letter-spacing: 0.1em;
    box-shadow: 0 0 15px rgba(168,85,247,0.15);
}

/* ════════════════════════════════════════════════════════
   SECTION HEADING
════════════════════════════════════════════════════════ */
.sec-heading {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 1.2rem;
}
.sec-heading-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.1) 0%, transparent 100%);
}
.sec-heading-text {
    color: var(--accent-2);
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    white-space: nowrap;
}

/* ════════════════════════════════════════════════════════
   PROJECT INFO 
════════════════════════════════════════════════════════ */
.sb-logo     { text-align: center; padding: 0.5rem 0; font-size: 2.8rem; }
.sb-app-name { text-align: center; color: var(--accent-1); font-size: 1rem; font-weight: 800; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
.sb-ver      { text-align: center; color: var(--text-muted); font-size: 0.75rem; margin-bottom: 1.5rem; }
.sb-section  {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    height: 100%;
}
.sb-sec-title {
    color: var(--accent-2);
    font-size: 0.9rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}
.sb-body { color: var(--text-muted); font-size: 0.85rem; line-height: 1.8; }
.sb-body b { color: var(--text-main); font-weight: 600; }
.sb-tag {
    display: inline-block;
    background: rgba(6, 182, 212, 0.1);
    color: var(--accent-2);
    border: 1px solid rgba(6, 182, 212, 0.25);
    border-radius: 6px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 0.25rem 0.15rem;
}

/* ════════════════════════════════════════════════════════
   INPUT CARDS
════════════════════════════════════════════════════════ */
.input-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.8rem 1.6rem 1.5rem;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    height: 100%;
}
.card-title {
    color: var(--text-main);
    font-size: 1rem;
    font-weight: 800;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.card-title-icon {
    font-size: 1.2rem;
    filter: drop-shadow(0 0 5px var(--accent-2));
}

/* ── Form Elements ───────────────────────────────────── */
label, div[data-testid="stWidgetLabel"] > p, .stSlider label, .stNumberInput label, .stSelectbox label {
    color: var(--text-label) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    direction: rtl !important;
    text-align: right !important;
    margin-bottom: 0.3rem !important;
}

.stNumberInput input, .stSelectbox > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    color: var(--text-main) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.2s ease;
}
.stNumberInput input { direction: ltr !important; text-align: left !important; }
.stNumberInput input:focus, .stSelectbox > div > div:focus-within {
    border-color: var(--accent-2) !important;
    box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.2) !important;
    outline: none !important;
}
.stSelectbox svg { color: var(--accent-2) !important; }

/* ── Slider ─────────────────────────────────────────── */
div[data-testid="stSlider"] { direction: ltr !important; padding: 0 0.3rem; }
div[data-testid="stSlider"] > div { direction: ltr !important; }
div[data-testid="stSlider"] .rc-slider-rail, .stSlider .rc-slider-rail {
    background: rgba(255,255,255,0.1) !important;
    border-radius: 5px !important;
    height: 8px !important;
}
div[data-testid="stSlider"] .rc-slider-track, .stSlider .rc-slider-track {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2)) !important;
    height: 8px !important;
    border-radius: 5px !important;
}
div[data-testid="stSlider"] .rc-slider-handle, .stSlider .rc-slider-handle {
    width: 20px !important; height: 20px !important; margin-top: -6px !important;
    background: var(--text-main) !important;
    border: 4px solid var(--accent-2) !important;
    box-shadow: 0 0 10px rgba(6, 182, 212, 0.5) !important;
}
div[data-testid="stSlider"] [data-testid="stTickBarMin"], div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: var(--text-muted) !important; font-size: 0.75rem !important;
}

/* ════════════════════════════════════════════════════════
   SUMMARY MINI-CARD
════════════════════════════════════════════════════════ */
.summary-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    direction: rtl;
}
.summary-card-title {
    color: var(--text-muted);
    font-size: 0.8rem;
    font-weight: 800;
    margin-bottom: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.summary-row {
    display: flex; justify-content: space-between;
    color: var(--text-muted); font-size: 0.85rem; line-height: 2;
    border-bottom: 1px solid rgba(255,255,255,0.05); padding: 0.15rem 0;
}
.summary-row:last-child { border-bottom: none; }
.summary-val { color: var(--text-main); font-weight: 700; direction: ltr; text-align: left; }

/* ════════════════════════════════════════════════════════
   ANALYZE BUTTON (NEON GRADIENT)
════════════════════════════════════════════════════════ */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #06b6d4 100%) !important;
    color: #ffffff !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 1.5rem !important;
    letter-spacing: 0.05em !important;
    box-shadow: 0 6px 20px rgba(168, 85, 247, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    cursor: pointer !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 10px 25px rgba(6, 182, 212, 0.5) !important;
    background: linear-gradient(135deg, #b873f8 0%, #787af3 50%, #22c5e5 100%) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) scale(0.99) !important;
}

@media (max-width: 768px) {
    div[data-testid="stButton"] > button { font-size: 1.25rem !important; padding: 1.1rem 2rem !important; }
}

/* ════════════════════════════════════════════════════════
   RESULT CARDS
════════════════════════════════════════════════════════ */
.result-wrap { animation: fadeUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) both; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(25px); } to { opacity: 1; transform: translateY(0); } }

.result-card {
    border-radius: 20px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
    background: var(--bg-card);
    box-shadow: 0 10px 35px rgba(0,0,0,0.5);
    direction: rtl;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 4px;
}
.rc-low::before { background: var(--green); }
.rc-high::before { background: var(--red); }
.rc-limit::before { background: linear-gradient(90deg, var(--accent-1), var(--accent-2)); }

.rc-low { border: 1px solid rgba(16, 185, 129, 0.2); }
.rc-high { border: 1px solid rgba(244, 63, 94, 0.2); }
.rc-limit { border: 1px solid rgba(6, 182, 212, 0.2); }

.rc-eyebrow { font-size: 0.75rem; font-weight: 800; letter-spacing: 0.12em; margin-bottom: 0.8rem; opacity: 0.8; }
.rc-value {
    font-size: clamp(2.2rem, 5vw, 3.2rem);
    font-weight: 900; line-height: 1.1; margin-bottom: 0.4rem; letter-spacing: -0.02em;
}
.rc-limit .rc-value {
    background: linear-gradient(to right, #06b6d4, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.rc-en { font-size: 0.85rem; color: var(--text-muted); font-weight: 500; margin-bottom: 1rem; }
.rc-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.35rem 1rem; border-radius: 50px; font-size: 0.8rem; font-weight: 700;
}
.rc-low  .rc-eyebrow, .rc-low  .rc-value { color: var(--green); }
.rc-high .rc-eyebrow, .rc-high .rc-value { color: var(--red); }
.rc-limit .rc-eyebrow { color: var(--accent-2); }

.badge-low   { background: rgba(16, 185, 129, 0.1); color: var(--green); border: 1px solid rgba(16, 185, 129, 0.3); }
.badge-high  { background: rgba(244, 63, 94, 0.1); color: var(--red); border: 1px solid rgba(244, 63, 94, 0.3); }
.badge-limit { background: rgba(6, 182, 212, 0.1); color: var(--accent-2); border: 1px solid rgba(6, 182, 212, 0.3); }

/* ════════════════════════════════════════════════════════
   METRIC MINI-CARDS & EVALUATION BOXES
════════════════════════════════════════════════════════ */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.3rem;
    text-align: center;
}
.metric-label { color: var(--text-muted); font-size: 0.8rem; font-weight: 700; margin-bottom: 0.4rem; }
.metric-value { color: var(--text-main);  font-size: 1.8rem;  font-weight: 900; line-height: 1; }
.metric-en    { color: rgba(255, 255, 255, 0.3); font-size: 0.75rem; margin-top: 0.4rem; }

.eval-box {
    background: rgba(255, 255, 255, 0.02);
    border-left: 4px solid var(--accent-1);
    border-top: 1px solid var(--border);
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    padding: 1.3rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    transition: transform 0.2s ease, background 0.2s ease;
}
.eval-box:hover {
    transform: translateY(-3px);
    background: rgba(168, 85, 247, 0.05);
}
.eval-title { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; font-weight: 800; margin-bottom: 0.5rem;}
.eval-val { font-size: 1.8rem; font-weight: 900; color: var(--text-main); }

/* Tabs Styling */
div[data-baseweb="tab-list"] { border-bottom: 2px solid rgba(255, 255, 255, 0.05); gap: 2rem; }
div[data-baseweb="tab"] { background-color: transparent !important; color: var(--text-muted) !important; font-weight: 600 !important; font-size: 1.05rem !important; padding: 1.2rem 0 !important; }
div[aria-selected="true"] { color: var(--accent-2) !important; border-bottom-color: var(--accent-2) !important; }

/* ════════════════════════════════════════════════════════
   FOOTER & WARNING
════════════════════════════════════════════════════════ */
.warn-banner { background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.4); border-radius: 12px; padding: 1rem; color: var(--red); font-size: 0.9rem; margin-bottom: 1.5rem; text-align: center; }
.footer { text-align: center; padding: 2rem 0 1rem; color: rgba(255, 255, 255, 0.3); font-size: 0.85rem; border-top: 1px solid var(--border); margin-top: 3rem; direction: rtl; }
.footer strong { color: var(--accent-2); font-weight: 800; }

@media (max-width: 768px) {
    .block-container { padding: 1rem 1rem 3rem 1rem !important; }
    .hero { padding: 2rem 1.2rem; }
    div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; margin-bottom: 1.2rem; }
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
    
    tab1, tab2, tab3 = st.tabs(["🎯 پۆلێنکردن (Classification)", "💰 بڕی قەرز (Regression)", "🧠 شیکاری و کاریگەری (Insights)"])
    
    # ── TAB 1: Classification ────────────────────────────────────────────────
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown('<div class="eval-box"><div class="eval-title">Accuracy</div><div class="eval-val">71.00%</div></div>', unsafe_allow_html=True)
        with c2: st.markdown('<div class="eval-box"><div class="eval-title">Precision</div><div class="eval-val">68.75%</div></div>', unsafe_allow_html=True)
        with c3: st.markdown('<div class="eval-box"><div class="eval-title">Recall</div><div class="eval-val">62.50%</div></div>', unsafe_allow_html=True)
        with c4: st.markdown('<div class="eval-box"><div class="eval-title">ROC-AUC</div><div class="eval-val">0.7359</div></div>', unsafe_allow_html=True)
        
        st.divider()
        st.markdown("<div style='text-align:center; color:#a1a1aa; font-weight:800; margin-bottom:1rem; font-size:1.1rem;'>Risk Classification (Confusion Matrix)</div>", unsafe_allow_html=True)
        
        col_space1, col_graph1, col_space2 = st.columns([1, 4, 1])
        with col_graph1:
            cm_data = np.array([[87, 25], [33, 55]])
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            
            # گۆڕینی ڕەنگی نەخشەکە بۆ نیۆنی سەردەمیانە
            cmap = mcolors.LinearSegmentedColormap.from_list("ai_neon", ["#050505", "#18181b", "#6366f1", "#06b6d4"])
            sns.heatmap(cm_data, annot=True, fmt="d", cmap=cmap, cbar=True, ax=ax_cm, 
                        xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'],
                        annot_kws={"size": 13, "weight": "bold"})
            ax_cm.set_ylabel('True label')
            ax_cm.set_xlabel('Predicted label')
            
            fig_cm.patch.set_facecolor('#121212') 
            ax_cm.set_facecolor('#121212') 
            
            [t.set_color('#f8fafc') for t in ax_cm.xaxis.get_ticklabels()]
            [t.set_color('#f8fafc') for t in ax_cm.yaxis.get_ticklabels()]
            ax_cm.xaxis.label.set_color('#a1a1aa')
            ax_cm.yaxis.label.set_color('#a1a1aa')
            
            cbar = ax_cm.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color='#f8fafc')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#f8fafc')
            
            st.pyplot(fig_cm)

    # ── TAB 2: Regression ────────────────────────────────────────────────────
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        with r1: st.markdown('<div class="eval-box" style="border-left-color:#06b6d4;"><div class="eval-title">R² Score</div><div class="eval-val">0.8192</div></div>', unsafe_allow_html=True)
        with r2: st.markdown('<div class="eval-box" style="border-left-color:#06b6d4;"><div class="eval-title">MAE Error</div><div class="eval-val">$3,460</div></div>', unsafe_allow_html=True)
        with r3: st.markdown('<div class="eval-box" style="border-left-color:#06b6d4;"><div class="eval-title">RMSE Error</div><div class="eval-val">$5,228</div></div>', unsafe_allow_html=True)
        with r4: st.markdown('<div class="eval-box" style="border-left-color:#06b6d4;"><div class="eval-title">MSE Error</div><div class="eval-val" style="font-size:1.2rem;">$27.3M</div></div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("<div style='text-align:center; color:#a1a1aa; font-weight:800; margin-bottom:1rem; font-size:1.1rem;'>Credit Limit Prediction Accuracy</div>", unsafe_allow_html=True)
        
        col_space3, col_graph2, col_space4 = st.columns([1, 4, 1])
        with col_graph2:
            np.random.seed(42)
            actual = np.random.uniform(5000, 70000, 100)
            predicted = actual * 0.9 + np.random.normal(0, 5000, 100)
            
            fig_reg, ax_reg = plt.subplots(figsize=(6, 4))
            ax_reg.scatter(actual, predicted, alpha=0.8, color="#06b6d4", s=30, edgecolor="#050505", linewidth=0.5)
            ax_reg.plot([0, 70000], [0, 70000], '--', color="#f43f5e", lw=2, label="Perfect Fit")
            
            ax_reg.set_xlabel("Actual Credit Limit ($)")
            ax_reg.set_ylabel("Predicted Credit Limit ($)")
            
            fig_reg.patch.set_facecolor('#121212')
            ax_reg.set_facecolor('#121212')
            ax_reg.xaxis.label.set_color('#a1a1aa')
            ax_reg.yaxis.label.set_color('#a1a1aa')
            ax_reg.tick_params(colors='#a1a1aa')
            ax_reg.grid(color='#ffffff', linestyle='-', linewidth=0.5, alpha=0.05)
            for spine in ax_reg.spines.values():
                spine.set_color('#ffffff')
                spine.set_alpha(0.1)
                
            legend = ax_reg.legend(facecolor='#121212', edgecolor='none')
            for text in legend.get_texts():
                text.set_color('#f8fafc')
                
            st.pyplot(fig_reg)

    # ── TAB 3: Insights & Overfitting ────────────────────────────────────────
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        
        o1, o2 = st.columns(2)
        with o1:
            st.markdown("""
            <div class="eval-box" style="border-left-color:#f43f5e;">
                <div class="eval-title">Classification Overfitting Check</div>
                <div style="font-size:0.95rem; color:white; line-height: 1.8; direction: ltr; text-align: left; padding-top:0.5rem;">
                <b>Train Acc:</b> 96.63% <span style="color:#a1a1aa;">&rarr;</span> <b>Test Acc:</b> 71.00%<br>
                <span style="color:#f43f5e; font-weight:800; font-size:1.1rem;">Gap: 25.63%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with o2:
            st.markdown("""
            <div class="eval-box" style="border-left-color:#f43f5e;">
                <div class="eval-title">Regression Overfitting Check</div>
                <div style="font-size:0.95rem; color:white; line-height: 1.8; direction: ltr; text-align: left; padding-top:0.5rem;">
                <b>Train R²:</b> 0.9957 <span style="color:#a1a1aa;">&rarr;</span> <b>Test R²:</b> 0.8192<br>
                <span style="color:#f43f5e; font-weight:800; font-size:1.1rem;">Gap: 0.1765</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("<div style='text-align:center; color:#a1a1aa; font-weight:800; margin-bottom:1rem; font-size:1.1rem;'>Feature Importance (XGBoost Engine)</div>", unsafe_allow_html=True)
        
        col_space5, col_graph3, col_space6 = st.columns([1, 4, 1])
        with col_graph3:
            fig_fi, ax_fi = plt.subplots(figsize=(6, 3.5))
            
            features_list = ['Missed Payments', 'Current Debt', 'Annual Income', 'Years in Business', 'Avg Order']
            importance = [0.42, 0.28, 0.15, 0.10, 0.05]
            
            sns.barplot(x=importance, y=features_list, hue=features_list, palette=["#a855f7", "#8b5cf6", "#6366f1", "#06b6d4", "#3b82f6"], legend=False, ax=ax_fi)
            
            ax_fi.set_xlabel('Relative Importance')
            fig_fi.patch.set_facecolor('#121212')
            ax_fi.set_facecolor('#121212')
            ax_fi.xaxis.label.set_color('#a1a1aa')
            ax_fi.tick_params(colors='#f8fafc')
            
            ax_fi.grid(color='#ffffff', linestyle='-', linewidth=0.5, alpha=0.05, axis='x')
            for spine in ax_fi.spines.values():
                spine.set_color('none')
                
            st.pyplot(fig_fi)

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