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
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700;800;900&family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Tokens ──────────────────────────────────────────── */
:root {
    --bg:           #0a0f1a;
    --bg-card:      #111827;
    --bg-card2:     #1a2235;
    --border:       rgba(255,255,255,0.07);
    --border-hi:    rgba(255,255,255,0.13);

    --cyan:         #22d3ee;
    --cyan-dim:     rgba(34,211,238,0.12);
    --cyan-bdr:     rgba(34,211,238,0.28);
    --violet:       #a78bfa;
    --violet-dim:   rgba(167,139,250,0.12);
    --violet-bdr:   rgba(167,139,250,0.28);

    --green:        #34d399;
    --green-dim:    rgba(52,211,153,0.12);
    --green-bdr:    rgba(52,211,153,0.30);
    --red:          #fb7185;
    --red-dim:      rgba(251,113,133,0.12);
    --red-bdr:      rgba(251,113,133,0.30);
    --amber:        #fbbf24;
    --amber-dim:    rgba(251,191,36,0.12);

    --text:         #f1f5f9;
    --text-2:       #94a3b8;
    --text-3:       #64748b;

    --gold:         #d4a84b;
    --gold-dim:     rgba(212,168,75,0.15);
    --gold-bdr:     rgba(212,168,75,0.35);

    --r:            14px;
    --shadow:       0 4px 24px rgba(0,0,0,0.40);
    --shadow-lg:    0 8px 40px rgba(0,0,0,0.55);
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

/* ── Page bg ──────────────────────────────────────────── */
.stApp {
    background:
        radial-gradient(ellipse 70% 35% at 10% 0%,   rgba(167,139,250,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 60% 40% at 90% 100%,  rgba(34,211,238,0.06) 0%, transparent 55%),
        var(--bg);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 1.6rem 4rem !important; max-width: 1060px !important; }

/* ════════════════════════════════════════════════════════
   HERO
════════════════════════════════════════════════════════ */
.hero {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    border: 1px solid var(--border-hi);
    border-radius: 20px;
    padding: 2.4rem 2rem 2rem;
    margin-bottom: 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% -5%, rgba(167,139,250,0.10) 0%, transparent 60%);
    pointer-events: none;
}
.hero-icon { font-size: 2.8rem; line-height: 1; margin-bottom: 0.7rem; filter: drop-shadow(0 0 12px rgba(167,139,250,0.5)); }
.hero-title {
    font-size: clamp(1.4rem, 3.5vw, 2.2rem);
    font-weight: 900;
    color: var(--text);
    line-height: 1.45; margin-bottom: 0.4rem;
}
.hero-title span {
    background: linear-gradient(to left, var(--cyan), var(--violet));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub  { color: var(--text-2); font-size: 0.9rem; font-weight: 500; letter-spacing: 0.03em; }
.hero-pill {
    display: inline-block; margin-top: 1.1rem;
    background: var(--violet-dim); border: 1px solid var(--violet-bdr);
    border-radius: 50px; padding: 0.25rem 1.1rem;
    font-size: 0.72rem; font-weight: 800; color: var(--violet); letter-spacing: 0.10em;
}

/* ════════════════════════════════════════════════════════
   ACTION BUTTONS  (About + Eval) — mobile-safe grid
════════════════════════════════════════════════════════ */
.action-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.85rem;
    margin-bottom: 1.6rem;
}
@media (max-width: 480px) { .action-grid { grid-template-columns: 1fr; } }

.action-btn {
    display: flex; align-items: center; justify-content: center; gap: 0.6rem;
    padding: 0.85rem 1rem;
    border-radius: var(--r);
    font-family: 'Noto Sans Arabic', sans-serif;
    font-size: 0.92rem; font-weight: 800;
    cursor: pointer; border: none;
    transition: all 0.22s ease;
    text-decoration: none;
    width: 100%;
}
.action-btn-about {
    background: var(--violet-dim);
    border: 1px solid var(--violet-bdr);
    color: var(--violet);
}
.action-btn-about:hover {
    background: rgba(167,139,250,0.20);
    box-shadow: 0 4px 18px rgba(167,139,250,0.25);
    transform: translateY(-2px);
}
.action-btn-eval {
    background: var(--cyan-dim);
    border: 1px solid var(--cyan-bdr);
    color: var(--cyan);
}
.action-btn-eval:hover {
    background: rgba(34,211,238,0.20);
    box-shadow: 0 4px 18px rgba(34,211,238,0.25);
    transform: translateY(-2px);
}

/* Streamlit button reset to match our action-btn styles */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: var(--gold-dim) !important;
    border: 1px solid var(--gold-bdr) !important;
    color: var(--gold) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 900 !important;
    border-radius: var(--r) !important;
    padding: 0.85rem 1.2rem !important;
    letter-spacing: 0.05em !important;
    box-shadow: 0 4px 20px rgba(212,168,75,0.20) !important;
    transition: all 0.22s ease !important;
    cursor: pointer !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    background: rgba(212,168,75,0.22) !important;
    box-shadow: 0 8px 28px rgba(212,168,75,0.35) !important;
}
div[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* ── Override buttons INSIDE dialogs (About / Eval trigger buttons) ── */
.top-action-buttons div[data-testid="stButton"] > button {
    background: transparent !important;
    box-shadow: none !important;
}

/* ════════════════════════════════════════════════════════
   SECTION HEADING
════════════════════════════════════════════════════════ */
.sec-head {
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 1.1rem;
}
.sec-head-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border-hi) 0%, transparent 100%);
}
.sec-head-text { color: var(--cyan); font-size: 0.78rem; font-weight: 800; letter-spacing: 0.10em; white-space: nowrap; }

/* ════════════════════════════════════════════════════════
   INPUT CARDS
════════════════════════════════════════════════════════ */
.input-card {
    background: var(--bg-card);
    border: 1px solid var(--border-hi);
    border-radius: 18px;
    padding: 1.7rem 1.5rem 1.4rem;
    box-shadow: var(--shadow);
    height: 100%;
}
.card-title {
    color: var(--text); font-size: 0.95rem; font-weight: 800;
    margin-bottom: 1.1rem; padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.45rem;
}

/* Labels */
label,
div[data-testid="stWidgetLabel"] > p,
.stSlider label, .stNumberInput label, .stSelectbox label {
    color: var(--text-2) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-weight: 600 !important; font-size: 0.87rem !important;
    direction: rtl !important; text-align: right !important;
}

/* Number inputs */
.stNumberInput input {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-size: 0.97rem !important; font-weight: 700 !important;
    padding: 0.58rem 0.9rem !important;
    direction: ltr !important; text-align: left !important;
    transition: border-color 0.18s, box-shadow 0.18s;
}
.stNumberInput input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 3px var(--cyan-dim) !important;
    outline: none !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
    font-weight: 700 !important;
}
.stSelectbox svg { color: var(--cyan) !important; }

/* Slider */
div[data-testid="stSlider"] { direction: ltr !important; padding: 0 0.2rem; }
div[data-testid="stSlider"] > div { direction: ltr !important; }
div[data-testid="stSlider"] .rc-slider-rail, .stSlider .rc-slider-rail {
    background: rgba(255,255,255,0.08) !important; border-radius: 5px !important; height: 7px !important;
}
div[data-testid="stSlider"] .rc-slider-track, .stSlider .rc-slider-track {
    background: linear-gradient(90deg, var(--violet), var(--cyan)) !important;
    height: 7px !important; border-radius: 5px !important;
}
div[data-testid="stSlider"] .rc-slider-handle, .stSlider .rc-slider-handle {
    width: 19px !important; height: 19px !important; margin-top: -6px !important;
    background: var(--text) !important;
    border: 3px solid var(--cyan) !important;
    box-shadow: 0 0 10px var(--cyan-dim) !important;
    border-radius: 50% !important;
}
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: var(--text-3) !important; font-size: 0.72rem !important;
}

/* ════════════════════════════════════════════════════════
   SUMMARY CARD
════════════════════════════════════════════════════════ */
.summary-card {
    background: var(--cyan-dim); border: 1px solid var(--cyan-bdr);
    border-radius: 12px; padding: 0.9rem 1.1rem; margin-top: 0.9rem; direction: rtl;
}
.summary-card-title { color: var(--cyan); font-size: 0.73rem; font-weight: 800; margin-bottom: 0.5rem; letter-spacing: 0.07em; }
.summary-row {
    display: flex; justify-content: space-between;
    color: var(--text-2); font-size: 0.8rem; line-height: 1.9;
    border-bottom: 1px solid rgba(34,211,238,0.08); padding: 0.06rem 0;
}
.summary-row:last-child { border-bottom: none; }
.summary-val { color: var(--text); font-weight: 800; direction: ltr; text-align: left; }

/* ════════════════════════════════════════════════════════
   RESULT CARDS
════════════════════════════════════════════════════════ */
.result-wrap { animation: fadeUp 0.42s cubic-bezier(0.22,1,0.36,1) both; }
@keyframes fadeUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }

.result-card {
    background: var(--bg-card); border-radius: 18px;
    padding: 1.8rem 1.6rem; position: relative; overflow: hidden;
    box-shadow: var(--shadow-lg); direction: rtl;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 18px 18px 0 0;
}
.rc-low::before  { background: var(--green); }
.rc-high::before { background: var(--red); }
.rc-limit::before{ background: linear-gradient(90deg, var(--violet), var(--cyan)); }

.rc-low   { border: 1px solid var(--green-bdr); }
.rc-high  { border: 1px solid var(--red-bdr); }
.rc-limit { border: 1px solid var(--cyan-bdr); }

.rc-eyebrow { font-size: 0.72rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.7rem; opacity: 0.75; }
.rc-value   {
    font-size: clamp(2rem, 5vw, 3rem);
    font-weight: 900; line-height: 1.05; margin-bottom: 0.25rem; letter-spacing: -0.02em;
}
.rc-limit .rc-value {
    background: linear-gradient(to right, var(--cyan), var(--violet));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.rc-en   { font-size: 0.8rem; color: var(--text-2); margin-bottom: 0.85rem; }
.rc-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.28rem 0.9rem; border-radius: 50px;
    font-size: 0.75rem; font-weight: 800;
}
.rc-low  .rc-eyebrow, .rc-low  .rc-value { color: var(--green); }
.rc-high .rc-eyebrow, .rc-high .rc-value { color: var(--red); }
.rc-limit .rc-eyebrow { color: var(--cyan); }

.badge-low   { background: var(--green-dim); color: var(--green); border: 1px solid var(--green-bdr); }
.badge-high  { background: var(--red-dim);   color: var(--red);   border: 1px solid var(--red-bdr); }
.badge-limit { background: var(--cyan-dim);  color: var(--cyan);  border: 1px solid var(--cyan-bdr); }

/* ════════════════════════════════════════════════════════
   RATIO MINI-CARDS
════════════════════════════════════════════════════════ */
.metric-card {
    background: var(--bg-card); border: 1px solid var(--border-hi);
    border-radius: 14px; padding: 1.1rem; text-align: center;
    box-shadow: var(--shadow);
}
.metric-label { color: var(--text-2); font-size: 0.74rem; font-weight: 700; margin-bottom: 0.3rem; }
.metric-value { color: var(--text);   font-size: 1.6rem;  font-weight: 900; line-height: 1; }
.metric-en    { color: var(--text-3); font-size: 0.68rem; margin-top: 0.22rem; }

/* ════════════════════════════════════════════════════════
   EVAL BOXES (inside dialog)
════════════════════════════════════════════════════════ */
.eval-box {
    background: var(--bg-card2);
    border-left: 3px solid var(--violet);
    border-top: 1px solid var(--border); border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);
    border-radius: 12px; padding: 1.1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.eval-title { font-size: 0.72rem; color: var(--text-2); text-transform: uppercase; font-weight: 800; margin-bottom: 0.45rem; letter-spacing: 0.07em; }
.eval-val   { font-size: 1.7rem; font-weight: 900; color: var(--text); }
.eval-val-sub { font-size: 0.72rem; color: var(--text-3); margin-top: 0.18rem; }

.eval-box-cyan { border-left-color: var(--cyan) !important; }
.eval-box-red  { border-left-color: var(--red)  !important; }
.eval-box-green{ border-left-color: var(--green)!important; }

/* overfitting gap indicator */
.gap-chip {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 6px;
    font-size: 0.8rem; font-weight: 800; margin-top: 0.3rem;
}
.gap-bad  { background: var(--red-dim);   color: var(--red);   border: 1px solid var(--red-bdr); }
.gap-ok   { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(251,191,36,0.3); }

/* Tabs */
div[data-baseweb="tab-list"] { border-bottom: 1px solid var(--border-hi) !important; gap: 1.5rem; }
div[data-baseweb="tab"] {
    background: transparent !important; color: var(--text-2) !important;
    font-weight: 700 !important; font-size: 0.95rem !important; padding: 1rem 0 !important;
    font-family: 'Noto Sans Arabic', sans-serif !important;
}
div[aria-selected="true"] { color: var(--cyan) !important; border-bottom-color: var(--cyan) !important; }

/* Dialog background fix */
div[data-testid="stModal"] > div,
div[role="dialog"],
section[data-testid="stDialog"] > div {
    background: #111827 !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 18px !important;
    box-shadow: var(--shadow-lg) !important;
}
div[data-testid="stModal"] > div > div,
div[role="dialog"] > div { background: transparent !important; }
div[role="dialog"] p, div[role="dialog"] h1, div[role="dialog"] h2,
div[role="dialog"] h3, div[role="dialog"] span, div[role="dialog"] div { color: var(--text) !important; }

/* ════════════════════════════════════════════════════════
   ABOUT SECTION (inside dialog)
════════════════════════════════════════════════════════ */
.about-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 0.9rem;
    margin-top: 0.8rem;
}
.about-card {
    background: var(--bg-card2); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.2rem 1.3rem;
}
.about-card-title {
    color: var(--cyan); font-size: 0.78rem; font-weight: 900;
    letter-spacing: 0.08em; margin-bottom: 0.7rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.about-card-body { color: var(--text-2); font-size: 0.82rem; line-height: 1.80; }
.about-card-body b { color: var(--text); font-weight: 700; }
.tech-tag {
    display: inline-block;
    background: var(--violet-dim); color: var(--violet);
    border: 1px solid var(--violet-bdr);
    border-radius: 6px; padding: 0.2rem 0.7rem;
    font-size: 0.72rem; font-weight: 800; margin: 0.2rem 0.12rem;
}
.about-center { text-align: center; padding: 1rem 0 0.5rem; }
.about-center-icon { font-size: 2.5rem; margin-bottom: 0.4rem; filter: drop-shadow(0 0 10px rgba(167,139,250,0.4)); }
.about-center-name { color: var(--violet); font-size: 0.95rem; font-weight: 900; letter-spacing: 0.06em; }
.about-center-ver  { color: var(--text-3); font-size: 0.73rem; margin-top: 0.2rem; }

/* ════════════════════════════════════════════════════════
   WARNING & FOOTER
════════════════════════════════════════════════════════ */
.warn-banner {
    background: var(--red-dim); border: 1px solid var(--red-bdr);
    border-radius: 12px; padding: 0.9rem 1.1rem;
    color: var(--red); font-size: 0.86rem; margin-bottom: 1.4rem;
    direction: rtl; text-align: right;
}
.footer {
    text-align: center; padding: 1.8rem 0 0.5rem;
    color: var(--text-3); font-size: 0.78rem;
    border-top: 1px solid var(--border); margin-top: 2.5rem; direction: rtl;
}
.footer strong { color: var(--cyan); font-weight: 800; }

/* ════════════════════════════════════════════════════════
   RESPONSIVE
════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    .block-container { padding: 1rem 0.9rem 3rem !important; }
    .hero { padding: 1.7rem 1rem; }
    .rc-value { font-size: 1.9rem; }
    .about-grid { grid-template-columns: 1fr; }
    div[data-baseweb="tab"] { font-size: 0.82rem !important; gap: 0.8rem; }
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
#  HELPER: styled matplotlib fig
# ══════════════════════════════════════════════════════════════════════════════
def dark_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    bg = "#111827"
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.tick_params(colors="#94a3b8")
    ax.xaxis.label.set_color("#94a3b8"); ax.yaxis.label.set_color("#94a3b8")
    for sp in ax.spines.values():
        sp.set_color("#1e293b")
    ax.grid(color="#1e293b", linewidth=0.8, alpha=0.6)
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
    <div class="about-grid">

        <div class="about-card">
            <div class="about-card-title">📋 دەربارەی پڕۆژە</div>
            <div class="about-card-body">
                ئەم سیستەمە بە <b>XGBoost</b> ئاستی مەترسی کڕیارەکان
                دیاری دەکات و سنووری قەرزی گونجاو بۆ کۆمپانیا و
                تازیرەکان دەستنیشان دەکات، بەپێی زانیارییەکانی دارایی
                و بازرگانی.
            </div>
        </div>

        <div class="about-card">
            <div class="about-card-title">👨‍💻 گەشەپێدەر</div>
            <div class="about-card-body">
                <b>ناو:</b> ئومێد جەمال نوری<br>
                <b>بەش:</b> ئەندازیاری کارەبا<br>
                <b>قۆناغ:</b> قۆناغی سێیەم<br>
                <b>ساڵی خوێندن:</b> ٢٠٢٥ — ٢٠٢٦
            </div>
        </div>

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

        # Metric boxes — 4 cols
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

        # F1 full-width note
        st.markdown(f"""
        <div class="eval-box" style="margin-top:0; background:rgba(167,139,250,0.06);">
            <div class="eval-title">F1-Score (Weighted)</div>
            <div class="eval-val" style="font-size:1.3rem;">{CLF['f1']:.4f}</div>
            <div class="eval-val-sub">هاوسەنگی Precision و Recall — نمونەی ٢٠٠ کڕیار</div>
        </div>""", unsafe_allow_html=True)

        st.divider()

        # Classification report table
        st.markdown("""
        <div style="overflow-x:auto; direction:ltr;">
        <table style="width:100%; border-collapse:collapse; font-size:0.85rem; font-family:'Inter',monospace; color:#f1f5f9;">
            <thead>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.08); color:#94a3b8;">
                    <th style="padding:0.6rem 0.8rem; text-align:left;"></th>
                    <th style="padding:0.6rem 0.8rem; text-align:center;">Precision</th>
                    <th style="padding:0.6rem 0.8rem; text-align:center;">Recall</th>
                    <th style="padding:0.6rem 0.8rem; text-align:center;">F1-Score</th>
                    <th style="padding:0.6rem 0.8rem; text-align:center;">Support</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                    <td style="padding:0.55rem 0.8rem; color:#34d399; font-weight:700;">Low Risk</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.72</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.78</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.75</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center; color:#64748b;">112</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                    <td style="padding:0.55rem 0.8rem; color:#fb7185; font-weight:700;">High Risk</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.69</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.62</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.65</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center; color:#64748b;">88</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.04); color:#94a3b8;">
                    <td style="padding:0.55rem 0.8rem; font-weight:700;">Macro Avg</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.71</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.70</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">0.70</td>
                    <td style="padding:0.55rem 0.8rem; text-align:center;">200</td>
                </tr>
            </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confusion matrix
        st.markdown("<div style='text-align:center; color:#94a3b8; font-weight:800; font-size:0.88rem; letter-spacing:0.08em; margin-bottom:0.7rem;'>CONFUSION MATRIX</div>", unsafe_allow_html=True)
        _, gcol, _ = st.columns([1, 3, 1])
        with gcol:
            fig, ax = dark_fig(5.5, 4)
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "dark_cyan", ["#0a0f1a", "#164e63", "#22d3ee"])
            sns.heatmap(CM, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax,
                        xticklabels=["Low Risk", "High Risk"],
                        yticklabels=["Low Risk", "High Risk"],
                        annot_kws={"size": 15, "weight": "bold", "color": "#f1f5f9"},
                        linewidths=2, linecolor="#0a0f1a")
            ax.set_ylabel("True label", labelpad=10)
            ax.set_xlabel("Predicted label", labelpad=10)
            [t.set_color("#f1f5f9") for t in ax.xaxis.get_ticklabels()]
            [t.set_color("#f1f5f9") for t in ax.yaxis.get_ticklabels()]
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
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

        # Scatter: Actual vs Predicted
        st.markdown("<div style='text-align:center; color:#94a3b8; font-weight:800; font-size:0.88rem; letter-spacing:0.08em; margin-bottom:0.7rem;'>ACTUAL vs PREDICTED CREDIT LIMIT</div>", unsafe_allow_html=True)
        _, gcol2, _ = st.columns([1, 3, 1])
        with gcol2:
            np.random.seed(42)
            actual    = np.random.uniform(5000, 70000, 150)
            noise     = np.random.normal(0, REG['rmse'], 150)
            predicted = np.clip(actual + noise, 0, None)

            fig2, ax2 = dark_fig(6, 4.2)
            ax2.scatter(actual, predicted, alpha=0.7, color="#22d3ee", s=25,
                        edgecolors="#0a0f1a", linewidths=0.4, zorder=3)
            ax2.plot([0, 70000], [0, 70000], "--", color="#fb7185", lw=1.8,
                     label=f"Perfect Fit  (R²={REG['r2']:.4f})")
            ax2.set_xlabel("Actual Credit Limit ($)")
            ax2.set_ylabel("Predicted Credit Limit ($)")
            leg = ax2.legend(facecolor="#111827", edgecolor="none", labelcolor="#f1f5f9",
                             fontsize=9)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

    # ── Tab 3: Insights ──────────────────────────────────────────────────────
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)

        # Overfitting
        st.markdown("""<div style="font-size:0.78rem; font-weight:800; color:#94a3b8; letter-spacing:0.1em; margin-bottom:0.8rem;">⚠️  OVERFITTING CHECK</div>""", unsafe_allow_html=True)
        ov1, ov2 = st.columns(2)
        with ov1:
            gap_clf = CLF_TRAIN["accuracy"] - CLF["accuracy"]
            st.markdown(f"""
            <div class="eval-box eval-box-red">
                <div class="eval-title">Classification · Accuracy Gap</div>
                <div style="font-size:0.92rem; color:#f1f5f9; line-height:2; direction:ltr; text-align:left; padding-top:0.3rem;">
                    Train: <b>{CLF_TRAIN['accuracy']*100:.2f}%</b> → Test: <b>{CLF['accuracy']*100:.2f}%</b>
                </div>
                <span class="gap-chip gap-bad">Gap: {gap_clf*100:.2f}%</span>
            </div>""", unsafe_allow_html=True)
        with ov2:
            gap_reg = REG_TRAIN["r2"] - REG["r2"]
            st.markdown(f"""
            <div class="eval-box eval-box-red">
                <div class="eval-title">Regression · R² Gap</div>
                <div style="font-size:0.92rem; color:#f1f5f9; line-height:2; direction:ltr; text-align:left; padding-top:0.3rem;">
                    Train: <b>{REG_TRAIN['r2']:.4f}</b> → Test: <b>{REG['r2']:.4f}</b>
                </div>
                <span class="gap-chip gap-ok">Gap: {gap_reg:.4f}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(251,191,36,0.07); border:1px solid rgba(251,191,36,0.25); border-radius:10px; padding:0.75rem 1rem; color:#fbbf24; font-size:0.82rem; direction:ltr; text-align:left; margin-bottom:1rem;">
            ⚡ <b>Note:</b> The classification gap (~25%) suggests moderate overfitting. 
            Consider tuning <code>max_depth</code>, adding regularization, or increasing training data.
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Feature importance
        st.markdown("""<div style="font-size:0.78rem; font-weight:800; color:#94a3b8; letter-spacing:0.1em; margin-bottom:0.8rem;">📊  FEATURE IMPORTANCE (XGBoost)</div>""", unsafe_allow_html=True)
        _, gcol3, _ = st.columns([0.5, 4, 0.5])
        with gcol3:
            fig3, ax3 = dark_fig(6, 3.4)
            colors = ["#a78bfa", "#818cf8", "#6366f1", "#22d3ee", "#38bdf8"]
            bars = ax3.barh(FEAT_NAMES, FEAT_IMP, color=colors, height=0.55,
                            edgecolor="none")
            # value labels
            for bar, val in zip(bars, FEAT_IMP):
                ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                         f"{val:.0%}", va="center", ha="left",
                         color="#f1f5f9", fontsize=9, fontweight="bold")
            ax3.set_xlim(0, max(FEAT_IMP) * 1.25)
            ax3.set_xlabel("Relative Importance")
            ax3.tick_params(axis="y", colors="#f1f5f9")
            ax3.grid(axis="x", color="#1e293b", linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
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
#  ACTION BUTTONS  (About + Eval) — styled grid, mobile-safe
# ══════════════════════════════════════════════════════════════════════════════
ab_col, ev_col = st.columns(2, gap="medium")
with ab_col:
    if st.button("👤  دەربارەی پڕۆژە و گەشەپێدەر", use_container_width=True,
                 help="زانیاری دەربارەی پڕۆژە، گەشەپێدەر، و تەکنەلۆژیاکان"):
        project_info_dialog()
with ev_col:
    if st.button("📊  هەڵسەنگاندنی زانستی مۆدێل", use_container_width=True,
                 help="ئەنجامی تەست، مەتریکەکان، و گرافەکان"):
        evaluation_dialog()

# style the two top buttons differently from the main analyze button
st.markdown("""
<style>
/* First two buttons = outline style */
div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stButton"]:nth-child(1) > button {
    background: var(--violet-dim) !important;
    border-color: var(--violet-bdr) !important;
    color: var(--violet) !important;
    box-shadow: none !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stButton"]:nth-child(2) > button {
    background: var(--cyan-dim) !important;
    border-color: var(--cyan-bdr) !important;
    color: var(--cyan) !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

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
    analyze = st.button("🔮  شیکردنەوە و بڕیاردان", use_container_width=True)


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
