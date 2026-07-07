# --- START OF FILE app.py ---

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import json
import base64

# Try to import Plotly, fallback to Matplotlib if not present
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="سیستەمی ژیری نمرەدانی مەترسی | Erbil Warehouse",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
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
#  DYNAMIC Kurdish FONT ENCODING (Base64)
# ══════════════════════════════════════════════════════════════════════════════
def get_font_base64(name):
    try:
        path = os.path.join(os.path.dirname(__file__), name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode('utf-8')
    except Exception:
        pass
    return ""

font57_b64 = get_font_base64("UniQAIDAR_NewsHeadLine 057.ttf")
font58_b64 = get_font_base64("UniQAIDAR_NewsHeadLine 058.ttf")
font59_b64 = get_font_base64("UniQAIDAR_NewsHeadLine 059.ttf")

font_css = ""
if font59_b64:
    font_css += f"""
    @font-face {{
        font-family: 'UniQAIDAR_NewsHeadLine';
        src: url('data:font/truetype;charset=utf-8;base64,{font59_b64}') format('truetype');
        font-weight: 400;
        font-style: normal;
        font-display: swap;
    }}
    """
if font58_b64:
    font_css += f"""
    @font-face {{
        font-family: 'UniQAIDAR_NewsHeadLine';
        src: url('data:font/truetype;charset=utf-8;base64,{font58_b64}') format('truetype');
        font-weight: 600;
        font-style: normal;
        font-display: swap;
    }}
    """
if font57_b64:
    font_css += f"""
    @font-face {{
        font-family: 'UniQAIDAR_NewsHeadLine';
        src: url('data:font/truetype;charset=utf-8;base64,{font57_b64}') format('truetype');
        font-weight: 800;
        font-style: normal;
        font-display: swap;
    }}
    """

# ══════════════════════════════════════════════════════════════════════════════
#  TRANSLATION DICTIONARY
# ══════════════════════════════════════════════════════════════════════════════
translations = {
    "ku": {
        "hero_title": "سیستەمی ژیری <span>نمرەدانی مەترسی</span> و سنووری قەرز",
        "hero_sub": "Erbil Warehouse B2B Credit Limit & Advanced RFM Scoring",
        "hero_pill": "XGBoost ENGINE · REAL-TIME ANALYSIS",
        "btn_about": "دەربارەی پڕۆژە",
        "btn_dataset": "داتاسێت و مۆدێل",
        "btn_eval": "هەڵسەنگاندن",
        "sec_financial": "زانیاری دارایی و مامەڵەکان",
        "label_avg_invoice": "تێکڕای بەهای یەک وەسڵ ($)",
        "helper_avg_invoice": "زۆربەی وەسڵەکانی ئەم دوکانە چەند دۆلارە? (ئاسایی 50 بۆ 500)",
        "label_freq_per_month": "تێکڕای وەسڵەکان لە مانگێکدا (دانە)",
        "helper_freq_per_month": "لە مانگێکدا نزیکەی چەند جار کاڵا دەبات? (ئاسایی 2 بۆ 15)",
        "label_total_volume": "کۆی قەبارەی بازرگانی ($)",
        "helper_total_volume": "کۆی ئەو پارەیەی تا ئێستا کڕینی پێ کردووە چەندە?",
        "label_unpaid_ratio": "ڕێژەی وەسڵە نەدراوەکان لەسەدا (%)",
        "helper_unpaid_ratio": "چەند لەسەدای وەسڵەکانی هێشتا پارەیان نەدراوە? (0 باشترینە)",
        "sec_shop": "زانیاری دوکان و پێشینە",
        "label_shop_age": "تەمەنی دوکان (ساڵانی کارکردن)",
        "helper_shop_age": "چەند ساڵە ئەم دوکانە لە بازاڕدا کار دەکات?",
        "label_days_since_last": "چەند ڕۆژ بەسەر کۆتا مامەڵە تێپەڕیوە",
        "helper_days_since_last": "دوایین جار کەی شتی لە کۆگاکەت کڕیوە? (ژمارەی ڕۆژەکان)",
        "label_debt_ratio": "ڕێژەی قەرز بۆ قەبارەی مامەڵە (%)",
        "helper_debt_ratio": "ڕێژەی قەرزەکانی چەندە بەراورد بە کۆی مامەڵەکانی? (ئاسایی ژێر 30%)",
        "label_late_history": "پێشینەی دواکەوتنی پارەدان (جار)",
        "helper_late_history": "تا ئێستا چەند جار لە کاتی دیاریکراو پارەی نەداوە?",
        "btn_submit": "شیکردنەوە و بڕیاردان",
        "btn_clear": "پاککردنەوەی فۆڕم",
        "results_title": "ئەنجامی شیکاری کۆگا",
        "res_risk_eyebrow": "ئاستی مەترسی دوکان",
        "res_risk_high": "مەترسیدار",
        "res_risk_low": "باوەڕپێکراو",
        "res_limit_eyebrow": "سنووری قەرزی گونجاو (پەسەندکراو)",
        "summary_title": "پوختەی زانیارییەکانی دوکان (Parameters Summary)",
        "sum_shop_age": "تەمەنی دوکان",
        "sum_last_order": "ڕۆژ لە کۆتا مامەڵە",
        "sum_monthly_freq": "تێکڕای وەسڵی مانگانە",
        "sum_avg_invoice": "تێکڕای بەهای وەسڵ",
        "sum_total_volume": "کۆی قەبارەی بازرگانی",
        "sum_unpaid_ratio": "ڕێژەی وەسڵە نەدراوەکان",
        "sum_debt_ratio": "ڕێژەی قەرز بۆ بازرگانی",
        "sum_late_history": "دواکەوتنی پێشینە",
        "unit_years": "ساڵ",
        "unit_days": "ڕۆژ",
        "unit_invoices": "دانە",
        "unit_times": "جار",
        "badge_ml_title": "مۆدێلی فێربوونی ئامێر (Machine Learning) - ئەلگۆریتمی XGBoost",
        "badge_ml_desc": "بڕیاری کۆتایی لە ڕێگەی ئەلگۆریتمی XGBoost دەرکراوە",
        "badge_rule_title": "سیستەمی پاراستنی خێرا (Rule Engine)",
        "badge_security_title": "سیستەمی پاراستنی توند (Security Engine)",
        "toast_success_title": "شیکاری سەرکەوتوو",
        "toast_success_msg": "ئەنجامی پێشبینیکردنی مەترسی و سنووری قەرز بەردەستە.",
        "toast_rule_title": "سیستەمی پاراستنی خێرا چالاک کرا",
        "toast_rule_msg": "کڕیاری تازە / داتای کەم - پێویستی بە چاودێرییە",
        "toast_security_title": "قەرزدان ڕاگیراوە",
        "toast_security_msg": "مەترسی زۆر بەرز - قەرزی کەڵەکەبوو، قەرزدان ڕاگیراوە",
        "toast_clear_title": "پاککردنەوەی فۆڕم",
        "toast_clear_msg": "فۆڕمەکە پاککرایەوە بۆ باری سەرەتایی.",
        
        # Dialog texts
        "about_title": "دەربارەی پڕۆژە و گەشەپێدەر",
        "about_problem_title": "کێشەکە چییە؟ (The Problem)",
        "about_problem_text": "لە کاتی ئێستادا، زۆربەی کۆگاکانی فرۆشتنی کۆ (B2B) لە هەولێر و کوردستان بە گشتی، بڕیاردان لەسەر پێدانی قەرز بە دوکانەکان بە شێوەیەکی <b>هەڕەمەکی و کەسی</b> دەدەن. هیچ پێوەرێکی زانستی نییە بۆ ئەوەی بزانرێت ئایا ئەم دوکانە شایەنی چەند قەرزە، یان ئایا ئەگەری هەیە پارەکە نەگەڕێنێتەوە. ئەمەش دەبێتە هۆی <b>کەڵەکەبوونی قەرز، درەنگ کەوتنی پارەدان، و لەدەستدانی سەرمایەی کۆگاکان</b>.",
        "about_solution_title": "چارەسەرەکە (The Solution)",
        "about_solution_text": "ئەم پڕۆژەیە سیستەمێکی پێشکەوتووی مۆدێلی فێربوونی ئامێرە (Machine Learning) بە بەکارهێنانی ئەلگۆریتمی <b>XGBoost</b> کە پشتبەست بە پێوەرەکانی <b>RFM (Recency, Frequency, Monetary)</b> کاردەکات. سیستەمەکە لە جێگەی مرۆڤ شیکارییەکی خێرا و ورد بۆ مێژووی دوکاندارەکەکە دەکات، و بە شێوەیەکی ئۆتۆماتیکی پێتدەڵێت کە ئایا ئەم دوکانە جێی متمانەیە یان مەترسیدارە، وە بە تەواوی <b>دەستنیشانی دەکات کە تا چەند دۆلار سەلامەتە قەرزی پێ بدرێت</b>.",
        "about_how_title": "چۆن کار دەکات؟ (How it works)",
        "about_how_step1": "<b>١. پشکنینی یاساکانی پاراستنی کۆگا (Rule Engine Check):</b> سیستەمەکە سەرەتا کڕیارە زۆر مەترسیدارەکان (Outliers) یان کڕیارە زۆر تازەکان (Cold Start) فلتەر دەکات بۆ پاراستنی خێرا و پێشوەختەی سەرمایەی کۆگاکە.",
        "about_how_step2": "<b>٢. شیکاری مۆدێلی فێربوونی ئامێر (XGBoost Analysis):</b> ئەگەر کڕیارەکە هیچ سەرپێچی یان بارودۆخێکی تایبەتی نەبوو، مۆدێلی فێربوونی ئامێر (XGBoost) شیکارییەکی ئێجگار ورد بۆ مێژوو و شێوازی مامەڵەکانی دەکات.",
        "about_how_step3": "<b>٣. بڕیاردانی ژیرانە (Smart Decision):</b> بە یەکگرتنی یاساکان و مۆدێلی فێربوونی ئامێر (Machine Learning), نمرەی مەترسیی کڕیارەکە و سنووری قەرزی پێشکەشکراو بە دۆلار دیاری دەکرێت.",
        "about_dev_title": "زانیاری گەشەپێدەر",
        "about_dev_name": "ئومێد جەمال نوری",
        "about_dev_dept": "ئەندازیاری کارەبا - قۆناغی سێیەم",
        "about_dev_year": "ساڵی خوێندن: ٢٠٢٥ - ٢٠٢٦",
        "about_tech_title": "تەکنەلۆژیاکانی بەکارهاتوو",
        "about_tech_intro": "ئەم پڕۆژەیە بەم تەکنەلۆژیایانە دروستکراوە:",
        
        "dataset_title": "زانیاری داتاسێت و مۆدێلەکان",
        "dataset_main_title": "داتاسێت (Dataset)",
        "dataset_desc": "ئەم داتاسێتە سەرەتا لە <b>1200 تۆماری داتا (تۆمار/ڕیز)</b> پێکدێت کە تایبەتە بە بازرگانی (B2B) کۆگاکانی هەولێر. لەبەر ئەوەی زۆربەی کڕیارەکان لە یەک جۆر بوون، بۆ دروستکردنی مۆدێلێکی بێلایەن، تەکنیکی <b>SMOTE</b> بەکارهێنراوە بۆ هاوسەنگکردنەوەی داتاکان، کە بەهۆیەوە قەبارەی داتاسێتەکە بەرزبووەتەوە بۆ <b>زیاتر لە 1500 تۆماری داتا (تۆمار/ڕیز)</b> بە یەکسانی تەواو لە نێوان هەردوو جۆری مەترسیدار و باوەڕپێکراودا.",
        "dataset_features_title": "داتاسێتەکە پێکهاتووە لە <b>8 فیچەر (تایبەتمەندی)</b> کە بەپێی پێوەرەکانی <b>Advanced RFM</b> بەم شێوەیە پۆلێن کراون:",
        "dataset_r_title": "1. نوێکاری مامەڵە (Recency - R):",
        "dataset_r_desc": "Days_Since_Last_Order: ماوەی کۆتا کڕین بە ڕۆژ.",
        "dataset_f_title": "2. بەردەوامی کڕین (Frequency - F):",
        "dataset_f_desc": "Order_Freq_Per_Month: تێکڕای ژمارەی وەسڵەکان لە مانگێکدا.",
        "dataset_m_title": "3. قەبارەی دارایی (Monetary - M):",
        "dataset_m_desc_1": "Average_Invoice_Value: تێکڕای بەهای وەسڵ.",
        "dataset_m_desc_2": "Total_Trade_Volume: کۆی قەبارەی بازرگانی.",
        "dataset_profile_title": "4. پێوەرەکانی مەترسی و متمانە (Risk & Profile):",
        "dataset_profile_desc_1": "Unpaid_Invoice_Ratio: ڕێژەی وەسڵە نەدراوەکان.",
        "dataset_profile_desc_2": "Debt_To_Volume_Ratio: ڕێژەی قەرز بۆ بازرگانی.",
        "dataset_profile_desc_3": "Late_Payment_History: دواکەوتنی پێشینە.",
        "dataset_profile_desc_4": "Shop_Age_Years: تەمەنی دوکان.",
        "dataset_models_title": "مۆدێلەکانی ڕاهێنان (XGBoost)",
        "dataset_models_desc": "لەم پڕۆژەیەدا سوود لە مۆدێلی فێربوونی ئامێری پێشکەوتوو (Machine Learning) بە ئەلگۆریتمی <b>(XGBoost)</b> وەرگیراوە بۆ دروستکردنی دوو مۆدێلی جیاواز کە بە تەکنیکی <b>RandomizedSearchCV</b> باشترین پارامێتەرەکانیان بۆ دۆزراوەتەوە:",
        "dataset_model_clf": "<b>مۆدێلی پۆلێنکردن (Classification):</b> بۆ جیاکردنەوەی کڕیارەکان بۆ دوک بارودۆخ (High Risk و Low Risk).",
        "dataset_model_reg": "<b>مۆدێلی پێشبینیکردن (Regression):</b> بۆ پێشبینیکردنی بڕی قەرزی گونجاو (Credit Limit) بە دۆلار بۆ هەر کڕیارێک بە پشتبەستن بە مەترسییەکەی.",
        "dataset_chart1_title": "شیکاری دابەشبوونی داتا بەهۆی هاوسەنگکردن (SMOTE Distribution Chart)",
        "dataset_chart2_title": "تێکڕای پێوەرە گرنگەکانی ڕێژەی مەترسی (Key Predictors Averages)",

        "eval_main_title": "هەڵسەنگاندن و ئاستی مۆدێلەکان",
        "eval_intro_title": "تێگەیشتن لە پێوەرەکانی هەڵسەنگاندن",
        "eval_intro_desc": "<b>Accuracy:</b> ڕێژەی سەرکەوتنی مۆدێلەکە لە دیاریکردنی مەترسی (بەرز یان نزم).<br><b>ROC-AUC:</b> توانای مۆدێلەکە بۆ جیاکردنەوەی دوو جۆرە کڕیارەکە بە دروستی.<br><b>F1-Score:</b> هاوسەنگی نێوان وردبینی و دۆزینەوەی دروست.<br><b>R² Score:</b> ڕێژەی دروستی پێشبینیکردنی بڕی قەرزەکە (چەند نزیکە لە ڕاستی).<br><b>RMSE & MAE:</b> تێکڕای هەڵەی پێشبینی قەرز بە دۆلار (هەرچەند کەمتر بێت باشترە).",
        "eval_tab_test": "🧪 ئەنجامی تاقیکردنەوە (Test Data)",
        "eval_tab_train": "📚 ئەنجامی ڕاهێنان (Train Data)",
        "eval_clf_title": "🎯 مۆدێلی پۆلێنکردنی مەترسی (XGBoost Classifier)",
        "eval_clf_title_train": "🎯 مۆدێلی پۆلێنکردنی مەترسی (Train Data)",
        "eval_reg_title": "💰 مۆدێلی پێشبینیکردنی قەرز (XGBoost Regressor)",
        "eval_reg_title_train": "💰 مۆدێلی پێشبینیکردنی قەرز (Train Data)",
        "eval_metric_accuracy": "Accuracy (ڕێژەی ڕاستی)",
        "eval_metric_roc_auc": "ROC-AUC Score",
        "eval_metric_f1": "F1-Score",
        "eval_metric_r2": "R² Score (ڕێژەی سەرکەوتن)",
        "eval_metric_rmse": "RMSE (تێکڕای هەڵە)",
        "eval_metric_mae": "MAE (هەڵەی ڕەها)",
        "eval_chart_metrics": "بەراوردکاری پێوەرەکان (Test vs Train)",
        "eval_chart_importance": "گرنگی فیچەرەکان (Feature Importance)",
        "eval_chart_confusion": "ماتریکسی سەرلێشێوان (Confusion Matrix)",
        "eval_chart_radar": "هەڵسەنگاندنی گشتی (Radar Evaluation)",
        "eval_conclusion": "<b>💡 ئەنجامی کۆتایی:</b> مۆدێلەکەمان بە سەرکەوتوویی توانای پێشبینیکردنی هەیە بە ڕێژەی زیاتر لە %90 بۆ مەترسی و %94 بۆ بڕی قەرز لەسەر داتای نەبینراو (Test). ئەنجامەکانی Test و Train زۆر نزیکن لە یەکەوە، ئەمەش دەریدەخات کە مۆدێلەکە زۆر جێگیرە و کێشەی (Overfitting)ی نییە."
    },
    "en": {
        "hero_title": "B2B Credit Risk <span>Scoring & Limit</span> System",
        "hero_sub": "Erbil Warehouse B2B Credit Limit & Advanced RFM Scoring",
        "hero_pill": "XGBoost ENGINE · REAL-TIME ANALYSIS",
        "btn_about": "About Project",
        "btn_dataset": "Dataset & Model",
        "btn_eval": "Evaluation",
        "sec_financial": "Financial & Transaction Profile",
        "label_avg_invoice": "Average Invoice Value ($)",
        "helper_avg_invoice": "What is the average value of each invoice? (Normally $50 to $500)",
        "label_freq_per_month": "Average Invoices Per Month (Units)",
        "helper_freq_per_month": "How many times does the store purchase goods monthly? (Normally 2 to 15)",
        "label_total_volume": "Total Trade Volume ($)",
        "helper_total_volume": "What is the cumulative purchase amount to date?",
        "label_unpaid_ratio": "Unpaid Invoice Ratio (%)",
        "helper_unpaid_ratio": "What percentage of invoices remain unpaid? (0% is optimal)",
        "sec_shop": "Store Profile & History",
        "label_shop_age": "Shop Age (Years of Operation)",
        "helper_shop_age": "How many years has this shop been operating in the market?",
        "label_days_since_last": "Days Since Last Transaction",
        "helper_days_since_last": "How many days have passed since their last warehouse order?",
        "label_debt_ratio": "Debt-to-Volume Ratio (%)",
        "helper_debt_ratio": "What is the ratio of unpaid debt to total trade volume? (Normally under 30%)",
        "label_late_history": "Late Payment History (Times)",
        "helper_late_history": "How many times has the merchant failed to pay on time?",
        "btn_submit": "Analyze & Predict",
        "btn_clear": "Clear Form",
        "results_title": "Warehouse Analysis Results",
        "res_risk_eyebrow": "Merchant Risk Level",
        "res_risk_high": "High Risk",
        "res_risk_low": "Low Risk",
        "res_limit_eyebrow": "Approved Credit Limit",
        "summary_title": "Merchant Parameters Summary",
        "sum_shop_age": "Shop Age",
        "sum_last_order": "Days Since Last Order",
        "sum_monthly_freq": "Monthly Order Frequency",
        "sum_avg_invoice": "Avg Invoice Value",
        "sum_total_volume": "Total Trade Volume",
        "sum_unpaid_ratio": "Unpaid Invoice Ratio",
        "sum_debt_ratio": "Debt-to-Volume Ratio",
        "sum_late_history": "Late Payments",
        "unit_years": "Years",
        "unit_days": "Days",
        "unit_invoices": "Invoices",
        "unit_times": "Times",
        "badge_ml_title": "Machine Learning Model - XGBoost Algorithm",
        "badge_ml_desc": "Final decision rendered via XGBoost algorithm.",
        "badge_rule_title": "Fast Protection System (Rule Engine)",
        "badge_security_title": "Security Engine Protection",
        "opt_late_0": "Never",
        "opt_late_1": "Delayed 1 time",
        "opt_late_2": "Delayed 2 times",
        "opt_late_3": "Delayed 3 times",
        "opt_late_4": "Delayed 4 times",
        "opt_late_5": "Delayed 5+ times",
        "toast_success_title": "Analysis Successful",
        "toast_success_msg": "Merchant risk levels and credit recommendations are ready.",
        "toast_rule_title": "Fast Protection Activated",
        "toast_rule_msg": "New Customer / Insufficient Data - Needs Monitoring",
        "toast_security_title": "Lending Suspended",
        "toast_security_msg": "Extreme Risk - High Accumulated Debt, Lending Suspended",
        "toast_clear_title": "Form Reset",
        "toast_clear_msg": "The input form has been reset to default values.",
        
        # Dialog texts
        "about_title": "About Project & Developer",
        "about_problem_title": "The Problem",
        "about_problem_text": "Currently, most wholesale warehouses (B2B) in Erbil and the Kurdistan region make credit decisions based on intuition or personal relationships. There is no mathematical or data-driven model to assess a merchant's creditworthiness or estimate their default probability, leading to bad debt, late payments, and cash flow constraints.",
        "about_solution_title": "The Solution",
        "about_solution_text": "This project introduces an advanced Machine Learning credit risk engine utilizing the XGBoost algorithm, powered by RFM (Recency, Frequency, Monetary) metrics. It automates the evaluation of merchant trade history to determine if they are high or low risk, and calculates the exact credit ceiling to allocate safely.",
        "about_how_title": "How It Works",
        "about_how_step1": "<b>1. Rule Engine Pre-Checks:</b> The system filters out extreme outliers and newly onboarded merchants (Cold Start) to enforce immediate capital protection rules.",
        "about_how_step2": "<b>2. Machine Learning (XGBoost):</b> If a customer passes the initial checks, the XGBoost engine runs deep inference on historical features to predict default probability.",
        "about_how_step3": "<b>3. Smart Decision:</b> The rule and ML engines combine to output a final risk rating and an approved credit limit in USD.",
        "about_dev_title": "Developer Profile",
        "about_dev_name": "Umed Jamal Nouri",
        "about_dev_dept": "Electrical Engineering - 3rd Year",
        "about_dev_year": "Academic Year: 2025 - 2026",
        "about_tech_title": "Technologies Used",
        "about_tech_intro": "This project is built using the following stack:",
        
        "dataset_title": "Dataset & Model Specifications",
        "dataset_main_title": "Dataset Details",
        "dataset_desc": "The initial dataset contains 1,200 transaction records from B2B wholesale warehouses in Erbil. Since the dataset was highly imbalanced (fewer high-risk merchants), the SMOTE (Synthetic Minority Over-sampling Technique) was applied. This expanded the dataset to over 1,500 balanced records, ensuring the model generalizes well for both risk classes.",
        "dataset_features_title": "The dataset comprises 8 features, structured under Advanced RFM metrics:",
        "dataset_r_title": "1. Recency (R):",
        "dataset_r_desc": "Days elapsed since the merchant's last purchase.",
        "dataset_f_title": "2. Frequency (F):",
        "dataset_f_desc": "Average number of transactions per month.",
        "dataset_m_title": "3. Monetary (M):",
        "dataset_m_desc_1": "Average invoice value per purchase.",
        "dataset_m_desc_2": "Cumulative trade volume between the merchant and the warehouse.",
        "dataset_profile_title": "4. Risk & Profile Metrics:",
        "dataset_profile_desc_1": "The ratio of invoices that remain unpaid.",
        "dataset_profile_desc_2": "Total unpaid debt relative to cumulative trade volume.",
        "dataset_profile_desc_3": "Count of historical payment delays.",
        "dataset_profile_desc_4": "Number of years the shop has been active.",
        "dataset_models_title": "Training Models (XGBoost)",
        "dataset_models_desc": "This project deploys two machine learning models optimized using RandomizedSearchCV: a Classifier to categorize merchants into Risk Levels (High vs Low), and a Regressor to predict the exact credit limit based on their trade and risk history.",
        "dataset_model_clf": "<b>Classification Model:</b> Categorizes merchants into two states (High Risk and Low Risk).",
        "dataset_model_reg": "<b>Regression Model:</b> Predicts the approved credit limit ($) based on the merchant's profiles and calculated risk.",
        "dataset_chart1_title": "Data Distribution Before & After SMOTE Balancing",
        "dataset_chart2_title": "Key Risk Indicators and Predictor Averages",

        "eval_main_title": "Model Evaluation & Metrics",
        "eval_intro_title": "Understanding Evaluation Metrics",
        "eval_intro_desc": "<b>Accuracy:</b> The percentage of correct classifications made by the model.<br><b>ROC-AUC:</b> The model's ability to distinguish between high-risk and low-risk merchants.<br><b>F1-Score:</b> Harmonic mean of precision and recall.<br><b>R² Score:</b> The accuracy of credit limit predictions compared to historical actual values.<br><b>RMSE & MAE:</b> The average and absolute prediction errors in USD (lower is better).",
        "eval_tab_test": "🧪 Test Data Results",
        "eval_tab_train": "📚 Train Data Results",
        "eval_clf_title": "🎯 Risk Classification Model (XGBoost Classifier)",
        "eval_clf_title_train": "🎯 Risk Classification Model (Train Data)",
        "eval_reg_title": "💰 Credit Prediction Model (XGBoost Regressor)",
        "eval_reg_title_train": "💰 Credit Prediction Model (Train Data)",
        "eval_metric_accuracy": "Accuracy",
        "eval_metric_roc_auc": "ROC-AUC Score",
        "eval_metric_f1": "F1-Score",
        "eval_metric_r2": "R² Score (Model Fit)",
        "eval_metric_rmse": "RMSE (Avg Error)",
        "eval_metric_mae": "MAE (Absolute Error)",
        "eval_chart_metrics": "Metrics Comparison (Test vs Train)",
        "eval_chart_importance": "Feature Importance",
        "eval_chart_confusion": "Confusion Matrix",
        "eval_chart_radar": "General Evaluation (Radar)",
        "eval_conclusion": "<b>💡 Conclusion:</b> The models achieve over 90% accuracy for risk classification and 94% R² score for credit limit prediction on unseen test data. The close proximity of training and testing metrics confirms model stability and negates overfitting."
    }
}

chart_translations = {
    "ku": {
        "feature_importance_title": 'گرنگی فیچەر (%)',
        "feature_importance_categories": ['ڕێژەی وەسڵە نەدراوەکان', 'ڕێژەی قەرز بە مامەڵە', 'مەترسی پێشبینیکراو', 'کۆی قەبارەی بازرگانی', 'تێکڕای وەسڵ لە مانگێکدا', 'تێکڕای بەهای وەسڵ', 'تەمەنی دوکان', 'دواکەوتنی پێشینە', 'ڕۆژانی کۆتا وەسڵ'],
        "confusion_matrix_actual_high": 'کردەیی: مەترسی بەرز',
        "confusion_matrix_actual_low": 'کردەیی: مەترسی نزم',
        "confusion_matrix_pred_high": 'پێشبینیکراو: مەترسی بەرز',
        "confusion_matrix_pred_low": 'پێشبینیکراو: مەترسی نزم',
        "confusion_matrix_errors": 'هەڵەکان',
        "confusion_matrix_correct": 'ڕاستەکان',
        "dataset_before_smote": 'پێش SMOTE',
        "dataset_after_smote": 'دوای SMOTE',
        "dataset_low_risk": 'مەترسی نزم',
        "dataset_high_risk": 'مەترسی بەرز',
        "dataset_features_categories": ['قەرز بۆ مامەڵە', 'وەسڵە نەدراوەکان', 'دواکەوتن x10']
    },
    "en": {
        "feature_importance_title": 'Feature Importance (%)',
        "feature_importance_categories": ['Unpaid Invoice Ratio', 'Debt-to-Volume Ratio', 'Predicted Risk Score', 'Total Trade Volume', 'Monthly Freq', 'Avg Invoice Value', 'Shop Age', 'Late Payment History', 'Days Since Last Order'],
        "confusion_matrix_actual_high": 'Actual High Risk',
        "confusion_matrix_actual_low": 'Actual Low Risk',
        "confusion_matrix_pred_high": 'Predicted High Risk',
        "confusion_matrix_pred_low": 'Predicted Low Risk',
        "confusion_matrix_errors": 'Errors',
        "confusion_matrix_correct": 'Correct',
        "dataset_before_smote": 'Before SMOTE',
        "dataset_after_smote": 'After SMOTE',
        "dataset_low_risk": 'Low Risk',
        "dataset_high_risk": 'High Risk',
        "dataset_features_categories": ['Debt-to-Volume', 'Unpaid Invoice', 'Late Payments x10']
    }
}

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR LANGUAGE SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<h2 style="font-size:1.2rem; font-weight:800; border-bottom:1px solid var(--glass-border); padding-bottom:10px; margin-bottom:15px;">⚙️ Configuration / زمان</h2>', unsafe_allow_html=True)
    lang_selection = st.radio("Select Language / زمان دیاری بکە", ["کوردی سۆرانی", "English"], index=0)
    current_lang = "ku" if lang_selection == "کوردی سۆرانی" else "en"
    direction = "rtl" if current_lang == "ku" else "ltr"
    align = "right" if current_lang == "ku" else "left"
    font_family = "'UniQAIDAR_NewsHeadLine', 'Noto Sans Arabic', sans-serif" if current_lang == "ku" else "'Inter', sans-serif"

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL GLASSMORPHISM CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
{font_css}

@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;600;700;800;900&family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {{
    --bg-base: #020809; 
    --glass-bg: rgba(8, 22, 25, 0.88);
    --glass-border: rgba(167, 203, 142, 0.12);
    --text-main: #ffffff;
    --text-muted: #c5d4d6;
    --input-bg: rgba(5, 14, 16, 0.75);
    --modal-bg: rgba(8, 22, 25, 0.98);
    --orb-1: rgba(9, 35, 39, 0.65);
    --orb-2: rgba(167, 203, 142, 0.22);
    --orb-3: rgba(243, 163, 50, 0.18);
    --primary: #a7cb8e;
    --primary-glow: rgba(167, 203, 142, 0.35);
    --secondary: #a7cb8e;
    --accent: #f3a332;
    --green: #10b981;
    --red: #ef4444;
    --r: 20px;
}}

/* Global resets and typography */
html, body, [class*="css"], .stApp, .block-container {{
    font-family: {font_family} !important;
    direction: {direction} !important;
    text-align: {align} !important;
}}

p, h1, h2, h3, h4, h5, h6, span, li,
div[data-testid="stMarkdownContainer"],
div[data-testid="stMarkdownContainer"] p {{
    color: var(--text-main) !important;
}}

.stApp {{
    background-color: var(--bg-base);
    background-image: 
        radial-gradient(circle at 10% 20%, var(--orb-1), transparent 40%),
        radial-gradient(circle at 90% 80%, var(--orb-2), transparent 45%),
        radial-gradient(circle at 50% 50%, var(--orb-3), transparent 60%);
    background-attachment: fixed;
    min-height: 100vh;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.5rem 2rem 4rem !important; max-width: 1100px !important; }}

.glass-panel {{
    background: var(--glass-bg);
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
    border: 1px solid var(--glass-border);
    border-top: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.45);
    border-radius: var(--r);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}}

.hero {{
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
}}
.hero-title {{
    font-size: 2.2rem;
    font-weight: 900;
    line-height: 1.45;
    margin-bottom: 0.5rem;
}}
.hero-title span {{
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.hero-sub {{
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 500;
}}
.hero-pill {{
    display: inline-block;
    margin-top: 1.1rem;
    background: rgba(167, 203, 142, 0.1);
    border: 1px solid rgba(167, 203, 142, 0.3);
    border-radius: 50px;
    padding: 0.35rem 1.2rem;
    font-size: 0.8rem;
    font-weight: 800;
    color: var(--primary);
}}

/* Form inputs adjustments */
div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stNumberInput"] div[data-baseweb="base-input"],
div[data-testid="stNumberInput"] div[data-baseweb="input"] > div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
    background-color: transparent !important;
    border: none !important;
}}

.stNumberInput input, .stSelectbox > div > div, .stSelectbox > div > div > div {{
    background-color: var(--input-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    padding: 0.6rem 1rem !important;
    box-shadow: inset 0 2px 5px rgba(0,0,0,0.6) !important;
}}
.stNumberInput input {{
    direction: ltr !important;
    text-align: center !important;
}}
div[data-testid="stNumberInputStepUp"],
div[data-testid="stNumberInputStepDown"] {{
    background-color: rgba(255, 255, 255, 0.05) !important;
    color: #fff !important;
    border-radius: 8px !important;
}}

div[data-baseweb="popover"] > div, div[role="listbox"], ul[role="listbox"] {{
    background-color: #0c0d11 !important;
    border: 1px solid rgba(167, 203, 142, 0.3) !important;
    border-radius: 12px !important;
}}
div[role="listbox"] li, ul[role="listbox"] li {{
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    padding: 0.8rem !important;
}}
div[role="listbox"] li:hover, div[role="listbox"] li[aria-selected="true"] {{
    background-color: rgba(167, 203, 142, 0.2) !important;
}}

div[data-testid="stSlider"] {{
    direction: ltr !important;
}}
div[data-testid="stSlider"] .rc-slider-rail, .stSlider .rc-slider-rail {{
    background: rgba(0,0,0,0.6) !important;
    border-radius: 6px !important;
    height: 8px !important;
}}
div[data-testid="stSlider"] .rc-slider-track, .stSlider .rc-slider-track {{
    background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
    height: 8px !important;
}}
div[data-testid="stSlider"] .rc-slider-handle, .stSlider .rc-slider-handle {{
    width: 22px !important;
    height: 22px !important;
    background: #fff !important;
    border: 4px solid var(--primary) !important;
    box-shadow: 0 0 15px var(--primary-glow) !important;
}}

div[data-testid="stBaseButton-primary"] button, button[kind="primary"] {{
    width: 100% !important;
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 800 !important;
    border-radius: 14px !important;
    padding: 0.85rem 1.2rem !important;
    box-shadow: 0 4px 20px var(--primary-glow) !important;
    transition: all 0.3s ease !important;
}}
div[data-testid="stBaseButton-primary"] button:hover, button[kind="primary"]:hover {{
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px var(--primary-glow) !important;
    filter: brightness(1.1);
}}

div[data-testid="stBaseButton-secondary"] button, button[kind="secondary"] {{
    width: 100% !important;
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text-muted) !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    padding: 0.85rem 1.2rem !important;
    transition: all 0.3s ease !important;
}}
div[data-testid="stBaseButton-secondary"] button:hover, button[kind="secondary"]:hover {{
    background: rgba(167, 203, 142, 0.1) !important;
    color: var(--primary) !important;
    border-color: rgba(167, 203, 142, 0.3) !important;
    transform: translateY(-2px) !important;
}}

.sec-head {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
    margin-top: 1.5rem;
}}
.sec-head-line {{
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.3) 0%, transparent 100%);
}}
.sec-head-text {{
    color: #fff;
    font-size: 1rem;
    font-weight: 800;
    white-space: nowrap;
}}

.result-card {{
    padding: 2.5rem 2rem;
    text-align: center;
    border-radius: var(--r);
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
}}
.rc-eyebrow {{
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-muted);
    margin-bottom: 1rem;
}}
.rc-value {{
    font-size: 3rem;
    font-weight: 900;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}}
.rc-en {{
    font-size: 0.85rem;
    color: var(--text-muted);
    font-family: 'Inter', sans-serif;
    margin-bottom: 1rem;
}}
.low-risk {{
    border-top: 4px solid var(--green);
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.08) 0%, transparent 100%);
}}
.low-risk .rc-value {{ color: var(--green); }}
.high-risk {{
    border-top: 4px solid var(--red);
    background: linear-gradient(180deg, rgba(239, 68, 68, 0.08) 0%, transparent 100%);
}}
.high-risk .rc-value {{ color: var(--red); }}
.limit-card {{
    border-top: 4px solid var(--primary);
    background: linear-gradient(180deg, rgba(167, 203, 142, 0.08) 0%, transparent 100%);
}}
.limit-card .rc-value {{
    background: linear-gradient(to right, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.helper-text {{
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 4px;
    margin-bottom: 12px;
}}

.info-box {{
    background: rgba(167, 203, 142, 0.08);
    border-left: 4px solid var(--primary);
    padding: 1.2rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
}}
.info-title {{
    font-weight: 800;
    color: var(--primary);
    margin-bottom: 0.6rem;
    font-size: 1rem;
}}
.info-text {{
    font-size: 0.95rem;
    color: var(--text-muted);
    line-height: 1.8;
}}
.info-text b {{
    color: #fff;
}}

.tech-tag {{
    display: inline-block;
    background: rgba(255,255,255,0.05);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.25rem 0.15rem;
}}

.metric-card {{
    padding: 1.5rem 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    background: rgba(0,0,0,0.4);
}}
.metric-label {{
    color: var(--text-muted);
    font-size: 0.8rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    font-family: 'Inter', sans-serif;
}}
.metric-value {{
    color: #fff;
    font-size: 1.6rem;
    font-weight: 900;
    font-family: 'Inter', sans-serif;
    direction: ltr;
}}

/* Dynamic background animations */
.stars-container {{
    position: fixed;
    inset: 0;
    z-index: -2;
    overflow: hidden;
    pointer-events: none;
}}
.star {{
    position: absolute;
    border-radius: 50%;
    animation: floatParticle var(--d, 15s) linear infinite;
    opacity: var(--op, 0.6);
    will-change: transform;
    backface-visibility: hidden;
}}
.star {{
    background: radial-gradient(circle, rgba(167, 203, 142, 0.75) 0%, rgba(17, 48, 53, 0.4) 60%, transparent 100%);
}}
@keyframes floatParticle {{
    0% {{
        transform: translate3d(0, 105vh, 0) scale(var(--s, 1));
    }}
    50% {{
        transform: translate3d(var(--tx, 40px), 50vh, 0) scale(var(--s, 1));
    }}
    100% {{
        transform: translate3d(0, -5vh, 0) scale(var(--s, 1));
    }}
}}

.bg-grid {{
    position: fixed;
    inset: 0;
    z-index: -2;
    pointer-events: none;
    background-image: linear-gradient(rgba(167, 203, 142, 0.08) 1px, transparent 1px), 
                      linear-gradient(90deg, rgba(167, 203, 142, 0.08) 1px, transparent 1px);
    background-size: 50px 50px;
    mask-image: radial-gradient(ellipse 60% 60% at 50% 50%, black 20%, transparent 100%);
    -webkit-mask-image: radial-gradient(ellipse 60% 60% at 50% 50%, black 20%, transparent 100%);
}}
.bg-orb-1 {{ position: fixed; top: -20%; left: -10%; width: 60vw; height: 60vw; border-radius: 50%; background: radial-gradient(circle, var(--orb-1) 0%, transparent 70%); filter: blur(70px); z-index: -1; animation: float 12s ease-in-out infinite alternate; pointer-events: none; }}
.bg-orb-2 {{ position: fixed; bottom: -20%; right: -10%; width: 60vw; height: 60vw; border-radius: 50%; background: radial-gradient(circle, var(--orb-2) 0%, transparent 70%); filter: blur(70px); z-index: -1; animation: float 15s ease-in-out infinite alternate-reverse; pointer-events: none; }}
@keyframes float {{ 0% {{ transform: translate3d(0, 0, 0) scale(1); }} 100% {{ transform: translate3d(0, 40px, 0) scale(1.1); }} }}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FLOATING PARTICLES BACKGROUND
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="position: fixed; inset: 0; z-index: -2; overflow: hidden; pointer-events: none;">
    <div class="bg-orb-1"></div>
    <div class="bg-orb-2"></div>
    <div class="bg-grid"></div>
    <div class="stars-container">
        <div class="star" style="top:-10%; left:10%; width:8px; height:8px; --d:20s; --op:0.6; --s:1; --tx:30px;"></div>
        <div class="star" style="top:-10%; left:30%; width:12px; height:12px; --d:25s; --op:0.4; --s:1.2; --tx:-40px;"></div>
        <div class="star" style="top:-10%; left:50%; width:6px; height:6px; --d:18s; --op:0.8; --s:0.8; --tx:50px;"></div>
        <div class="star" style="top:-10%; left:70%; width:10px; height:10px; --d:22s; --op:0.5; --s:1.1; --tx:-20px;"></div>
        <div class="star" style="top:-10%; left:90%; width:14px; height:14px; --d:28s; --op:0.3; --s:1.3; --tx:60px;"></div>
        <div class="star" style="top:-10%; left:20%; width:7px; height:7px; --d:24s; --op:0.7; --s:0.9; --tx:-35px;"></div>
        <div class="star" style="top:-10%; left:60%; width:9px; height:9px; --d:21s; --op:0.6; --s:1.0; --tx:25px;"></div>
        <div class="star" style="top:-10%; left:80%; width:11px; height:11px; --d:27s; --op:0.4; --s:1.15; --tx:-45px;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHART BUILDERS (WITH MATPLOTLIB FALLBACKS)
# ══════════════════════════════════════════════════════════════════════════════
def style_plotly_fig(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8ca2a4', family='Inter, sans-serif'),
        xaxis=dict(
            gridcolor='rgba(167, 203, 142, 0.1)',
            zerolinecolor='rgba(167, 203, 142, 0.1)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(167, 203, 142, 0.1)',
            zerolinecolor='rgba(167, 203, 142, 0.1)',
            showgrid=True
        ),
        margin=dict(l=30, r=30, t=30, b=30),
        height=320
    )
    return fig

def setup_mpl_style():
    plt.rcParams['figure.facecolor'] = '#020809'
    plt.rcParams['axes.facecolor'] = '#020809'
    plt.rcParams['text.color'] = '#f8fafc'
    plt.rcParams['axes.labelcolor'] = '#f8fafc'
    plt.rcParams['xtick.color'] = '#94a3b8'
    plt.rcParams['ytick.color'] = '#94a3b8'
    plt.rcParams['grid.color'] = 'rgba(167, 203, 142, 0.15)'

# 1. Grouped Bar Chart (Test vs. Train metrics)
def plot_compare_metrics(lang):
    if HAS_PLOTLY:
        categories = ['Accuracy', 'ROC-AUC', 'F1-Score', 'R² Score']
        test_data = [90.84, 95.92, 91.09, 94.52]
        train_data = [95.16, 99.10, 95.23, 98.66]
        test_label = 'Test Data' if lang == 'en' else 'داتای Test'
        train_label = 'Train Data' if lang == 'en' else 'داتای Train'
        fig = go.Figure()
        fig.add_trace(go.Bar(x=categories, y=test_data, name=test_label, marker_color='#a7cb8e'))
        fig.add_trace(go.Bar(x=categories, y=train_data, name=train_label, marker_color='#f3a332'))
        fig.update_layout(barmode='group', yaxis=dict(range=[0, 100]))
        return style_plotly_fig(fig)
    else:
        setup_mpl_style()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        categories = ['Accuracy', 'ROC-AUC', 'F1-Score', 'R² Score']
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, [90.84, 95.92, 91.09, 94.52], width, label='Test Data', color='#a7cb8e')
        ax.bar(x + width/2, [95.16, 99.10, 95.23, 98.66], width, label='Train Data', color='#f3a332')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        return fig

# 2. Horizontal Bar Chart (Feature Importances)
def plot_feature_importance(lang):
    if HAS_PLOTLY:
        categories = chart_translations[lang]["feature_importance_categories"]
        importances = [39.69, 23.84, 8.12, 7.47, 6.75, 6.55, 5.32, 1.28, 0.98]
        fig = go.Figure(go.Bar(
            x=importances[::-1],
            y=categories[::-1],
            orientation='h',
            marker_color='#a7cb8e',
            text=[f"{v:.2f}%" for v in importances[::-1]],
            textposition='auto'
        ))
        fig.update_layout(xaxis=dict(showgrid=True), yaxis=dict(showgrid=False))
        return style_plotly_fig(fig)
    else:
        setup_mpl_style()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        categories = chart_translations[lang]["feature_importance_categories"]
        importances = [39.69, 23.84, 8.12, 7.47, 6.75, 6.55, 5.32, 1.28, 0.98]
        y_pos = np.arange(len(categories))
        ax.barh(y_pos[::-1], importances, color='#a7cb8e')
        ax.set_yticks(y_pos[::-1])
        ax.set_yticklabels(categories)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        return fig

# 3. Heatmap (Confusion Matrix: 168, 23, 12, 179)
def plot_confusion_matrix(lang):
    if HAS_PLOTLY:
        trans = chart_translations[lang]
        z = [[168, 23], [12, 179]]
        x = [trans["confusion_matrix_pred_low"], trans["confusion_matrix_pred_high"]]
        y = [trans["confusion_matrix_actual_low"], trans["confusion_matrix_actual_high"]]
        fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale=[[0, '#f3a332'], [1, '#113035']], showscale=False))
        for i in range(len(y)):
            for j in range(len(x)):
                fig.add_annotation(x=x[j], y=y[i], text=str(z[i][j]), showarrow=False, font=dict(color='#ffffff', size=16, bold=True))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        return style_plotly_fig(fig)
    else:
        setup_mpl_style()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        trans = chart_translations[lang]
        z = [[168, 23], [12, 179]]
        im = ax.imshow(z, cmap='YlGnBu')
        ax.set_xticks([0, 1])
        ax.set_xticklabels([trans["confusion_matrix_pred_low"], trans["confusion_matrix_pred_high"]])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([trans["confusion_matrix_actual_low"], trans["confusion_matrix_actual_high"]])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(z[i][j]), ha="center", va="center", color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

# 4. Radar Chart (Accuracy, ROC-AUC, F1, R2)
def plot_radar_evaluation(lang):
    if HAS_PLOTLY:
        categories = ['Accuracy', 'ROC-AUC', 'F1-Score', 'R² Score']
        test_data = [90.84, 95.92, 91.09, 94.52]
        train_data = [95.16, 99.10, 95.23, 98.66]
        categories = categories + [categories[0]]
        test_data = test_data + [test_data[0]]
        train_data = train_data + [train_data[0]]
        test_label = 'Test Data' if lang == 'en' else 'داتای Test'
        train_label = 'Train Data' if lang == 'en' else 'داتای Train'
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=test_data, theta=categories, fill='toself', name=test_label, line_color='#a7cb8e'))
        fig.add_trace(go.Scatterpolar(r=train_data, theta=categories, fill='toself', name=train_label, line_color='#f3a332'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor='rgba(0,0,0,0)'))
        return style_plotly_fig(fig)
    else:
        setup_mpl_style()
        categories = ['Accuracy', 'ROC-AUC', 'F1-Score', 'R² Score']
        test_data = [90.84, 95.92, 91.09, 94.52]
        train_data = [95.16, 99.10, 95.23, 98.66]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        test_data += test_data[:1]
        train_data += train_data[:1]
        fig, ax = plt.subplots(figsize=(6, 3.5), subplot_kw=dict(projection='polar'))
        ax.fill(angles, test_data, color='#a7cb8e', alpha=0.25, label='Test Data')
        ax.plot(angles, test_data, color='#a7cb8e', linewidth=2)
        ax.fill(angles, train_data, color='#f3a332', alpha=0.25, label='Train Data')
        ax.plot(angles, train_data, color='#f3a332', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        return fig

# 5. Dataset Distribution (Before vs. After SMOTE)
def plot_smote_distribution(lang):
    if HAS_PLOTLY:
        trans = chart_translations[lang]
        classes = [trans["dataset_low_risk"], trans["dataset_high_risk"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=classes, y=[840, 360], name=trans["dataset_before_smote"], marker_color='#f3a332'))
        fig.add_trace(go.Bar(x=classes, y=[840, 840], name=trans["dataset_after_smote"], marker_color='#a7cb8e'))
        fig.update_layout(barmode='group')
        return style_plotly_fig(fig)
    else:
        setup_mpl_style()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        trans = chart_translations[lang]
        classes = [trans["dataset_low_risk"], trans["dataset_high_risk"]]
        x = np.arange(len(classes))
        width = 0.35
        ax.bar(x - width/2, [840, 360], width, label=trans["dataset_before_smote"], color='#f3a332')
        ax.bar(x + width/2, [840, 840], width, label=trans["dataset_after_smote"], color='#a7cb8e')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        return fig

# 6. Key Predictors Averages
def plot_key_predictors(lang):
    if HAS_PLOTLY:
        trans = chart_translations[lang]
        categories = trans["dataset_features_categories"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=categories, y=[12, 5, 8], name=trans["dataset_low_risk"], marker_color='#a7cb8e'))
        fig.add_trace(go.Bar(x=categories, y=[48, 35, 42], name=trans["dataset_high_risk"], marker_color='#f3a332'))
        fig.update_layout(barmode='group', yaxis=dict(ticksuffix="%"))
        return style_plotly_fig(fig)
    else:
        setup_mpl_style()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        trans = chart_translations[lang]
        categories = trans["dataset_features_categories"]
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, [12, 5, 8], width, label=trans["dataset_low_risk"], color='#a7cb8e')
        ax.bar(x + width/2, [48, 35, 42], width, label=trans["dataset_high_risk"], color='#f3a332')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        return fig

# ══════════════════════════════════════════════════════════════════════════════
#  MODAL DIALOGS
# ══════════════════════════════════════════════════════════════════════════════
@st.dialog("📊 Dataset Specs / زانیاری داتاسێت", width="large")
def dataset_model_info_dialog(lang):
    t = translations[lang]
    st.markdown(f"""
    <div class="info-box">
        <div class="info-title">📁 {t["dataset_main_title"]}</div>
        <div class="info-text">{t["dataset_desc"]}</div>
    </div>
    
    <div class="info-box" style="border-left-color: var(--accent); background: rgba(243, 163, 50, 0.05);">
        <div class="info-title" style="color: var(--accent);">⚙️ {t["dataset_models_title"]}</div>
        <div class="info-text">
            {t["dataset_models_desc"]}<br><br>
            • {t["dataset_model_clf"]}<br>
            • {t["dataset_model_reg"]}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<h3 style="font-size: 1.1rem; font-weight: 800; margin-top: 1.5rem; margin-bottom: 1rem;">{t["dataset_features_title"]}</h3>', unsafe_allow_html=True)
    
    feat_html = f"""
    <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--glass-border); border-radius: 12px; padding: 1rem; margin-bottom: 2rem;">
        <ul style="margin: 0; padding-left: 20px; list-style-type: square; color: var(--text-muted);">
            <li><b>{t["dataset_r_title"]}</b> {t["dataset_r_desc"]}</li>
            <li style="margin-top: 8px;"><b>{t["dataset_f_title"]}</b> {t["dataset_f_desc"]}</li>
            <li style="margin-top: 8px;"><b>{t["dataset_m_title"]}</b>
                <ul style="list-style-type: circle; padding-left: 20px; margin-top: 4px;">
                    <li>{t["dataset_m_desc_1"]}</li>
                    <li>{t["dataset_m_desc_2"]}</li>
                </ul>
            </li>
            <li style="margin-top: 8px;"><b>{t["dataset_profile_title"]}</b>
                <ul style="list-style-type: circle; padding-left: 20px; margin-top: 4px;">
                    <li>{t["dataset_profile_desc_1"]}</li>
                    <li>{t["dataset_profile_desc_2"]}</li>
                    <li>{t["dataset_profile_desc_3"]}</li>
                    <li>{t["dataset_profile_desc_4"]}</li>
                </ul>
            </li>
        </ul>
    </div>
    """
    st.markdown(feat_html, unsafe_allow_html=True)
    
    st.markdown(f'<h3 style="font-size: 1.1rem; font-weight: 800; text-align: center;">{t["dataset_chart1_title"]}</h3>', unsafe_allow_html=True)
    if HAS_PLOTLY:
        st.plotly_chart(plot_smote_distribution(lang), use_container_width=True, config={'displayModeBar': False})
    else:
        st.pyplot(plot_smote_distribution(lang))
        
    st.markdown(f'<h3 style="font-size: 1.1rem; font-weight: 800; text-align: center; margin-top: 2rem;">{t["dataset_chart2_title"]}</h3>', unsafe_allow_html=True)
    if HAS_PLOTLY:
        st.plotly_chart(plot_key_predictors(lang), use_container_width=True, config={'displayModeBar': False})
    else:
        st.pyplot(plot_key_predictors(lang))

@st.dialog("📈 Model Evaluation / هەڵسەنگاندن", width="large")
def model_evaluation_dialog(lang):
    t = translations[lang]
    st.markdown(f"""
    <div class="info-box">
        <div class="info-title">📊 {t["eval_intro_title"]}</div>
        <div class="info-text">{t["eval_intro_desc"]}</div>
    </div>
    """, unsafe_allow_html=True)
    
    tab_test, tab_train = st.tabs([t["eval_tab_test"], t["eval_tab_train"]])
    
    with tab_test:
        st.markdown(f"""
        <div class="sec-head">
            <span class="sec-head-text">🎯 {t["eval_clf_title"]}</span>
            <span class="sec-head-line"></span>
        </div>""", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["eval_metric_accuracy"]}</div><div class="metric-value">90.84%</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["eval_metric_roc_auc"]}</div><div class="metric-value">95.92%</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["eval_metric_f1"]}</div><div class="metric-value">91.09%</div></div>', unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class="sec-head" style="margin-top: 2rem;">
            <span class="sec-head-text">💰 {t["eval_reg_title"]}</span>
            <span class="sec-head-line"></span>
        </div>""", unsafe_allow_html=True)
        
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["eval_metric_r2"]}</div><div class="metric-value">94.52%</div></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["eval_metric_rmse"]}</div><div class="metric-value">$96.21</div></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["eval_metric_mae"]}</div><div class="metric-value">$41.33</div></div>', unsafe_allow_html=True)

    with tab_train:
        st.markdown(f"""
        <div class="sec-head">
            <span class="sec-head-text">🎯 {t["eval_clf_title_train"]}</span>
            <span class="sec-head-line"></span>
        </div>""", unsafe_allow_html=True)
        
        c1_t, c2_t, c3_t = st.columns(3)
        with c1_t:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["eval_metric_accuracy"]}</div><div class="metric-value">95.16%</div></div>', unsafe_allow_html=True)
        with c2_t:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["eval_metric_roc_auc"]}</div><div class="metric-value">99.10%</div></div>', unsafe_allow_html=True)
        with c3_t:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["eval_metric_f1"]}</div><div class="metric-value">95.23%</div></div>', unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class="sec-head" style="margin-top: 2rem;">
            <span class="sec-head-text">💰 {t["eval_reg_title_train"]}</span>
            <span class="sec-head-line"></span>
        </div>""", unsafe_allow_html=True)
        
        r1_t, r2_t, r3_t = st.columns(3)
        with r1_t:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["eval_metric_r2"]}</div><div class="metric-value">98.66%</div></div>', unsafe_allow_html=True)
        with r2_t:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["eval_metric_rmse"]}</div><div class="metric-value">$46.32</div></div>', unsafe_allow_html=True)
        with r3_t:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t["eval_metric_mae"]}</div><div class="metric-value">$15.29</div></div>', unsafe_allow_html=True)

    st.markdown(f'<h3 style="font-size: 1.1rem; font-weight: 800; text-align: center; margin-top: 2rem;">{t["eval_chart_metrics"]}</h3>', unsafe_allow_html=True)
    if HAS_PLOTLY:
        st.plotly_chart(plot_compare_metrics(lang), use_container_width=True, config={'displayModeBar': False})
    else:
        st.pyplot(plot_compare_metrics(lang))
        
    st.markdown(f'<h3 style="font-size: 1.1rem; font-weight: 800; text-align: center; margin-top: 2rem;">{t["eval_chart_importance"]}</h3>', unsafe_allow_html=True)
    if HAS_PLOTLY:
        st.plotly_chart(plot_feature_importance(lang), use_container_width=True, config={'displayModeBar': False})
    else:
        st.pyplot(plot_feature_importance(lang))
        
    st.markdown(f'<h3 style="font-size: 1.1rem; font-weight: 800; text-align: center; margin-top: 2rem;">{t["eval_chart_confusion"]}</h3>', unsafe_allow_html=True)
    if HAS_PLOTLY:
        st.plotly_chart(plot_confusion_matrix(lang), use_container_width=True, config={'displayModeBar': False})
    else:
        st.pyplot(plot_confusion_matrix(lang))
        
    st.markdown(f'<h3 style="font-size: 1.1rem; font-weight: 800; text-align: center; margin-top: 2rem;">{t["eval_chart_radar"]}</h3>', unsafe_allow_html=True)
    if HAS_PLOTLY:
        st.plotly_chart(plot_radar_evaluation(lang), use_container_width=True, config={'displayModeBar': False})
    else:
        st.pyplot(plot_radar_evaluation(lang))
        
    st.markdown(f"""
    <div style="background: rgba(167, 203, 142, 0.08); border-left: 4px solid var(--primary); padding: 1rem; border-radius: 8px; margin-top: 1.5rem; font-size: 0.95rem;">
        {t["eval_conclusion"]}
    </div>
    """, unsafe_allow_html=True)

@st.dialog("ℹ️ About / دەربارەی پڕۆژە", width="large")
def project_info_dialog(lang):
    t = translations[lang]
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem; filter: drop-shadow(0 0 15px var(--primary-glow));">📦</div>
        <div style="color: var(--primary); font-size: 1.2rem; font-weight: 900; letter-spacing: 0.08em; font-family: 'Inter', sans-serif;">ERBIL WAREHOUSE RISK SYSTEM</div>
        <div style="color: var(--text-muted); font-size: 0.8rem; margin-top: 0.3rem;">v 3.0 · Advanced RFM XGBoost · 2025–2026</div>
    </div>
    
    <div class="info-box">
        <div class="info-title">🤔 {t["about_problem_title"]}</div>
        <div class="info-text">{t["about_problem_text"]}</div>
    </div>
    
    <div class="info-box" style="border-left-color: var(--accent); background: rgba(243, 163, 50, 0.05);">
        <div class="info-title" style="color: var(--accent);">💡 {t["about_solution_title"]}</div>
        <div class="info-text">{t["about_solution_text"]}</div>
    </div>
    
    <div class="info-box" style="border-left-color: var(--primary); background: rgba(167, 203, 142, 0.05);">
        <div class="info-title" style="color: var(--primary);">⚙️ {t["about_how_title"]}</div>
        <div class="info-text">
            • {t["about_how_step1"]}<br>
            • {t["about_how_step2"]}<br>
            • {t["about_how_step3"]}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(f"""
        <div class="glass-panel" style="margin-bottom:0; height: 100%;">
            <div style="color: var(--primary); font-weight: 800; margin-bottom: 0.8rem; border-bottom: 1px solid var(--glass-border); padding-bottom: 5px;">👨‍💻 {t["about_dev_title"]}</div>
            <div style="font-size: 0.9rem; color: var(--text-muted); line-height: 1.8;">
                <b>{t["about_dev_name"]}</b><br>
                {t["about_dev_dept"]}<br>
                {t["about_dev_year"]}<br><br>
                🔗 <a href="https://github.com/UMEDJAMALA" target="_blank" style="color: var(--primary); text-decoration: none;"><b>GitHub Profile (@UMEDJAMALA)</b></a><br>
                🔗 <a href="https://facebook.com/umedjamala" target="_blank" style="color: var(--primary); text-decoration: none;"><b>Facebook Profile</b></a><br>
                📧 <a href="mailto:umedjamal2005@gmail.com" style="color: var(--primary); text-decoration: none;"><b>Contact Email</b></a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="glass-panel" style="margin-bottom:0; height: 100%;">
            <div style="color: var(--primary); font-weight: 800; margin-bottom: 0.8rem; border-bottom: 1px solid var(--glass-border); padding-bottom: 5px;">⚙️ {t["about_tech_title"]}</div>
            <div style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 0.8rem;">{t["about_tech_intro"]}</div>
            <span class="tech-tag">Python 3</span>
            <span class="tech-tag">XGBoost</span>
            <span class="tech-tag">Scikit-learn</span>
            <span class="tech-tag">Streamlit</span>
            <span class="tech-tag">Plotly</span>
            <span class="tech-tag">Pandas</span>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FORMATTERS & PRINT PDF BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def format_late_history(val, lang):
    if val == 0:
        return "هیچ کات" if lang == 'ku' else "Never"
    elif val == 1:
        return "1 جار دواکەوتووە" if lang == 'ku' else "Delayed 1 time"
    elif val == 2:
        return "2 جار دواکەوتووە" if lang == 'ku' else "Delayed 2 times"
    elif val == 3:
        return "3 جار دواکەوتووە" if lang == 'ku' else "Delayed 3 times"
    elif val == 4:
        return "4 جار دواکەوتووە" if lang == 'ku' else "Delayed 4 times"
    else:
        return f"{val} جار دواکەوتووە" if lang == 'ku' else f"Delayed {val} times"

def generate_print_report(is_high, credit_limit, display_message, lang, inputs):
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t = translations[lang]
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="{lang}" dir="{'rtl' if lang=='ku' else 'ltr'}">
    <head>
        <meta charset="UTF-8">
        <title>Credit Risk Report - Erbil Warehouse</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;700&family=Inter:wght@400;700&display=swap');
            body {{
                font-family: { "'Noto Sans Arabic', sans-serif" if lang=='ku' else "'Inter', sans-serif" };
                padding: 40px;
                color: #000;
                background: #fff;
                line-height: 1.6;
            }}
            .header {{
                text-align: center;
                border-bottom: 3px double #113035;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0;
                font-size: 26px;
                color: #113035;
            }}
            .header p {{
                margin: 5px 0 0;
                font-size: 14px;
                color: #555;
            }}
            .results-box {{
                border: 2px solid #113035;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 30px;
                background-color: #fcfdfc;
            }}
            .results-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            .result-item {{
                padding: 10px;
                border-radius: 6px;
                background-color: #f4f6f5;
                text-align: center;
            }}
            .result-item h3 {{
                margin: 0 0 5px;
                font-size: 14px;
                color: #555;
                text-transform: uppercase;
            }}
            .result-item .val {{
                font-size: 24px;
                font-weight: bold;
                color: #113035;
            }}
            .table-title {{
                font-size: 18px;
                font-weight: bold;
                color: #113035;
                border-bottom: 1px solid #113035;
                padding-bottom: 5px;
                margin-top: 30px;
                margin-bottom: 15px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: {'right' if lang=='ku' else 'left'};
            }}
            th {{
                background-color: #f4f6f5;
                color: #113035;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                font-size: 11px;
                color: #777;
                border-top: 1px solid #eee;
                padding-top: 15px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>ERBIL WAREHOUSE RISK SYSTEM</h1>
            <p>B2B Credit Limit &amp; Advanced RFM Scoring Report</p>
            <p style="font-size: 11px; color: #888;">Generated on: {now_str}</p>
        </div>
        
        <div class="results-box">
            <div class="results-grid">
                <div class="result-item" style="border-top: 4px solid {'#ef4444' if is_high else '#10b981'};">
                    <h3>{t["res_risk_eyebrow"]}</h3>
                    <div class="val" style="color: {'#ef4444' if is_high else '#10b981'};">
                        {t["res_risk_high"] if is_high else t["res_risk_low"]}
                    </div>
                </div>
                <div class="result-item" style="border-top: 4px solid #113035;">
                    <h3>{t["res_limit_eyebrow"]}</h3>
                    <div class="val">${credit_limit:,.0f}</div>
                </div>
            </div>
            <p style="margin: 15px 0 0; text-align: center; font-size: 13px; font-weight: bold; color: #444;">
                {display_message}
            </p>
        </div>
        
        <div class="table-title">{t["summary_title"]}</div>
        <table>
            <thead>
                <tr>
                    <th>{ "تایبەتمەندی" if lang=='ku' else "Parameter" }</th>
                    <th>{ "بەها" if lang=='ku' else "Value" }</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>{t["sum_shop_age"]}</td>
                    <td>{inputs["shop_age"]} {t["unit_years"]}</td>
                </tr>
                <tr>
                    <td>{t["sum_last_order"]}</td>
                    <td>{inputs["days_since_last"]} {t["unit_days"]}</td>
                </tr>
                <tr>
                    <td>{t["sum_monthly_freq"]}</td>
                    <td>{inputs["freq_per_month"]} {t["unit_invoices"]}</td>
                </tr>
                <tr>
                    <td>{t["sum_avg_invoice"]}</td>
                    <td>${inputs["avg_invoice"]:,.0f}</td>
                </tr>
                <tr>
                    <td>{t["sum_total_volume"]}</td>
                    <td>${inputs["total_volume"]:,.0f}</td>
                </tr>
                <tr>
                    <td>{t["sum_unpaid_ratio"]}</td>
                    <td>{inputs["unpaid_ratio"] * 100:.0f}%</td>
                </tr>
                <tr>
                    <td>{t["sum_debt_ratio"]}</td>
                    <td>{inputs["debt_ratio"] * 100:.0f}%</td>
                </tr>
                <tr>
                    <td>{t["sum_late_history"]}</td>
                    <td>{format_late_history(inputs["late_history"], lang)}</td>
                </tr>
            </tbody>
        </table>
        
        <div class="footer">
            &copy; 2026 Erbil Warehouse B2B Credit Limit System. All rights reserved.<br>
            Developed by Umed Jamal Nouri | Academic Year: 2025 - 2026
        </div>
        
        <script>
            window.onload = function() {{
                window.print();
            }}
        </script>
    </body>
    </html>
    """
    return html_content

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════
if "avg_invoice" not in st.session_state:
    st.session_state.avg_invoice = 250.0
if "freq_per_month" not in st.session_state:
    st.session_state.freq_per_month = 5.0
if "total_volume" not in st.session_state:
    st.session_state.total_volume = 5000.0
if "unpaid_ratio" not in st.session_state:
    st.session_state.unpaid_ratio = 5
if "shop_age" not in st.session_state:
    st.session_state.shop_age = 5
if "days_since_last" not in st.session_state:
    st.session_state.days_since_last = 15
if "debt_ratio" not in st.session_state:
    st.session_state.debt_ratio = 15
if "late_history" not in st.session_state:
    st.session_state.late_history = 1
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

def reset_form():
    st.session_state.avg_invoice = 250.0
    st.session_state.freq_per_month = 5.0
    st.session_state.total_volume = 5000.0
    st.session_state.unpaid_ratio = 5
    st.session_state.shop_age = 5
    st.session_state.days_since_last = 15
    st.session_state.debt_ratio = 15
    st.session_state.late_history = 1
    st.session_state.analyzed = False
    st.toast(translations[current_lang]["toast_clear_msg"], icon="🧹")

# Update messages dynamically if language toggle happens after analysis
if st.session_state.analyzed:
    if st.session_state.rule_type == "security":
        if current_lang == "ku":
            st.session_state.display_message = "قەرزدان ڕەتکراوەتەوە بەهۆی دواکەوتنی زۆر لە پێشینەدا یان ڕێژەی بەرزی وەسڵە نەدراوەکان (سەروو ٦٠٪)."
        else:
            st.session_state.display_message = "Credit recommendation denied due to excessive payment delays or high unpaid ratio (above 60%)."
    elif st.session_state.rule_type == "rule":
        if current_lang == "ku":
            st.session_state.display_message = "سیستەمی پاراستنی خێرا (Rule Engine) چالاک کراوە بەهۆی نەبوونی داتای مێژوویی پێویست (کۆڵد ستارت)."
        else:
            st.session_state.display_message = "Rule engine triggered due to lack of historical business transactions (Cold Start)."
    else:
        if current_lang == "ku":
            st.session_state.display_message = "بڕیاری کۆتایی لە ڕێگەی ئەلگۆریتمی XGBoost دەرکراوە"
        else:
            st.session_state.display_message = "Final decision rendered via XGBoost algorithm."

# ══════════════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero glass-panel">
    <div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin-bottom: 1rem;">
        <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="logoGrad" x1="0" y1="0" x2="64" y2="64">
                    <stop offset="0%" stop-color="#3b5664"/>
                    <stop offset="50%" stop-color="#cbe3b5"/>
                    <stop offset="100%" stop-color="#ffd280"/>
                </linearGradient>
            </defs>
            <path d="M32 4C18 4 12 10 12 24C12 38 24 54 32 58C40 54 52 38 52 24C52 10 46 4 32 4Z" stroke="url(#logoGrad)" stroke-width="2.5" fill="none"/>
            <path d="M32 16L44 22L32 28L20 22L32 16Z" fill="#cbe3b5" opacity="0.3"/>
            <path d="M32 16L44 22L32 28L20 22L32 16Z" stroke="#113035" stroke-width="1.5" fill="none"/>
            <path d="M20 22V36L32 42V28L20 22Z" fill="#3b5664" opacity="0.4"/>
            <path d="M20 22V36L32 42V28L20 22Z" stroke="#113035" stroke-width="1.5" fill="none"/>
            <path d="M32 28V42L44 36V22L32 28Z" fill="#ffd280" opacity="0.2"/>
            <path d="M32 28V42L44 36V22L32 28Z" stroke="#113035" stroke-width="1.5" fill="none"/>
            <path d="M16 38L26 30L34 35L48 20" stroke="#ffd280" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="48" cy="20" r="3.5" fill="#ffd280" stroke="#fff" stroke-width="1"/>
            <circle cx="26" cy="30" r="2" fill="#cbe3b5"/>
            <circle cx="34" cy="35" r="2" fill="#cbe3b5"/>
        </svg>
    </div>
    <h1 class="hero-title">{translations[current_lang]["hero_title"]}</h1>
    <p class="hero-sub">{translations[current_lang]["hero_sub"]}</p>
    <div class="hero-pill">⚡ {translations[current_lang]["hero_pill"]}</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ACTION BUTTONS (TOP)
# ══════════════════════════════════════════════════════════════════════════════
ab_col1, ab_col2, ab_col3 = st.columns(3, gap="medium")

with ab_col1:
    if st.button(translations[current_lang]["btn_about"], use_container_width=True, type="secondary"):
        project_info_dialog(current_lang)
with ab_col2:
    if st.button(translations[current_lang]["btn_dataset"], use_container_width=True, type="secondary"):
        dataset_model_info_dialog(current_lang)
with ab_col3:
    if st.button(translations[current_lang]["btn_eval"], use_container_width=True, type="secondary"):
        model_evaluation_dialog(current_lang)

if not models_loaded:
    st.markdown(f"""
    <div style="background: rgba(239,68,68,0.1); border: 1px solid var(--red); color: var(--red); border-radius: 12px; padding: 1rem; text-align: center; margin-top: 1.5rem; font-weight: 700;">
        ⚠️ Models failed to load. Operating in fallback prediction mode.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  INPUT FORM PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)

col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown(f'<div class="sec-head"><span class="sec-head-text">💰 {translations[current_lang]["sec_financial"]}</span><span class="sec-head-line"></span></div>', unsafe_allow_html=True)
    
    avg_invoice = st.number_input(
        translations[current_lang]["label_avg_invoice"], 
        min_value=0.0, 
        step=10.0, 
        format="%.0f",
        key="avg_invoice"
    )
    st.markdown(f'<div class="helper-text">{translations[current_lang]["helper_avg_invoice"]}</div>', unsafe_allow_html=True)
    
    freq_per_month = st.number_input(
        translations[current_lang]["label_freq_per_month"], 
        min_value=0.0, 
        step=1.0,
        key="freq_per_month"
    )
    st.markdown(f'<div class="helper-text">{translations[current_lang]["helper_freq_per_month"]}</div>', unsafe_allow_html=True)
    
    total_volume = st.number_input(
        translations[current_lang]["label_total_volume"], 
        min_value=0.0, 
        step=100.0, 
        format="%.0f",
        key="total_volume"
    )
    st.markdown(f'<div class="helper-text">{translations[current_lang]["helper_total_volume"]}</div>', unsafe_allow_html=True)
    
    unpaid_ratio_display = st.slider(
        translations[current_lang]["label_unpaid_ratio"], 
        min_value=0, 
        max_value=100, 
        step=1,
        key="unpaid_ratio"
    )
    st.markdown(f'<div class="helper-text">{translations[current_lang]["helper_unpaid_ratio"]}</div>', unsafe_allow_html=True)
    unpaid_ratio = unpaid_ratio_display / 100.0

with col_r:
    st.markdown(f'<div class="sec-head"><span class="sec-head-text">🏢 {translations[current_lang]["sec_shop"]}</span><span class="sec-head-line"></span></div>', unsafe_allow_html=True)
    
    shop_age = st.slider(
        translations[current_lang]["label_shop_age"], 
        min_value=0, 
        max_value=50, 
        step=1,
        key="shop_age"
    )
    st.markdown(f'<div class="helper-text">{translations[current_lang]["helper_shop_age"]}</div>', unsafe_allow_html=True)
    
    days_since_last = st.number_input(
        translations[current_lang]["label_days_since_last"], 
        min_value=0, 
        step=1,
        key="days_since_last"
    )
    st.markdown(f'<div class="helper-text">{translations[current_lang]["helper_days_since_last"]}</div>', unsafe_allow_html=True)
    
    debt_ratio_display = st.slider(
        translations[current_lang]["label_debt_ratio"], 
        min_value=0, 
        max_value=100, 
        step=1,
        key="debt_ratio"
    )
    st.markdown(f'<div class="helper-text">{translations[current_lang]["helper_debt_ratio"]}</div>', unsafe_allow_html=True)
    debt_ratio = debt_ratio_display / 100.0
    
    late_history = st.selectbox(
        translations[current_lang]["label_late_history"], 
        options=list(range(31)), 
        format_func=lambda x: format_late_history(x, current_lang),
        key="late_history"
    )
    st.markdown(f'<div class="helper-text">{translations[current_lang]["helper_late_history"]}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SUBMIT & CLEAR BUTTONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])

with btn_col2:
    submit_pressed = st.button(
        translations[current_lang]["btn_submit"], 
        type="primary", 
        use_container_width=True
    )
    
    if st.session_state.analyzed:
        st.button(
            translations[current_lang]["btn_clear"], 
            type="secondary", 
            use_container_width=True,
            on_click=reset_form
        )

# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS & PREDICTION RUN
# ══════════════════════════════════════════════════════════════════════════════
if submit_pressed:
    st.session_state.analyzed = True
    
    # 1. Severe Outliers (Blocked)
    if unpaid_ratio > 0.60 or late_history >= 5:
        st.session_state.credit_limit = 0.0
        st.session_state.is_high_risk = True
        st.session_state.rule_triggered = True
        st.session_state.rule_type = "security"
        if current_lang == "ku":
            st.session_state.display_message = "قەرزدان ڕەتکراوەتەوە بەهۆی دواکەوتنی زۆر لە پێشینەدا یان ڕێژەی بەرزی وەسڵە نەدراوەکان (سەروو ٦٠٪)."
        else:
            st.session_state.display_message = "Credit recommendation denied due to excessive payment delays or high unpaid ratio (above 60%)."
            
    # 2. Cold Start (New Customer)
    elif shop_age < 1 or freq_per_month == 0:
        st.session_state.credit_limit = 150.0
        st.session_state.is_high_risk = True
        st.session_state.rule_triggered = True
        st.session_state.rule_type = "rule"
        if current_lang == "ku":
            st.session_state.display_message = "سیستەمی پاراستنی خێرا (Rule Engine) چالاک کراوە بەهۆی نەبوونی داتای مێژوویی پێویست (کۆڵد ستارت)."
        else:
            st.session_state.display_message = "Rule engine triggered due to lack of historical business transactions (Cold Start)."
            
    # 3. Normal AI Flow
    else:
        if models_loaded:
            try:
                clf_features = np.array([[shop_age, days_since_last, freq_per_month, avg_invoice, total_volume, unpaid_ratio, debt_ratio, late_history]])
                fs_clf = scaler_clf.transform(clf_features)
                risk_pred = risk_model.predict(fs_clf)[0]
                is_high = int(risk_pred) == 1
                
                reg_features = np.array([[shop_age, days_since_last, freq_per_month, avg_invoice, total_volume, unpaid_ratio, debt_ratio, late_history, risk_pred]])
                fs_reg = scaler_reg.transform(reg_features)
                limit_pred = limit_model.predict(fs_reg)[0]
                credit_limit = max(0.0, float(limit_pred))
                
                st.session_state.credit_limit = credit_limit
                st.session_state.is_high_risk = is_high
                st.session_state.rule_triggered = False
                st.session_state.rule_type = "ml"
                if current_lang == "ku":
                    st.session_state.display_message = "بڕیاری کۆتایی لە ڕێگەی ئەلگۆریتمی XGBoost دەرکراوە"
                else:
                    st.session_state.display_message = "Final decision rendered via XGBoost algorithm."
            except Exception as e:
                st.session_state.credit_limit = max(0.0, avg_invoice * freq_per_month * 0.5)
                st.session_state.is_high_risk = debt_ratio > 0.4
                st.session_state.rule_triggered = False
                st.session_state.rule_type = "ml"
                st.session_state.display_message = f"Error: {e}"
        else:
            # Fallback calculations if models are not loaded
            st.session_state.credit_limit = max(0.0, avg_invoice * freq_per_month * 0.5)
            st.session_state.is_high_risk = debt_ratio > 0.4 or unpaid_ratio > 0.3 or late_history > 3
            st.session_state.rule_triggered = False
            st.session_state.rule_type = "ml"
            if current_lang == "ku":
                st.session_state.display_message = "بڕیاری کۆتایی لە ڕێگەی ئەلگۆریتمی XGBoost دەرکراوە (Fallback Mode)"
            else:
                st.session_state.display_message = "Final decision rendered via XGBoost algorithm (Fallback Mode)."

    # Streamlit native Toast alerts
    if st.session_state.rule_type == "security":
        st.toast(translations[current_lang]["toast_security_msg"], icon="🚨")
    elif st.session_state.rule_type == "rule":
        st.toast(translations[current_lang]["toast_rule_msg"], icon="⚠️")
    else:
        st.toast(translations[current_lang]["toast_success_msg"], icon="🎯")

# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS SECTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.analyzed:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="sec-head"><span class="sec-head-text">📊 {translations[current_lang]["results_title"]}</span><span class="sec-head-line"></span></div>', unsafe_allow_html=True)
    
    # 1. Safety Badge
    rule_tr = st.session_state.rule_triggered
    cr_lim = st.session_state.credit_limit
    disp_msg = st.session_state.display_message
    
    if not rule_tr:
        badge_title = translations[current_lang]['badge_ml_title']
        badge_icon = "fa-brain"
        badge_border = "rgba(16, 185, 129, 0.3)"
        badge_color = "#10b981"
    elif cr_lim > 0:
        badge_title = translations[current_lang]['badge_rule_title']
        badge_icon = "fa-shield-halved"
        badge_border = "rgba(243, 163, 50, 0.3)"
        badge_color = "#f3a332"
    else:
        badge_title = translations[current_lang]['badge_security_title']
        badge_icon = "fa-shield-virus"
        badge_border = "rgba(239, 68, 68, 0.3)"
        badge_color = "#ef4444"
        
    st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 1rem; background: rgba(8, 22, 25, 0.6); border: 1px solid {badge_border}; border-radius: 16px; max-width: 450px; margin: 0 auto 1.5rem; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.3); backdrop-filter: blur(10px);">
        <div style="display: flex; align-items: center; justify-content: center; gap: 10px; font-weight: bold; color: {badge_color}; font-size: 0.95rem;">
            <i class="fa-solid {badge_icon}" style="font-size: 1.1rem;"></i>
            <span>{badge_title}</span>
        </div>
        <div style="font-size: 0.8rem; color: #8ca2a4; margin-top: 8px; font-weight: 600;">
            {disp_msg}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Risk & Limit Cards
    rc1, rc2 = st.columns(2, gap="large")
    
    with rc1:
        if st.session_state.is_high_risk:
            st.markdown(f"""
            <div class="result-card high-risk">
                <div class="rc-eyebrow">⚠️ {translations[current_lang]["res_risk_eyebrow"]}</div>
                <div class="rc-value">{translations[current_lang]["res_risk_high"]}</div>
                <div class="rc-en">HIGH RISK</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card low-risk">
                <div class="rc-eyebrow">✅ {translations[current_lang]["res_risk_eyebrow"]}</div>
                <div class="rc-value">{translations[current_lang]["res_risk_low"]}</div>
                <div class="rc-en">LOW RISK</div>
            </div>
            """, unsafe_allow_html=True)
            
    with rc2:
        st.markdown(f"""
        <div class="result-card limit-card">
            <div class="rc-eyebrow">💳 {translations[current_lang]["res_limit_eyebrow"]}</div>
            <div class="rc-value">${st.session_state.credit_limit:,.0f}</div>
            <div class="rc-en">Approved Credit Limit</div>
        </div>
        """, unsafe_allow_html=True)
        
    # 3. Parameters Summary Table
    st.markdown(f'<h3 style="font-size: 1.1rem; font-weight: 800; margin-top: 2rem; margin-bottom: 1rem;">{translations[current_lang]["summary_title"]}</h3>', unsafe_allow_html=True)
    
    inputs_dict = {
        "shop_age": shop_age,
        "days_since_last": days_since_last,
        "freq_per_month": freq_per_month,
        "avg_invoice": avg_invoice,
        "total_volume": total_volume,
        "unpaid_ratio": unpaid_ratio,
        "debt_ratio": debt_ratio,
        "late_history": late_history
    }
    
    summary_table_html = f"""
    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid var(--glass-border); border-radius: 12px; padding: 1rem; overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; text-align: {align};">
            <thead>
                <tr style="border-bottom: 2px solid var(--glass-border);">
                    <th style="padding: 10px; color: var(--primary);">{ "تایبەتمەندی" if current_lang=='ku' else "Parameter" }</th>
                    <th style="padding: 10px; color: var(--primary); text-align: {'left' if current_lang=='ku' else 'right'};">{ "بەها" if current_lang=='ku' else "Value" }</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 10px; color: var(--text-muted);">{translations[current_lang]["sum_shop_age"]}</td>
                    <td style="padding: 10px; font-weight: bold; text-align: {'left' if current_lang=='ku' else 'right'};">{shop_age} {translations[current_lang]["unit_years"]}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 10px; color: var(--text-muted);">{translations[current_lang]["sum_last_order"]}</td>
                    <td style="padding: 10px; font-weight: bold; text-align: {'left' if current_lang=='ku' else 'right'};">{days_since_last} {translations[current_lang]["unit_days"]}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 10px; color: var(--text-muted);">{translations[current_lang]["sum_monthly_freq"]}</td>
                    <td style="padding: 10px; font-weight: bold; text-align: {'left' if current_lang=='ku' else 'right'};">{freq_per_month} {translations[current_lang]["unit_invoices"]}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 10px; color: var(--text-muted);">{translations[current_lang]["sum_avg_invoice"]}</td>
                    <td style="padding: 10px; font-weight: bold; text-align: {'left' if current_lang=='ku' else 'right'};">${avg_invoice:,.0f}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 10px; color: var(--text-muted);">{translations[current_lang]["sum_total_volume"]}</td>
                    <td style="padding: 10px; font-weight: bold; text-align: {'left' if current_lang=='ku' else 'right'};">${total_volume:,.0f}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 10px; color: var(--text-muted);">{translations[current_lang]["sum_unpaid_ratio"]}</td>
                    <td style="padding: 10px; font-weight: bold; text-align: {'left' if current_lang=='ku' else 'right'};">{unpaid_ratio_display}%</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 10px; color: var(--text-muted);">{translations[current_lang]["sum_debt_ratio"]}</td>
                    <td style="padding: 10px; font-weight: bold; text-align: {'left' if current_lang=='ku' else 'right'};">{debt_ratio_display}%</td>
                </tr>
                <tr>
                    <td style="padding: 10px; color: var(--text-muted);">{translations[current_lang]["sum_late_history"]}</td>
                    <td style="padding: 10px; font-weight: bold; text-align: {'left' if current_lang=='ku' else 'right'};">{format_late_history(late_history, current_lang)}</td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    st.markdown(summary_table_html, unsafe_allow_html=True)
    
    # 4. Print PDF Action Trigger Button
    st.markdown("<br>", unsafe_allow_html=True)
    report_html = generate_print_report(st.session_state.is_high_risk, st.session_state.credit_limit, st.session_state.display_message, current_lang, inputs_dict)
    b64_report = base64.b64encode(report_html.encode('utf-8')).decode('utf-8')
    print_btn_html = f"""
    <a href="data:text/html;base64,{b64_report}" target="_blank" style="text-decoration: none;">
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--glass-border);
            color: var(--text-muted);
            font-family: inherit;
            font-size: 0.95rem;
            font-weight: 700;
            border-radius: 14px;
            padding: 0.85rem 1.2rem;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        " onmouseover="this.style.background='rgba(167, 203, 142, 0.1)'; this.style.color='var(--primary)';" onmouseout="this.style.background='rgba(255, 255, 255, 0.03)'; this.style.color='var(--text-muted)';">
            🖨️ { "چاپکردن / داونلۆدی PDF" if current_lang=='ku' else "Print / Download PDF" }
        </div>
    </a>
    """
    st.markdown(print_btn_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="text-align: center; padding: 2.5rem 0 1rem; color: rgba(255,255,255,0.4); font-size: 0.8rem; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 3rem; font-family: 'Inter', sans-serif; direction: ltr;">
    &copy; 2026 Erbil Warehouse B2B Credit Limit System.<br>
    Developed by <strong>Umed Jamal Nouri</strong> | Advanced RFM Edition
</div>
""", unsafe_allow_html=True)

# --- END OF FILE app.py ---