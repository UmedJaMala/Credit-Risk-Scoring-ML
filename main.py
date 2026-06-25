from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI(title="Erbil Warehouse Risk API", version="3.0")

# ڕێگەدان بە پەیوەندی فڕۆنت-ئێند بۆ وەرگرتنی داتا (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# خوێندنەوەی مۆدێلە باشترکراوەکان (Improved Models)
BASE_DIR = "outputs"
rm = joblib.load(os.path.join(BASE_DIR, "risk_model_improved.joblib"))
lm = joblib.load(os.path.join(BASE_DIR, "limit_model_improved.joblib"))
sc_clf = joblib.load(os.path.join(BASE_DIR, "scaler_clf_improved.joblib"))
sc_reg = joblib.load(os.path.join(BASE_DIR, "scaler_reg_improved.joblib"))

# دیاریکردنی جۆری داتاکان بەپێی ٨ فیچەرەکەی فۆڕمەکە
class CustomerData(BaseModel):
    shop_age: int
    days_since_last: int
    freq_per_month: float
    avg_invoice: float
    total_volume: float
    unpaid_ratio: float  # وەکو ڕێژە دێت بۆ نموونە 0.05
    debt_ratio: float    # وەکو ڕێژە دێت بۆ نموونە 0.15
    late_history: int

@app.post("/predict")
def predict_credit(data: CustomerData):
    # ١. ڕیزبەندی داتاکان ڕێک بەپێی نۆتبووکەکەت
    clf_features = np.array([[
        data.shop_age, data.days_since_last, data.freq_per_month, 
        data.avg_invoice, data.total_volume, data.unpaid_ratio, 
        data.debt_ratio, data.late_history
    ]])
    
    # پێشبینیکردنی مەترسی (Classification)
    fs_clf = sc_clf.transform(clf_features)
    risk_pred = rm.predict(fs_clf)[0]
    is_high = int(risk_pred) == 1
    
    # ٢. ئامادەکردنی داتا بۆ Regression (زیادکردنی دەرەنجامی مەترسییەکە)
    reg_features = np.array([[
        data.shop_age, data.days_since_last, data.freq_per_month, 
        data.avg_invoice, data.total_volume, data.unpaid_ratio, 
        data.debt_ratio, data.late_history, risk_pred
    ]])
    
    # پێشبینیکردنی بڕی قەرز (Regression)
    fs_reg = sc_reg.transform(reg_features)
    limit_pred = float(lm.predict(fs_reg)[0])
    credit_limit = max(0.0, limit_pred)

    return {
        "is_high_risk": is_high,
        "credit_limit": credit_limit
    }