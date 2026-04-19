#!/usr/bin/env python
# coding: utf-8

# # XGBoost: Credit Risk Scoring & Credit Limit Prediction
# 
# **Objective:** Classify customer risk and predict credit limits
# - **Task 1:** Risk Classification (High/Low)
# - **Task 2:** Credit Limit Regression (USD amount)

# In[16]:


import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('outputs', exist_ok=True)


# ## Dataset Generation

# In[17]:


def generate_erbil_warehouse_dataset(n_samples=1000):
    rng = np.random.default_rng(42)

    # 1. تەمەنی دوکان یان کڕیار بە ساڵ (1 بۆ 20 ساڵ)
    shop_age_years = rng.integers(1, 21, size=n_samples)

    # 2. کۆی گشتی پسوولەکان لای کۆگاکەمان
    total_invoices = rng.integers(5, 500, size=n_samples)

    # 3. تێکڕای بەهای هەر پسوولەیەک (بە دۆلار: 200$ تا 15,000$)
    avg_invoice_value = rng.uniform(200, 15000, size=n_samples).round(-1)

    # 4. ژمارەی ئەو پسوولانەی پارەیان نەدراوە (قەرز)
    unpaid_invoices_count = rng.integers(0, 15, size=n_samples)
    # دڵنیابوونەوە لەوەی پسوولەی نەدراو لە کۆی پسوولەکان زیاتر نەبێت
    unpaid_invoices_count = np.minimum(unpaid_invoices_count, total_invoices)

    # 5. بڕی قەرزی ئێستا (دەگۆڕێت بەپێی ژمارەی پسوولە نەدراوەکان)
    current_debt = (unpaid_invoices_count * avg_invoice_value * rng.uniform(0.6, 1.2, size=n_samples)).round(-1)

    # 6. مێژووی دواکەوتنی پارەدان (بەپێی یاسای Poisson Distribution)
    late_payment_history = rng.poisson(lam=1.5, size=n_samples)

    # --- لۆژیکی زانستی بۆ Risk Score ---
    # مەترسی ڕاستەوانە دەگۆڕێت لەگەڵ ڕێژەی قەرز، وە پێچەوانە دەگۆڕێت لەگەڵ تەمەنی دوکانەکە
    debt_ratio = np.where(total_invoices > 0, unpaid_invoices_count / total_invoices, 0)

    risk_logit = (-2.5 
                  + 6.0 * debt_ratio 
                  + 0.8 * late_payment_history 
                  - 0.15 * shop_age_years 
                  - 0.002 * total_invoices 
                  + rng.normal(0, 0.4, n_samples))

    # بەکارهێنانی Sigmoid بۆ گۆڕینی لۆجیتەکە بۆ ئەگەری نێوان 0 و 1
    risk_prob = 1 / (1 + np.exp(-risk_logit))
    high_risk = (rng.uniform(size=n_samples) < risk_prob).astype(int)

    # --- لۆژیکی Credit Limit ---
    # لیمیتەکە لەسەر بنەمای بەهای پسوولەکان دەدرێت، بەڵام قەرز و مەترسی کەمی دەکەنەوە
    risk_multiplier = np.where(high_risk == 1, 0.3, 1.2)
    base_limit = (avg_invoice_value * 8) + (shop_age_years * 500)

    credit_limit = (base_limit * risk_multiplier) - (current_debt * 0.4) + rng.normal(0, 500, n_samples)
    credit_limit = np.clip(credit_limit, 500, 100000).round(-2) # نابێت لە 500$ کەمتر بێت

    return pd.DataFrame({
        'Shop_Age_Years': shop_age_years,
        'Total_Invoices': total_invoices,
        'Average_Invoice_Value': avg_invoice_value,
        'Unpaid_Invoices_Count': unpaid_invoices_count,
        'Current_Debt': current_debt,
        'Late_Payment_History': late_payment_history,
        'High_Risk': high_risk,
        'Credit_Limit': credit_limit,
    })

df = generate_erbil_warehouse_dataset(1000)

# --- SMOTE OVERSAMPLING ---
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_temp = df.drop(columns=['High_Risk'])
y_temp = df['High_Risk']
X_res, y_res = smote.fit_resample(X_temp, y_temp)
df = pd.DataFrame(X_res, columns=X_temp.columns)
df['High_Risk'] = y_res
# --------------------------
df.to_csv('outputs/erbil_warehouse_dataset.csv', index=False)
print(f'Dataset: {df.shape} | Class: {dict(df["High_Risk"].value_counts())}')


# ## Preprocessing & Split

# In[18]:


# دیاریکردنی تایبەتمەندییە نوێیەکان بۆ ڕاهێنان (Features)
X_clf = df[['Shop_Age_Years', 'Total_Invoices', 'Average_Invoice_Value', 
        'Unpaid_Invoices_Count', 'Current_Debt', 'Late_Payment_History']].values

# زیادکردنی High_Risk بۆ مۆدێلی قەرز بۆ ئەوەی پێشبینییەکە زۆر وردتر بێت
X_reg = df[['Shop_Age_Years', 'Total_Invoices', 'Average_Invoice_Value', 
        'Unpaid_Invoices_Count', 'Current_Debt', 'Late_Payment_History', 'High_Risk']].values

# دیاریکردنی ئامانجەکان (Targets)
y_risk = df['High_Risk'].values
y_credit = df['Credit_Limit'].values

# دابەشکردنی داتاکە بۆ ڕاهێنان و تاقیکردنەوە (80% بۆ ڕاهێنان، 20% بۆ تاقیکردنەوە)
X_train_clf, X_test_clf, X_train_reg, X_test_reg, y_risk_train, y_risk_test, y_credit_train, y_credit_test = train_test_split(
    X_clf, X_reg, y_risk, y_credit, test_size=0.20, random_state=42, stratify=y_risk)

# پێوانەکردنی داتاکان (Scaling) بۆ ئەوەی جیاوازی زۆر لە نێوان ژمارەکان کاریگەری نەرێنی دروست نەکات
scaler_clf = StandardScaler()
X_train_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_scaled = scaler_clf.transform(X_test_clf)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)


# ## Model Training

# In[19]:


import joblib
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# ١. دۆزینەوەی کێشی داتا نابەرامبەرەکان (Class Imbalance Ratio)
negative_class = sum(y_risk_train == 0)
positive_class = sum(y_risk_train == 1)
scale_weight = negative_class / positive_class if positive_class > 0 else 1

print(f"ئامادەکردنی مۆدێل... کێشی کڕیارە مەترسیدارەکان کرا بە: {scale_weight:.2f} هێندە")

# Creating an evaluation set (10% of training data) for Early Stopping
X_train_clf_split, X_val_clf, y_risk_train_split, y_risk_val = train_test_split(
    X_train_scaled, y_risk_train, test_size=0.10, random_state=42, stratify=y_risk_train)

X_train_reg_split, X_val_reg, y_credit_train_split, y_credit_val = train_test_split(
    X_train_reg_scaled, y_credit_train, test_size=0.10, random_state=42)

# ٢. مۆدێلی پۆلێنکردنی مەترسی (Risk Classifier) بە ڕێکخستنی پێشکەوتووەوە
print("\n" + "="*45)
print("🚀 OPTIMIZING RISK CLASSIFIER")
print("="*45)

xgb_clf_base = XGBClassifier(
    n_estimators=500,
    scale_pos_weight=scale_weight,
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42,
    verbosity=0
)

param_grid_clf = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [3, 5, 7],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'alpha': [0.1, 0.5, 1.0],
    'lambda': [1.0, 2.0, 5.0]
}

clf_search = RandomizedSearchCV(
    estimator=xgb_clf_base, param_distributions=param_grid_clf,
    n_iter=20, scoring='roc_auc', cv=5, verbose=1, random_state=42, n_jobs=-1
)

fit_params_clf = {"eval_set": [(X_val_clf, y_risk_val)], "verbose": False}
clf_search.fit(X_train_clf_split, y_risk_train_split, **fit_params_clf)

print(f"✅ Best Classifier Parameters: {clf_search.best_params_}")
xgb_clf = clf_search.best_estimator_

# ٣. مۆدێلی پێشبینیکردنی بڕی قەرز (Credit Limit Regressor)
print("\n" + "="*45)
print("🚀 OPTIMIZING CREDIT LIMIT REGRESSOR")
print("="*45)

xgb_reg_base = XGBRegressor(
    n_estimators=500,
    eval_metric='rmse',
    early_stopping_rounds=50,
    random_state=42,
    verbosity=0
)

param_grid_reg = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [3, 5, 7],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'alpha': [0.5, 1.0, 2.0],
    'lambda': [2.0, 3.0, 5.0]
}

reg_search = RandomizedSearchCV(
    estimator=xgb_reg_base, param_distributions=param_grid_reg,
    n_iter=20, scoring='r2', cv=5, verbose=1, random_state=42, n_jobs=-1
)

fit_params_reg = {"eval_set": [(X_val_reg, y_credit_val)], "verbose": False}
reg_search.fit(X_train_reg_split, y_credit_train_split, **fit_params_reg)

print(f"✅ Best Regressor Parameters: {reg_search.best_params_}")
xgb_reg = reg_search.best_estimator_

# ٤. پاشەکەوتکردنی مۆدێلە باشترکراوەکان بۆ ناو فۆڵدەری outputs
joblib.dump(xgb_clf, 'outputs/risk_model_improved.joblib')
joblib.dump(xgb_reg, 'outputs/limit_model_improved.joblib')
joblib.dump(scaler_clf, 'outputs/scaler_clf_improved.joblib')
joblib.dump(scaler_reg, 'outputs/scaler_reg_improved.joblib')

print("\n✅ مۆدێلەکان بە سەرکەوتوویی ڕاهێنران و پاشەکەوتکران.")


# ## Results

# In[20]:


y_risk_pred = xgb_clf.predict(X_test_scaled)
y_risk_proba = xgb_clf.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_risk_test, y_risk_pred)
prec = precision_score(y_risk_test, y_risk_pred)
rec = recall_score(y_risk_test, y_risk_pred)
f1 = f1_score(y_risk_test, y_risk_pred)
auc = roc_auc_score(y_risk_test, y_risk_proba)

print('CLASSIFICATION (Risk Scoring)')
print('='*40)
print(f'Accuracy : {acc:.4f} | Precision: {prec:.4f}')
print(f'Recall   : {rec:.4f} | F1-Score : {f1:.4f}')
print(f'ROC-AUC  : {auc:.4f}\n')
print(classification_report(y_risk_test, y_risk_pred, target_names=['Low Risk', 'High Risk']))


# In[21]:


y_credit_pred = xgb_reg.predict(X_test_reg_scaled)

mse = mean_squared_error(y_credit_test, y_credit_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_credit_test, y_credit_pred)
r2 = r2_score(y_credit_test, y_credit_pred)

print('REGRESSION (Credit Limit Prediction)')
print('='*40)
print(f'MSE  : ${mse:,.2f}')
print(f'RMSE : ${rmse:,.2f}')
print(f'MAE  : ${mae:,.2f}')
print(f'R²   : {r2:.4f}')


# In[22]:


# Evaluate on training set to check for overfitting
y_risk_train_pred = xgb_clf.predict(X_train_scaled)
y_risk_train_proba = xgb_clf.predict_proba(X_train_scaled)[:, 1]

train_acc = accuracy_score(y_risk_train, y_risk_train_pred)
train_f1 = f1_score(y_risk_train, y_risk_train_pred)
train_auc = roc_auc_score(y_risk_train, y_risk_train_proba)

print('TRAINING CLASSIFICATION (Risk Scoring)')
print('='*40)
print(f'Accuracy : {train_acc:.4f} | F1-Score : {train_f1:.4f}')
print(f'ROC-AUC  : {train_auc:.4f}\n')

y_credit_train_pred = xgb_reg.predict(X_train_reg_scaled)

train_mse = mean_squared_error(y_credit_train, y_credit_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_credit_train, y_credit_train_pred)
train_r2 = r2_score(y_credit_train, y_credit_train_pred)

print('TRAINING REGRESSION (Credit Limit Prediction)')
print('='*40)
print(f'MSE  : ${train_mse:,.2f}')
print(f'RMSE : ${train_rmse:,.2f}')
print(f'MAE  : ${train_mae:,.2f}')
print(f'R²   : {train_r2:.4f}')

print('\nOVERFITTING CHECK:')
print('='*40)
print('If training metrics are significantly better than test metrics, the model may be overfitting.')
print(f'Classification - Train Acc: {train_acc:.4f} vs Test Acc: {acc:.4f} (Gap: {train_acc - acc:.4f})')
print(f'Classification - Train AUC: {train_auc:.4f} vs Test AUC: {auc:.4f} (Gap: {train_auc - auc:.4f})')
print(f'Regression - Train R²: {train_r2:.4f} vs Test R²: {r2:.4f} (Gap: {train_r2 - r2:.4f})')


# In[23]:


import numpy as np

def evaluate_new_customer(shop_age, total_invoices, avg_invoice_val, unpaid_count, current_debt, late_history):
    # ١. کۆکردنەوەی داتاکانی کڕیارەکە
    customer_data = np.array([[shop_age, total_invoices, avg_invoice_val, 
                               unpaid_count, current_debt, late_history]])

    # ٢. پێوانەکردنی داتاکە (Scaling) بە هەمان ئەو پێوەرەی لە ڕاهێناندا بەکارمان هێنا
    customer_scaled = scaler_clf.transform(customer_data)

    # ٣. وەرگرتنی پێشبینییەکان لە هەردوو مۆدێلەکە
    risk_pred = xgb_clf.predict(customer_scaled)[0]
    risk_prob = xgb_clf.predict_proba(customer_scaled)[0][1] # ئەگەری مەترسی بە لەسەدا
    # ئامادەکردنی داتای قەرز بە زیادکردنی پێشبینی مەترسی
    customer_reg_data = np.array([[shop_age, total_invoices, avg_invoice_val, 
                                   unpaid_count, current_debt, late_history, risk_pred]])
    customer_reg_scaled = scaler_reg.transform(customer_reg_data)
    limit_pred = xgb_reg.predict(customer_reg_scaled)[0]

    # ٤. ڕێکخستنی ئەنجامەکان بۆ خوێندنەوە
    risk_status = "بەرزە (High Risk) ⚠️" if risk_pred == 1 else "نزمە و پارێزراوە (Low Risk) ✅"

    print("="*45)
    print("📊 ئەنجامی هەڵسەنگاندنی کڕیاری نوێ")
    print("="*45)
    print(f"ئاستی مەترسی:        {risk_status}")
    print(f"ئەگەری مەترسی:       {risk_prob * 100:.1f}%")
    print(f"باشترین لیمیتی قەرز: ${limit_pred:,.2f}")
    print("="*45 + "\n")

# -- تاقیکردنەوەی یەکەم (کڕیارێکی جێگای متمانە و کۆن) --
# تەمەنی دوکان 10 ساڵ، 150 پسوولە، تێکڕای 2000 دۆلار، تەنها 1 پسوولەی نەدراو، 1500 قەرز، 0 دواکەوتن
print("تاقیکردنەوەی ١: کڕیارێکی باش")
evaluate_new_customer(shop_age=10, total_invoices=150, avg_invoice_val=2000, 
                      unpaid_count=1, current_debt=1500, late_history=0)

# -- تاقیکردنەوەی دووەم (کڕیارێکی نوێ و مەترسیدار) --
# تەمەنی دوکان 1 ساڵ، 10 پسوولە، تێکڕا 800 دۆلار، 6 پسوولەی نەدراو، 4500 قەرز، 3 دواکەوتن
print("تاقیکردنەوەی ٢: کڕیارێکی مەترسیدار")
evaluate_new_customer(shop_age=1, total_invoices=10, avg_invoice_val=800, 
                      unpaid_count=6, current_debt=4500, late_history=3)


# ## Visualization

# In[24]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('XGBoost: Risk Scoring & Credit Limit Prediction', fontsize=13, fontweight='bold')

cm = confusion_matrix(y_risk_test, y_risk_pred)
ConfusionMatrixDisplay(cm, display_labels=['Low Risk', 'High Risk']).plot(ax=axes[0], colorbar=True, cmap='Blues')
axes[0].set_title('Risk Classification', fontweight='bold')

axes[1].scatter(y_credit_test, y_credit_pred, alpha=0.5, s=30, color='#1f77b4', edgecolors='white', linewidth=0.5)
lims = [min(y_credit_test.min(), y_credit_pred.min()), max(y_credit_test.max(), y_credit_pred.max())]
axes[1].plot(lims, lims, 'r--', lw=2, label='Perfect Fit')
axes[1].set_xlabel('Actual Credit Limit ($)', fontweight='bold')
axes[1].set_ylabel('Predicted Credit Limit ($)', fontweight='bold')
axes[1].set_title(f'Credit Limit Prediction (R² = {r2:.4f})', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/credit_risk_scoring_results.png', dpi=150, bbox_inches='tight')
plt.show()


# In[25]:


import pandas as pd
import matplotlib.pyplot as plt

# ناساندنی ناوی تایبەتمەندییەکان وەک ئەوەی لە داتاسێتە نوێیەکەدا هەیە
feature_names = ['Shop_Age_Years', 'Total_Invoices', 'Average_Invoice_Value', 
                 'Unpaid_Invoices_Count', 'Current_Debt', 'Late_Payment_History']

# وەرگرتنی ڕێژەی گرنگی فیچەرەکان لە هەردوو مۆدێلەکەدا
risk_importance = xgb_clf.feature_importances_
limit_importance = xgb_reg.feature_importances_

# دروستکردنی دەیتافرەیم بۆ ڕێکخستنیان بەپێی گرنگی
df_risk_imp = pd.DataFrame({'Feature': feature_names, 'Importance': risk_importance}).sort_values(by='Importance', ascending=True)
df_limit_imp = pd.DataFrame({'Feature': feature_names + ['High_Risk'], 'Importance': limit_importance}).sort_values(by='Importance', ascending=True)

# کێشانی هێڵکارییەکان
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Feature Importance Analysis - Erbil Warehouse', fontsize=14, fontweight='bold')

# هێڵکاری مۆدێلی مەترسی (Risk Model)
axes[0].barh(df_risk_imp['Feature'], df_risk_imp['Importance'], color='#e74c3c', edgecolor='black', linewidth=0.7)
axes[0].set_title('Risk Scoring (High/Low Risk)', fontweight='bold')
axes[0].set_xlabel('Relative Importance')
axes[0].grid(axis='x', linestyle='--', alpha=0.6)

# هێڵکاری مۆدێلی لیمیت (Credit Limit Model)
axes[1].barh(df_limit_imp['Feature'], df_limit_imp['Importance'], color='#3498db', edgecolor='black', linewidth=0.7)
axes[1].set_title('Credit Limit Prediction', fontweight='bold')
axes[1].set_xlabel('Relative Importance')
axes[1].grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()


# ## Export Results

# In[26]:


results_export = pd.DataFrame({
    'Actual_Risk': y_risk_test,
    'Predicted_Risk': y_risk_pred,
    'Risk_Probability': y_risk_proba,
    'Actual_Credit_Limit': y_credit_test,
    'Predicted_Credit_Limit': y_credit_pred,
    'Prediction_Error': np.abs(y_credit_test - y_credit_pred)
})
results_export.to_csv('outputs/credit_risk_predictions.csv', index=False)

print('\nExport Complete:')
print('  • credit_risk_dataset.csv')
print('  • credit_risk_predictions.csv')
print('  • credit_risk_scoring_results.png')


# In[27]:


import joblib

# پاشەکەوتکردنی مۆدێلەکە و پاککەرەوەی داتاکە (Scaler)
# تێبینی: پێویستە ناوی گۆڕاوەکان ڕێک ئەوانە بن کە لە کۆدەکەتدا بەکارت هێناون
joblib.dump(xgb_clf, 'outputs/risk_model.joblib')
joblib.dump(xgb_reg, 'outputs/limit_model.joblib')
joblib.dump(scaler_clf, 'outputs/scaler_clf.joblib')
joblib.dump(scaler_reg, 'outputs/scaler_reg.joblib')

print("Models successfully saved to 'outputs' folder!")


# ## Hyperparameter Tuning & Cross-Validation
# 
# Here we apply `RandomizedSearchCV` on the XGBoost Regression model to find optimal parameters and ensure the $R^2$ is stable across 5 cross-validation folds.

# In[28]:


from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Param grid for XGBRegressor
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [500, 1000, 1500],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'subsample': [0.7, 0.8, 0.9, 1.0]
}

# Base Regressor
xgb_reg_base = XGBRegressor(
    min_child_weight=1,
    gamma=0.1,
    lambda_=3.0,
    alpha=1.0,
    random_state=42,
    verbosity=0
)

# 5-Fold Randomized Search (n_iter=10 for speed)
random_search = RandomizedSearchCV(
    estimator=xgb_reg_base,
    param_distributions=param_grid,
    n_iter=10,
    scoring='r2',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("Starting Hyperparameter Tuning & Cross-Validation...")
random_search.fit(X_train_reg_scaled, y_credit_train)

print("\n--- BEST PARAMETERS FOUND ---")
print(random_search.best_params_)

print(f"\nBest Cross-Validation R² Score: {random_search.best_score_:.4f}")

# Evaluate best model on Test Set
best_xgb_reg = random_search.best_estimator_
y_credit_pred_tuned = best_xgb_reg.predict(X_test_reg_scaled)
r2_tuned = r2_score(y_credit_test, y_credit_pred_tuned)
print(f"Test Set R² Score (Tuned Model): {r2_tuned:.4f}")

# Overwrite the improved model with the tuned one
import joblib
joblib.dump(best_xgb_reg, 'outputs/limit_model_tuned.joblib')
print("✅ Tuned model saved to 'outputs/limit_model_tuned.joblib'")


# In[29]:


import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# دڵنیابوونەوە لە بوونی فۆڵدەری outputs
os.makedirs("outputs", exist_ok=True)

# 1. پێشبینییەکانی مۆدێلی پۆلێنکردنی مەترسی (Classification)
# تێبینی: لێرەدا وا دامان ناوە ناوی مۆدێلەکەت best_xgb_clf ە
y_clf_train_pred = xgb_clf.predict(X_train_scaled)
y_clf_test_pred = xgb_clf.predict(X_test_scaled)
y_clf_train_prob = xgb_clf.predict_proba(X_train_scaled)[:, 1]
y_clf_test_prob = xgb_clf.predict_proba(X_test_scaled)[:, 1]

CLF_TRAIN = {
    "accuracy": float(accuracy_score(y_risk_train, y_clf_train_pred)),
    "precision": float(precision_score(y_risk_train, y_clf_train_pred, zero_division=0)),
    "recall": float(recall_score(y_risk_train, y_clf_train_pred, zero_division=0)),
    "f1": float(f1_score(y_risk_train, y_clf_train_pred, zero_division=0)),
    "auc_roc": float(roc_auc_score(y_risk_train, y_clf_train_prob))
}

CLF = {
    "accuracy": float(accuracy_score(y_risk_test, y_clf_test_pred)),
    "precision": float(precision_score(y_risk_test, y_clf_test_pred, zero_division=0)),
    "recall": float(recall_score(y_risk_test, y_clf_test_pred, zero_division=0)),
    "f1": float(f1_score(y_risk_test, y_clf_test_pred, zero_division=0)),
    "auc_roc": float(roc_auc_score(y_risk_test, y_clf_test_prob))
}

# 2. پێشبینییەکانی مۆدێلی بڕی قەرز (Regression)
y_reg_train_pred = xgb_reg.predict(X_train_reg_scaled)
y_reg_test_pred = xgb_reg.predict(X_test_reg_scaled)

REG_TRAIN = {
    "mse": float(mean_squared_error(y_credit_train, y_reg_train_pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_credit_train, y_reg_train_pred))),
    "mae": float(mean_absolute_error(y_credit_train, y_reg_train_pred)),
    "r2": float(r2_score(y_credit_train, y_reg_train_pred))
}

REG = {
    "mse": float(mean_squared_error(y_credit_test, y_reg_test_pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_credit_test, y_reg_test_pred))),
    "mae": float(mean_absolute_error(y_credit_test, y_reg_test_pred)),
    "r2": float(r2_score(y_credit_test, y_reg_test_pred))
}

# 3. گرنگی تایبەتمەندییەکان (Feature Importance)
feat_importances = xgb_reg.feature_importances_.tolist()
FEAT_NAMES = ['Shop Age', 'Total Invoices', 'Average Invoice', 'Unpaid Invoices', 'Current Debt', 'Late Payments']
if len(feat_importances) < len(FEAT_NAMES):
    FEAT_NAMES = FEAT_NAMES[:len(feat_importances)]

# کۆکردنەوەی هەموو داتاکان
metrics_data = {
    "CLF_TRAIN": CLF_TRAIN,
    "CLF": CLF,
    "REG_TRAIN": REG_TRAIN,
    "REG": REG,
    "FEAT_NAMES": FEAT_NAMES,
    "FEAT_IMP": feat_importances
}

# پاشەکەوتکردن
with open("outputs/model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=4)

print("✅ داتای مۆدێلە نوێیەکە بەسەرکەوتوویی پاشەکەوت کرا.")


# In[ ]:


import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# دڵنیابوونەوە لە بوونی فۆڵدەری outputs
os.makedirs("outputs", exist_ok=True)

# ==============================================================================
# تێبینی: تکایە دڵنیابە لەوەی ناوی مۆدێلەکان و داتاکان (X_train, y_train...) 
# هەمان ئەو ناوانەن کە لە نۆتبووکەکەتدا بەکارت هێناون، ئەگەر جیاواز بوون تەنها بیانگۆڕە.
# ==============================================================================

# 1. پێشبینییەکانی مۆدێلی پۆلێنکردنی مەترسی (Risk Classification)
# لێرەدا وا دامان ناوە ناوی مۆدێلەکەت xgb_clf ە
clf_model = xgb_clf  # ئەگەر ناوی مۆدێلەکەت جیاوازە، لێرە بیپێچەرەوە

y_clf_train_pred = clf_model.predict(X_train_scaled)
y_clf_test_pred = clf_model.predict(X_test_scaled)
y_clf_train_prob = clf_model.predict_proba(X_train_scaled)[:, 1]
y_clf_test_prob = clf_model.predict_proba(X_test_scaled)[:, 1]

CLF_TRAIN = {
    "accuracy": float(accuracy_score(y_risk_train, y_clf_train_pred)),
    "precision": float(precision_score(y_risk_train, y_clf_train_pred, zero_division=0)),
    "recall": float(recall_score(y_risk_train, y_clf_train_pred, zero_division=0)),
    "f1": float(f1_score(y_risk_train, y_clf_train_pred, zero_division=0)),
    "auc_roc": float(roc_auc_score(y_risk_train, y_clf_train_prob))
}

CLF = {
    "accuracy": float(accuracy_score(y_risk_test, y_clf_test_pred)),
    "precision": float(precision_score(y_risk_test, y_clf_test_pred, zero_division=0)),
    "recall": float(recall_score(y_risk_test, y_clf_test_pred, zero_division=0)),
    "f1": float(f1_score(y_risk_test, y_clf_test_pred, zero_division=0)),
    "auc_roc": float(roc_auc_score(y_risk_test, y_clf_test_prob))
}

# 2. پێشبینییەکانی مۆدێلی بڕی قەرز (Credit Limit Regression)
# لە نۆتبووکەکەتدا بینیم ناوی مۆدێلەکە best_xgb_reg ە 
reg_model = best_xgb_reg 

y_reg_train_pred = reg_model.predict(X_train_reg_scaled)
y_reg_test_pred = reg_model.predict(X_test_reg_scaled)

REG_TRAIN = {
    "mse": float(mean_squared_error(y_credit_train, y_reg_train_pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_credit_train, y_reg_train_pred))),
    "mae": float(mean_absolute_error(y_credit_train, y_reg_train_pred)),
    "r2": float(r2_score(y_credit_train, y_reg_train_pred))
}

REG = {
    "mse": float(mean_squared_error(y_credit_test, y_reg_test_pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_credit_test, y_reg_test_pred))),
    "mae": float(mean_absolute_error(y_credit_test, y_reg_test_pred)),
    "r2": float(r2_score(y_credit_test, y_reg_test_pred))
}

# 3. گرنگی تایبەتمەندییەکان (Feature Importance)
# وەرگرتنی ڕێژەی گرنگی لە مۆدێلەکەوە و گۆڕینی بۆ لیستی ئاسایی بۆ ئەوەی لە JSON بخوێنرێتەوە
feat_importances = reg_model.feature_importances_.tolist()

# ناوی فیچەرەکان وەک وێبسایتەکە
FEAT_NAMES = ['Shop Age', 'Total Invoices', 'Average Invoice', 'Unpaid Invoices', 'Current Debt', 'Late Payments']

# گونجاندنی ناوەکان لەگەڵ ژمارەکاندا لە ئەگەری کەمتربوونی فیچەرەکان
if len(feat_importances) < len(FEAT_NAMES):
    FEAT_NAMES = FEAT_NAMES[:len(feat_importances)]

# کۆکردنەوەی هەموو داتاکان لە یەک شوێن
metrics_data = {
    "CLF_TRAIN": CLF_TRAIN,
    "CLF": CLF,
    "REG_TRAIN": REG_TRAIN,
    "REG": REG,
    "FEAT_NAMES": FEAT_NAMES,
    "FEAT_IMP": feat_importances
}

# پاشەکەوتکردنی داتاکان بۆ ناو فایلی JSON
with open("outputs/model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=4)

print("✅ Model metrics successfully computed and saved to 'outputs/model_metrics.json'")
print("🔗 ئێستا وێبسایتەکە بەشێوەیەکی ئۆتۆماتیکی داتا نوێیەکان دەخوێنێتەوە!")

