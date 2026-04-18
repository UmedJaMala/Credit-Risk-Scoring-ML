import json

with open('Credit_Limit_Risk_Scoring.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Identify "Model Training" cell
        if "xgb_clf = XGBClassifier" in source and "import joblib" in source:
            new_source = """import joblib
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
print("\\n" + "="*45)
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
print("\\n" + "="*45)
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

print("\\n✅ مۆدێلەکان بە سەرکەوتوویی ڕاهێنران و پاشەکەوتکران.")"""
            
            lines = new_source.split('\n')
            cell['source'] = [line + '\n' for line in lines]
            cell['source'][-1] = cell['source'][-1].replace('\n', '')

with open('Credit_Limit_Risk_Scoring.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
