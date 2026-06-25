# 📦 Intelligent Credit Risk & Limit Scoring System

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Production_Ready-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Dual_Task_Pipeline-purple.svg)

## 📌 Project Overview
Currently, many B2B warehouses in the Erbil market rely on subjective, manual assessments to determine credit limits for retail shops. This often leads to accumulated debt, delayed payments, and capital loss. 

This project solves this problem by introducing an **Intelligent, Data-Driven Machine Learning System** powered by **XGBoost**. The system analyzes historical B2B transaction data using **Advanced RFM (Recency, Frequency, Monetary)** metrics to automatically classify customer risk (High/Low) and predict a safe, personalized credit limit in USD.

## 🚀 Key Features
* **Advanced RFM Modeling:** Extracts 8 critical behavioral and financial features from raw transaction data to build a comprehensive customer profile.
* **Class Imbalance Handling:** Utilizes **SMOTE** to balance the dataset (scaling up to 1500+ realistic corpus entries), ensuring an unbiased and highly accurate model.
* **Dual-Task ML Pipeline:**
  * **Classification (`XGBClassifier`):** Predicts the probability of a customer being "High Risk" or "Low Risk" (Accuracy: ~87%+).
  * **Regression (`XGBRegressor`):** Calculates the exact, safe credit limit in USD based on the risk profile (R² Score: ~90%+).
* **Dual Deployment Strategy:** * **Analytical Dashboard:** A rich, interactive UI built with **Streamlit** (Custom Dark Liquid Glass CSS) for data visualization and prototyping.
  * **Production API (SPA):** A lightweight, high-performance RESTful API using **FastAPI** + **SQLite**, paired with a Vanilla JS/Tailwind frontend to eliminate server reruns and ensure seamless real-world integration.

## 📊 Dataset & Features
The model evaluates the following 8 core features to determine creditworthiness:
1. `Shop_Age_Years`: Business longevity and trust.
2. `Days_Since_Last_Order`: **Recency** metric.
3. `Order_Freq_Per_Month`: **Frequency** metric.
4. `Average_Invoice_Value`: Average transaction size.
5. `Total_Trade_Volume`: **Monetary** metric (Lifetime value).
6. `Unpaid_Invoice_Ratio`: Percentage of unpaid invoices.
7. `Debt_To_Volume_Ratio`: Current debt relative to total transaction volume.
8. `Late_Payment_History`: Count of historical late payments.

## 📂 Project Structure
```text
├── Credit_Limit_Risk_Scoring.ipynb  # Data generation, SMOTE, and XGBoost training pipeline
├── app.py                           # Streamlit Web Application (Rich Dashboard)
├── main.py                          # FastAPI Backend Server (Production API)
├── index.html                       # Frontend SPA (Vanilla JS + Tailwind CSS)
├── requirements.txt                 # Python dependencies
└── outputs/                         # Serialized models (.joblib), scalers, and metric JSONs
