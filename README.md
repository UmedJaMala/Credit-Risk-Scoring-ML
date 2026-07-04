# 📦 Intelligent Credit Risk & Limit Scoring System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Production_Ready-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Spaces-FFD21E.svg)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-Styled-38B2AC.svg)

## 🌐 Live Demo
* **Frontend Web App:** [Live Dashboard](https://umedjamala.github.io/Credit-Risk-Scoring-ML/)
* **Backend API Docs (Swagger UI):** [FastAPI on Hugging Face](https://umedjamala-credit-risk-scoring-ml.hf.space/docs)

## 📌 Project Overview
Currently, many B2B warehouses in the Erbil market rely on subjective, manual assessments to determine credit limits for retail shops. This often leads to accumulated debt, delayed payments, and capital loss. 

This project solves this problem by introducing an **Intelligent, Data-Driven Machine Learning System** powered by **XGBoost**. The system analyzes historical B2B transaction data using **Advanced RFM (Recency, Frequency, Monetary)** metrics to automatically classify customer risk (High/Low) and predict a safe, personalized credit limit in USD.

## 🚀 Key Features
* **Modern Decoupled Architecture (End-to-End):**
  * **Backend:** A lightweight, high-performance RESTful API using **FastAPI**, containerized with **Docker**, and hosted on **Hugging Face Spaces**.
  * **Frontend:** A blazing-fast, Glassmorphism-styled SPA using Vanilla JS & **TailwindCSS**, hosted on **GitHub Pages** to eliminate server reruns and ensure seamless real-world integration.
* **Advanced RFM Modeling:** Extracts 8 critical behavioral and financial features from raw transaction data to build a comprehensive customer profile.
* **Class Imbalance Handling:** Utilizes **SMOTE** to balance the dataset (scaling up to 1500+ realistic corpus entries), ensuring an unbiased model.
* **High-Performance Dual-Task ML Pipeline:**
  * **Classification (`XGBClassifier`):** Predicts the probability of a customer being "High Risk" or "Low Risk" (Accuracy: **~95%**, ROC-AUC: **~99%**).
  * **Regression (`XGBRegressor`):** Calculates the exact, safe credit limit in USD based on the risk profile (R² Score: **~98%**).

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
├── Dockerfile                       # Docker configuration for Hugging Face deployment
├── main.py                          # FastAPI Backend Server (Production API)
├── index.html                       # Frontend SPA (Vanilla JS + Tailwind CSS + Glassmorphism UI)
├── requirements.txt                 # Python dependencies
└── outputs/                         # Serialized models (.joblib), scalers, and metric JSONs
