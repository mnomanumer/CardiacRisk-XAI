# Real-Time Cardiovascular Risk Assessment & Explainable AI (XAI) Dashboard

A high-intermediate health informatics dashboard combining robust clinical risk prediction with transparent, human-interpretable feature explanations.

## The Core Uniqueness
Addresses the medical "black box" dilemma by injecting an Explainable AI (XAI) evaluation layer. Instead of generating an uninterpretable probability index, the application uses Game Theory weights to break down the exact risk attribution vector for individual patient features in real time.

## Tech Stack & Requirements
* Machine Learning: LightGBM
* Explainable AI: SHAP (SHapley Additive exPlanations), LIME
* Data Pipeline: IterativeImputer (Multivariate missing data recovery)
* User Interface: Streamlit Cloud

## Engineering & Hardware Optimization
Generating SHAP matrix kernel calculations is highly CPU-intensive. To bypass local processing bottlenecks, the entire training notebook runs inside an isolated cloud instance via Google Colab, outputting optimized lightweight model weights to fuel the interactive Streamlit interface.