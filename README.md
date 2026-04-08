# Multi Expert Integrated Algorithm for Kidney Biopsy Triage (MEIA-KBT)

This repository contains the core computational framework and implementation of the MEIA-KBT algorithm.

## Description
The goal of this study is to provide a decision-support tool for kidney biopsy triage using a multi-expert integrated approach based on XGBoost and SHAP analysis. 

Due to strict institutional data security policies and IRB (Institutional Review Board) restrictions regarding patient privacy, the clinical datasets from Severance Hospital and Yongin Severance Hospital are not publicly accessible.

To support the scientific reproducibility of our findings, we provide this structural implementation that demonstrates the data preprocessing, model training, and interpretability pipeline.

## Repository Structure
- `main_pipeline.py`: The core script containing the model architecture, feature processing, and SHAP analysis logic.

## How to Run
1. Install dependencies:
   ```bash
   pip install xgboost shap scikit-learn pandas numpy
2. Run the main script:
   ```bash
   python main_pipeline.py
