# Multi Expert Integrated Algorithm for Kidney Biopsy Triage (MEIA-KBT)

This repository provides the official structural implementation for the following manuscript:

- Title: Multi Expert Integrated Algorithm for Kidney Biopsy Triage based on XGBoost and SHAP analysis
- Journal: npj Digital Medicine
- Status: Under Second Revision
- Authors: Hae-Ryong Yun†, Nak-Hoon Son†, Gyubok Lee, Hyung Woo Kim, Tae Ik Chang, Jung Tak Park, Seung Hyeok Han, Shin-Wook Kang, and Tae-Hyun Yoo  
  († These authors contributed equally to this work)

Note: This computational framework was entirely developed and implemented by the Biostatistics and Health Data Analytics Laboratory (biostatnhlab), a specialized research group focused on Biostatistics-based Medical Artificial Intelligence..

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
