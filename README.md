
# 🔍 Classification Analysis

## 📌 Project Overview

This project is classification analysis on the Optical Recognition of Handwritten Digits 

dataset. It includes data preprocessing, dimensionality reduction, model training, evaluation, 

and visualization. The notebook covers multiple classification algorithms and provides 

comprehensive evaluation metrics and visualizations.

## Classification Analysis App:
https://pcaclassificationanalysis.streamlit.app/

## 🚀 Features

Data Preprocessing – Standard scaling and PCA for dimensionality reduction

Multiple Classification Algorithms – Support Vector Machine (SVM), Logistic Regression, K-Nearest Neighbors (KNN), Random Forest

Model Evaluation – Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve

Visualization – PCA-based 2D data visualization, confusion matrix heatmap, ROC curves

<img width="666" height="547" alt="image" src="https://github.com/user-attachments/assets/c38bf1c7-03fd-4f59-aeb4-055b5aae45f4" />

<img width="883" height="747" alt="image" src="https://github.com/user-attachments/assets/8f92e808-631f-434c-9f19-ebba5135315e" />


## 🛠️ Tech Stack

Python 🐍

Scikit-learn – Classification models, metrics, and preprocessing

Pandas / NumPy – Data handling

Matplotlib / Seaborn – Visualization

UCI ML Repository – Dataset fetching

## 📂 Dataset
Optical Recognition of Handwritten Digits

Source: UCI Machine Learning Repository (ID: 80)

Features: 64 attributes (8x8 image pixels)

Target: 10 classes (digits 0–9)

## ⚙️ Installation & Setup
Install dependencies:

bash

pip install matplotlib seaborn numpy pandas scikit-learn

Run the Jupyter Notebook:

bash

jupyter notebook classification.ipynb

## 📊 Example Workflow
Load Data – Fetch dataset from UCI repository

Preprocess – Scale features using StandardScaler

Reduce Dimensions – Apply PCA for 2D visualization

Train Models – Fit multiple classifiers

Evaluate – Compute accuracy, precision, recall, F1-score, and plot confusion matrices & ROC curves

Visualize Feature Contributions to PC1 and PC2
<img width="705" height="516" alt="image" src="https://github.com/user-attachments/assets/1a317108-e26d-4cde-a9f8-44c1a56aa27d" />


Visualize – Plot Cummulative explained Variance
<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/8e82dca6-ed73-4407-9db4-a719f1e328c6" />


## 📈 Evaluation Metrics
Accuracy – Overall correctness

Precision – True positives among predicted positives

Recall – True positives among actual positives

F1-Score – Harmonic mean of precision and recall

📸 Example Visualizations
2D PCA scatter plot of digit classes
<img width="671" height="547" alt="image" src="https://github.com/user-attachments/assets/0a246c89-9eeb-4c54-8fd5-146cb687e78c" />
