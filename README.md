# Skin-Detect
Developed different machine learning models to classify skin lesions as benign or malignant using image data.

#Skin Lesion Classification (HAM10000 Project)

Overview
This project builds machine learning models to classify skin lesions as benign or malignant using the HAM10000 dataset. The goal is to compare deep learning (CNN) with classical machine learning methods (kNN and Random Forest) and evaluate their strengths, limitations, and ethical implications in medical image classification.
This project fulfills Part A (Data Cleaning & Preprocessing) and Part B (Machine Learning Analysis) of the CS 330 Intro to Machine Learning class project.
Dataset
We use the HAM10000 dataset from Kaggle.
Each entry includes:
A dermoscopic image (RGB)


Metadata:
 lesion_id, image_id, dx (diagnosis), age, sex, localization


The original 7 classes were mapped to binary labels:
Benign (0): nv, bkl, df, vasc


Malignant (1): mel, bcc, akiec


After filtering missing/unknown metadata and matching resized images, the final dataset contained 9,761 images.

Repository Structure
├── 01_preprocessing_and_cnn.ipynb      # Data cleaning, filtering, resizing, CNN model
├── 02_classical_models_knn_rf.ipynb    # HOG feature extraction, kNN, Random Forest
├── df_final.csv                        # Cleaned dataset with labels + image paths
└── README.md                           # Project overview and instructions


Project Steps
Part A : Data Cleaning & Preprocessing
Performed in 01_preprocessing_and_cnn.ipynb:
Loaded metadata from HAM10000 CSV


Removed entries with:


Unknown sex


Unknown localization


Missing age


Mapped 7-class dx → binary benign/malignant


Resized images to 224×224 using PIL


Normalized pixel values


Generated final dataset (df_final.csv) with:


image paths


metadata


binary labels


Trained a baseline CNN model:


3 Conv2D + MaxPooling blocks


Dense(64) + Dropout


Sigmoid output


Part B : Classical ML Models
Performed in 02_classical_models_knn_rf.ipynb:
Loaded df_final.csv


Loaded all resized images


Extracted HOG features


Output shape: (4954,9)


Standardized features (StandardScaler)


Split dataset (70% train, 30% test)


Models:
kNN (k=5) trained on HOG features


Random Forest (200 trees) trained on HOG features


For each model, we computed:
Accuracy


Precision


Recall


F1-score


Classification report


Model Performance Summary
Model
Accuracy
Malignant Recall
Notes
Baseline CNN
~0.823
~0.003
Best model; can detect malignant tumors reasonably
kNN (HOG)
0.797
0.03
Predicts almost all lesions as benign
Random Forest (HOG)
0.802
0.00
Fails to detect malignant lesions entirely


Key Findings
Deep learning (CNN) significantly outperforms classical models.


kNN and Random Forest achieve decent accuracy only because the dataset is imbalanced as they misclassify nearly all malignant cases.


HOG features are not expressive enough to capture medical lesion patterns.


Class imbalance strongly affects malignant detection.


Ethical Considerations
Machine learning systems misclassifying malignant lesions pose serious risk. Skin cancer detection models require:
High recall on malignant cases


Diverse training datasets


Fairness across skin tones and demographics


Human-in-the-loop evaluation


Strong clinical validation


This project is educational only and must not be used in real medical settings.

Requirements
Install the following packages:
numpy
pandas
opencv-python
scikit-learn
scikit-image
tensorflow
matplotlib
Pillow

Or install via:
pip install -r requirements.txt
How to Run the Notebooks
1. Preprocessing + CNN
01_preprocessing_and_cnn.ipynb

This notebook:
Loads metadata


Filters, cleans, normalizes


Resizes images


Saves df_final.csv


Trains baseline CNN


2. Classical Models
02_classical_models_knn_rf.ipynb

This notebook:
Loads df_final


Loads resized images


Extracts HOG features


Trains kNN & Random Forest


Prints evaluation metrics


Creates comparison table

Authors
Nhu Nguyen
Sanjina Kumari


CS 330: Introduction to Machine Learning
 Pacific Lutheran University
