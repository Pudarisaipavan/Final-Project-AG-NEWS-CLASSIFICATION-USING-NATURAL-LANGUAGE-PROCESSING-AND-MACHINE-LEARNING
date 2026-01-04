# AG News Classification using NLP + Machine Learning

This project builds an end-to-end text classification pipeline to classify news articles into **four categories** using **NLP preprocessing**, **TF-IDF feature extraction**, and **machine learning models**.

**Classes:** World, Sports, Business, Science/Technology

---

## Project Overview

The workflow includes:
- Loading AG News dataset
- Cleaning and preparing text (Title + Description)
- Exploratory Data Analysis (EDA)
- Feature extraction using **TF-IDF (max_features = 5000)**
- Training and evaluating:
  - Random Forest
  - K-Nearest Neighbours (KNN)
  - Multinomial Naïve Bayes
- Comparing models using accuracy and classification metrics
- Visualising results using confusion matrices

---

## Dataset

- **AG News Classification Dataset (Kaggle):**  
  https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

Dataset fields:
- `Class Index`
- `Title`
- `Description`

---

## Methods

### Text Preparation
- Cleaned Title and Description (removed special characters, extra spaces, escape characters)
- Combined text:
  - `text = Title + " " + Description`

### Feature Engineering
- **TF-IDF Vectorisation**
  - `max_features = 5000`

### Models and Hyperparameters
- **Random Forest**
  - `n_estimators = 200`
  - `max_depth = 20`
  - `random_state = 42`

- **KNN**
  - `n_neighbors = 5`
  - `metric = 'euclidean'`
  - `weights = 'distance'`

- **Multinomial Naïve Bayes**
  - Default parameters

---

## Results Summary (from report/notebook)

| Model | Accuracy |
|------|----------|
| Random Forest | ~0.80 |
| KNN | ~0.89 |
| Multinomial Naïve Bayes | ~0.893 (Best) |

**Key observation:** Naïve Bayes performed best and remained computationally efficient, which is common for sparse TF-IDF text features.


