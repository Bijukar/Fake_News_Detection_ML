# Fake_News_Detection_ML

## 📌 Abstract

The rise of misinformation has made fake news detection a vital task. This project focuses on building a machine learning pipeline that detects whether a news article is **REAL** or **FAKE** using natural language processing and various ML models.

---

## 📁 Dataset

- WELFAKE_Dataset downloaded from kaggle dataset.The dataset have two label.

- `0` → FAKE
- `1` → REAL

---

## 🔧 Project Pipeline

### 1. Import Libraries
Load essential libraries for preprocessing, visualization, modeling, and evaluation.

### 2. Load Dataset
Read the CSV files and see the dataset.

### 3. Data Cleaning
- Remove missing values.
- Drop unrelated columns.

### 4. Exploratory Data Analysis (EDA)
- **Pie Chart** showing the distribution of real vs. fake news.
- **Histogram** showing word count distribution by label.
- **Word Clouds** for common words in fake and real news.

### 5. Text Preprocessing
- Convert text to lowercase.
- Remove URLs, numbers, and punctuation.
- Tokenization and lemmatization

### 6. Feature Engineering
- Use **TF-IDF Vectorizer** to convert text into numerical features.

### 7. Train-Test Split
Split the data 80% train, 20% test to evaluate model generalization.

### 8. Model Building
Train multiple machine learning models:
- Logistic Regression
- Passive Aggressive Classifier
- Random Forest Classifier
- Multinomial Naive Bayes
  
Comparing all model by **Accuracy** and **F1-score** and choose the best model for further train and evaluation. 

### 9. Model Evaluation
Use:
- Classification Report
- Confusion Matrix
- ROC Curve and AUC Score

### 10. Save Model
Save the best-performing model in .pkl file  as `news_model.pkl`  using `joblib` for future use.

### 11. Predict on New Input
- Preprocess user input text.
- Transform using TF-IDF.
- Use the saved model to classify it as FAKE or REAL.

---

## 📊 Model Performance Comparison

| Model                     |  Accuracy   |  F1 Score |
|---------------------------|-------------|-----------|
| Passive Aggressive Class. | 0.965       | 0.967     |
| Logistic Regression       | 0.947       | 0.949     |
| Random Forest Classifier  | 0.931       | 0.933     |
| Multinomial Naive Bayes   | 0.871       | 0.870     |

> ✅ **Passive Aggressive Classifier** performed the best and was selected as the final model.

---

## 💾 Project Files

- `WELFAKE.Dataset.csv` — Raw datasets
- `fake_news_classification.ipynb` — Full code with visualizations and model training
- `news_model.pkl` — Saved trained model
- `README.md` — Project documentation
- `requirements.txt` - required libraries

---

## 🧠 Tech Stack

- Python 🐍
- Scikit-learn
- Pandas
- Matplotlib / Seaborn
- Nltk
- string
- re

---

## 📌 Keywords

`Fake News Detection` • `Machine Learning` • `Text Classification` • `Natural Language Processing` • `Logistic Regression` • `TF-IDF`
