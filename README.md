# 📰 Fake News Detection using Machine Learning

A Machine Learning-based web application that detects whether a news article or headline is **Real** or **Fake** using **Logistic Regression** and **TF-IDF Vectorization**. The project includes a user-friendly **Streamlit** interface where users can paste news content and receive predictions with confidence scores and text statistics.

## 🚀 Features

* Detects Fake and Real news articles
* Text preprocessing (URL removal, punctuation removal, stopword removal)
* TF-IDF Vectorization (Unigrams & Bigrams)
* Logistic Regression classifier
* Prediction confidence scores
* Interactive Streamlit web interface
* Sample news examples for testing
* Text statistics (word count, punctuation count, CAPS ratio)

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* Joblib
* Regular Expressions (Regex)

## 📊 Machine Learning Pipeline

1. Load ISOT Fake News Dataset
2. Clean and preprocess text
3. Convert text into TF-IDF features
4. Train Logistic Regression model
5. Evaluate model performance
6. Save trained model using Joblib
7. Deploy using Streamlit

## 📂 Dataset

* **ISOT Fake News Dataset**
* Contains approximately **45,000** real and fake news articles used for training and evaluation.

## 📈 Model Performance

* Algorithm: Logistic Regression
* Feature Extraction: TF-IDF (1–2 grams, 10,000 features)
* Accuracy: **~98%** (dataset dependent)

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📸 Application Features

* News prediction with confidence percentage
* Real vs Fake probability visualization
* Interactive dashboard
* Clean and responsive UI

## 🎯 Project Objective

The goal of this project is to help users identify misleading news by applying Natural Language Processing (NLP) and Machine Learning techniques, providing a simple and fast fake news detection system.

