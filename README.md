# Fake-News-Detection

A machine learning web application that predicts whether a news article is **Real** or **Fake** using Natural Language Processing (NLP).

## Dataset Information
Dataset: WELFake dataset (Fake News Classification)
The model is trained on the WELFake Dataset, a comprehensive set of 72,134 news articles (35,028 Real and 37,106 Fake)
Link: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

## Performance Comparison
To ensure the best results, we compared Four different machine learning models:
Random Forest Classifier, Logistic Regression, Multinomial Naive Bayes, Support Vector Machine (SVM)

## Features
* **Machine Learning Model:** Random Forest Classifier (95.8% Accuracy).
* **NLP Pipeline:** Text cleaning, stopword removal, and stemming.
* **Web Interface:** Interactive UI built with Flask, HTML, and CSS.

## Tech Stack
* **Language:** Python
* **Libraries:** Scikit-Learn, Pandas, NLTK, Flask
* **Frontend:** HTML5, CSS3

## Project Structure
* `app.py`: Flask backend logic and prediction engine.
* `Model_Training.ipynb`: Model building and evaluation.
* `fake_news_detection_data_cleaning.ipynb`: Data preprocessing.
* `static/style.css`: Visual styling of the app.
* `templates/index.html`: Web interface.

