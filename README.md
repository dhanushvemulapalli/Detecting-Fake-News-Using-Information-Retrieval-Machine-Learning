# 📰 Fake News Detection using Information Retrieval & Machine Learning

Blog : (https://medium.com/@dhanushvemulapalli717/detecting-fake-news-using-information-retrieval-machine-learning-a-practical-implementation-b529a9670b5a)

This project explores how Information Retrieval (IR) techniques combined with Machine Learning and Deep Learning models can be used to automatically classify news articles as REAL or FAKE. The system applies traditional NLP preprocessing, vectorization methods like TF-IDF and Word2Vec embeddings, and evaluates multiple classifiers — from classical ML models to a Bi-LSTM neural network.

## 🚀 Project Overview

Online misinformation is growing rapidly, and detecting fake news has become a critical challenge. This project demonstrates how the following pipeline helps identify misinformation:

Raw Text → Preprocessing → Feature Extraction → Model Training → Evaluation


The workflow integrates:

Classical IR text processing (tokenization, stemming, stopword removal)
Feature representation (TF-IDF, n-grams, embeddings)
Multiple ML and DL classification models

## 📁 Dataset

Two publicly available datasets were used and later merged:

WELFake Dataset
Kaggle Fake-News Dataset

Labels were standardized to:

Label	Meaning
0	Real News
1	Fake News

🧹 Preprocessing

The text underwent several NLP processing steps:

Lowercasing
Tokenization
Stopword Removal (with negation handling)
Stemming (Porter Stemmer)
Removal of punctuation and non-alphabetic characters

This ensures consistency across articles and improves feature extraction accuracy.

## 🔠 Feature Extraction

Two major feature representation techniques were used:

Method	Description
TF-IDF (~5000 dims, with n-grams)	Sparse feature representation using word frequency statistics
Word2Vec (300-dim pretrained embeddings)	Dense semantic representation capturing word meaning

## 🤖 Models Used

Both machine learning and deep learning models were trained and evaluated:

## 🧩 Classical ML Models (TF-IDF)

Linear SVM
Decision Tree
Random Forest
Gradient Boosting
XGBoost

## 🧠 Embedding-Based ML Models

Word2Vec + SVM
Word2Vec + Ensemble Models

## 🔥 Deep Learning Model

Bi-LSTM with Word2Vec embeddings

## 📊 Results
TF-IDF (Full Dataset)
Model	Accuracy	Precision	Recall	F1 Score
Linear SVM	0.8609	0.8575	0.8735	0.8654
Decision Tree	0.8455	0.8423	0.8588	0.8505
Random Forest	0.8659	0.8560	0.8872	0.8714
Gradient Boosting	0.8525	0.8130	0.9246	0.8652
XGBoost	0.8683	0.8581	0.8899	0.8737
Word Embeddings (10% sample due to compute constraints)
Model	Accuracy	Precision	Recall	F1 Score
Linear SVM	0.7775	0.7751	0.8119	0.7931
Random Forest	0.7487	0.7512	0.7799	0.7653
Bi-LSTM	0.7949	0.7915	0.8274	0.8090
📌 Key Takeaways

TF-IDF + XGBoost performed best among classical models.
Bi-LSTM showed strong potential even with limited data and compute.
TF-IDF relies on word frequency patterns, while embeddings capture semantic meaning.
With full training, transformer-based models (BERT/RoBERTa) would likely outperform both.

## 🚧 Future Improvements

Fine-tune Transformer models (BERT, RoBERTa, DistilBERT)
Add topic modeling (LDA, BERTopic) for explainability
Deploy inference as API or Web App (FastAPI/Streamlit)

## 📚 References

See full reference list in the /docs folder or the Medium article linked below.

## ✨ Author

👤 Dhanush Vemulapalli
📌 Machine Learning & NLP Enthusiast

## 📝 Medium Article
(https://medium.com/@dhanushvemulapalli717/detecting-fake-news-using-information-retrieval-machine-learning-a-practical-implementation-b529a9670b5a)
