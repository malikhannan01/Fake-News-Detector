# Fake News Detection using Machine Learning

## Overview
This project is a Fake News Detection system built using Machine Learning and Natural Language Processing (NLP). It classifies news as **Real** or **Fake** based on textual content. The system is trained on the **WELFake dataset (Kaggle)** and uses multiple ML models for prediction.

---

## Dataset
- Source: Kaggle (WELFake Dataset)
- Contains labeled real and fake news articles
- Used for training and evaluation

---

## Models Used
This project uses multiple machine learning models:

- Logistic Regression  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Passive Aggressive Classifier  

### Final Prediction Strategy
All models predict the input independently, and the final decision is made using a **majority voting approach**, where the most common prediction is selected as the final output.

---

## Data Preprocessing
The following preprocessing steps were applied:

- Text cleaning (removing special characters, numbers, etc.)
- Lowercasing text
- Tokenization
- Stopwords removal
- Empty URLs

---

## Feature Engineering
- TF-IDF (Term Frequency–Inverse Document Frequency) vectorization was used to convert text into numerical features.

---

## Feature Scaling
- Applied where necessary depending on model requirements to normalize feature values.

---

## Validation Technique
- Cross Validation was used to evaluate model performance and reduce overfitting.

---

## Workflow
1. Data collection (WELFake dataset)
2. Data preprocessing
3. Feature extraction (TF-IDF)
4. Model training (multiple ML models)
5. Cross-validation
6. Majority voting for final prediction
7. Output: Real or Fake news

---

## Model Files
- fake_news_model.pkl  
- tfidf_vectorizer.pkl  

---

## How It Works
User inputs a news text →  
All trained models predict independently →  
Majority voting decides final output →  
Result is shown as **REAL or FAKE**

---

## Limitations
- Model is not fully generalizable to real-world live news articles  
- Performance depends heavily on dataset pattern  
- Cannot fully understand evolving news writing styles  
- May fail on unseen topics or highly modern/news-specific language  

---


## Future Improvements
- Train on larger and more diverse real-world datasets  
- Use deep learning models (LSTM, BERT)  
- Improve generalization using transfer learning  
- Add real-time news scraping system  
- Deploy as web app (Streamlit / Flask)  
- Improve ensemble learning strategy

  ---

  ## How to Use

### Load saved model and vectorizer
import pickle

model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

### Take user input
news = input("Enter news text: ")

### Transform input text
input_data = vectorizer.transform([news])

### Make prediction
prediction = model.predict(input_data)

### Show result
if prediction[0] == 1:
    print("Real News")
else:
    print("Fake News")
   
---

## Conclusion
This project demonstrates a multi-model machine learning approach for fake news detection using NLP. It provides a strong baseline system, but requires further improvements for real-world deployment.
