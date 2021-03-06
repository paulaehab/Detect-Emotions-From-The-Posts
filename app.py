import streamlit as st
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib as joblib
model = joblib.load("finalized_model.sav")



st.title('Emotional Analysis For Text')
text_uploader = st.text_input("Write text for emotional analysis")
tf1 = pickle.load(open("tfidf.pkl", 'rb'))
vec = TfidfVectorizer(binary=True, use_idf=True, ngram_range=(1, 2),vocabulary=tf1.vocabulary_)


if text_uploader is not None:
    pred_button = st.button("Predict")

    if pred_button:
        #Lines = cv.transform(Lines)
        Lines = vec.fit_transform([text_uploader])
        scores = model.predict(Lines)  
        print(scores)
        score = scores[0]
        if score == 0:
            result = 'The emotion of this post is sadness'
        elif score == 1:
            result = 'The emotion of this post is happiness'
        elif score == 2:
            result = 'The emotion of this post is love'
        elif score == 3 :
            result = 'The emotion of this post is anger'
        elif score == 6 :
            result = 'The emotion of this post is neutral'
        st.write(result)
    

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 import streamlit as st
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib as joblib
model = joblib.load("finalized_model.sav")



st.title('Emotional Analysis For Text')
text_uploader = st.text_input("Write text for emotional analysis")
tf1 = pickle.load(open("tfidf.pkl", 'rb'))
vec = TfidfVectorizer(binary=True, use_idf=True, vocabulary=tf1.vocabulary_)


if text_uploader is not None:
    pred_button = st.button("Predict")

    if pred_button:
        #Lines = cv.transform(Lines)
        Lines = vec.fit_transform([text_uploader])
        scores = model.predict(Lines)  
        print(scores)
        score = scores[0]
        if score == 0:
            result = 'The emotion of this post is sadness'
        elif score == 1:
            result = 'The emotion of this post is happiness'
        elif score == 2:
            result = 'The emotion of this post is love'
        elif score == 3 :
            result = 'The emotion of this post is anger'
        elif score == 6 :
            result = 'The emotion of this post is neutral'
        st.write(result)
    

