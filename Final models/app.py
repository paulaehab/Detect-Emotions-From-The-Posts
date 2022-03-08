import streamlit as st
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib as joblib
model = joblib.load("finalized_model.sav")
# model = XGBClassifier()
# model.load_model("finalized_model.sav")


st.title('Emotional Analysis For Text')
text_uploader = st.text_input("Write text for emotional analysis")
tf1 = pickle.load(open("tfidf3.pkl", 'rb'))
print("Text uploaded", tf1)
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
            result = 'The post is sad'
        elif score == 1:
            result = 'The post is happy'
        elif score == 2:
            result = 'The post is Love'
        elif score == 3 :
            result = 'The post is anger'
        elif score == 6 :
            result = 'The post is Neutral'
        st.write(result)
    

