# Detect Emotions From Posts

#### This is a machine learning algorithum to detect emotions (Angry - Happy - Sad - Neutral - Love) from humans' posts.

## How to run the code.
you have to install python > 3.7.0, you can download python for windows from here: https://www.python.org/downloads/windows/

### Create Virtual Evironment
```
pip install --upgrade virtualenv
virtualenv  envname
cd envname/
cd Scripts/
activate
cd..
cd..

pip install -r requirements.txt
```
### Run the streamlit app on localhost 
```
streamlit run app.py
```
### Run the trainning script  
```
python train.py
```
### Run the prediction script
```
python predict.py --csv path/to/file.csv
```
```
python predict.py --text any_text
```
## About Dataset.
### 1- Data collection
First I used dataset about English Twitter posts with six basic emotions: anger, fear, joy, love, sadness, and surprise.  
The authors constructed a set of hashtags to collect a separate dataset of English tweets from the Twitter API belonging to six basic emotions.  
Dataset link: https://www.kaggle.com/parulpandey/emotion-dataset  
Second dataset is a collection of tweets annotated with the emotions behind them with 13 emotions  
Dataset link: https://www.kaggle.com/pashupatigupta/emotion-detection-from-text  
I used a merged version between the two datasets to provide a data set with the following emotion (Angry - Happy - Sad - Neutral - Love) 
### 2- Data Preprocessing and Data cleaning
#### Data Preprocessing
I merged the two datasets to get a full dataset with the four emotions and you can find the code of merging in 
 ``` Final model/create new data set .ipynb``` and result is saved to ```Final model/ merged data```   
The result dataset in conatin :  
1- train.csv consists of: 16200 records  
2- test.csv consists of: 2200 records  
3- val.cs conists of: 2200 records  
The dataset labels is as the following {0:'sad', 1:'happy', 2:'love', 3:'anger', 4:'anger', 5:'surprise', 6:'neutral')
 #### Data Cleaning
 1-Remove any HTML tags in the text by using function ```clean_html``` in the code  
 2-Convert all text to lowercase using function ```convert_lower``` in the code  
 3- Remove any tags in posts like @name using function ```cleaning_tags``` in the code  
 4-Remove punctuations in posts using function ```cleaning_punctuations``` in the code  
 5-Remove any URLS in text using function ```cleaning_URLs``` in the code  
 6- Remove any number in text using function ```cleaning_numbers``` in the code     
 7- Remove any stop words in text like (the - this - any -etc.)  
 8- Stem each word in the text which mean reducing a word to its word stem that affixes to suffixes and prefixes
 ## Approaching models  
 ### First classical machine learning  model  
 I made a mchine learning model on the first dataset to detect 6 emotions (Sad - Happy - Love - Anger - Fear - Surprise)  
 I used the first dataset and train a Lazzycalssifier on cloud to test 27 mahcine learning  models for me,  
 and it give me that the following three models is the best ones:  
 1- PassiveAggressiveClassifier which gave me accuracy = 87.29%  
 2- LogisticRegression which gave me accuracy = 85.5%  
 3-XGBClassifier which gave me accuracy = 87.56%  
 In this solution the XGBClassifier seem to be the best model for the problem 
 you can find the model in the following folder: ```/Fisrt model with first dataset/first classical model .ipynb```  
 ### Second classical machine learning  model
 The problem with the first model was it detect 5 emotions but now Neutral emotion between them so after mergeing two datasets,  
 I get a data wit 7 emotions but as tha result of mergeing is Unbalanced data I had to drop out the (Fear - Surprise ) emotions,  
 so this model will predict only the following emotions ( Sad - Happy - Love - Anger - Neutral )  
 also I run lazzy classfier and give me the same 3 model which are the following:  
 1- PassiveAggressiveClassifier which gave me accuracy =  88.68%%  
 2- LogisticRegression which gave me accuracy =  88.41%  
 3-XGBClassifier which gave me accuracy = 89.78%  
 as we can see the accuracy is increased a little and the XGBClassifier is doing well with this problem.  
 you can find the code in the following folder: ```Final models/Best Classical model.ipynb```  
 Also I saved the wieghts of the model for future use you can find it in the following folder ```Final models/finalized_model.sav```  
 ### Deep learning approach 
 I tried to build a deep learning model hopping to increase accuracy of the model  
 I used keras library to build my deep learning archticture and searched for a good archticture and found a one  
 suitable for this problem and use it but it give me accuracy about 88.60% only and this was disappointing for me  
 you can find the model in the following folder ```Final models/deep learnig model.ipynb```  
 
  ## Conclusion 
  the second classical machine learning which is working on XGBClassifier algorithum  was the best model between all models which gave me the best accuracy which is : 89.78%  
  it approximately to 90% and it work very good to detect the following emotions ( Sad - Happy - Love - Anger - Neutral )  
  so I used this model for deployment and for final results  
   ## Deployment
   
  ## References 
  Datasets References :  
  1- https://www.kaggle.com/parulpandey/emotion-dataset  
  2-https://www.kaggle.com/pashupatigupta/emotion-detection-from-text  
  Article that help me to clean the data :  
  https://www.analyticsvidhya.com/blog/2021/06/twitter-sentiment-analysis-a-nlp-use-case-for-beginners/  
  Deep learning model that I have used:  
  https://www.kaggle.com/muratkarakurt/emotion-detect-comment-97

