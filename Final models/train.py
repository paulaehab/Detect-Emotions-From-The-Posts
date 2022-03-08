import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer ,ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
import re 
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pickle
nltk.download('stopwords')



# 1- Make a data set with all emotions
### Merge The two datasets to get neutral emotion 
df1 = pd.read_csv('./First Data/training.csv')
print(df1.shape)
df3 = pd.read_csv('./Third Data/tweet_emotions.csv')
print(df3.shape)
df3 = df3[df3.sentiment == 'neutral'] 

df3.shape
df3=df3[:200]
df3.shape
df3.head()
### Drop the id column
df3.drop(['tweet_id'],axis=1,inplace = True)
df3.head()
### Rename the coulmns names

df3.rename(
    columns={"sentiment":"label",
                "content":"text",
                   }
          ,inplace=True)
df3.head()
### insert value 6 insted of neutral word
df3.label =int(6) 
df3.head()
### Concat the two datasets
df = pd.concat([df1, df3], ignore_index = True, axis = 0)
df.shape
### shuffle the data set 
df=df.sample(frac = 1)
df.head()
#label_dict = {0:'sad', 1:'happy', 2:'love', 3:'angry', 4:'fear', 5:'surprise',6:'neutral'}

### drop the fear and surprise emotions
df = df[df.label != 4] #& 'boredom' & 'enthusiasm' & 'empty'
df = df[df.label != 5]

df = df[df.label != 'sentiment']
df.head()
#label_dict = {0:'sad', 1:'happy', 2:'love', 3:'angry',6:'neutral'}

sns.countplot(df['label'],order = df['label'].value_counts(normalize=True).index)
# Data Preprocessing
# 1- clean aimport pandas as pd 




# 1- Make a data set with all emotions
### Merge The two datasets to get neutral emotion 
df1 = pd.read_csv('./First Data/training.csv')
print(df1.shape)
df3 = pd.read_csv('./Third Data/tweet_emotions.csv')
print(df3.shape)
df3 = df3[df3.sentiment == 'neutral'] 

df3.shape
df3=df3[:200]
df3.shape
df3.head()
### Drop the id column
df3.drop(['tweet_id'],axis=1,inplace = True)
df3.head()
### Rename the coulmns names

df3.rename(
    columns={"sentiment":"label",
                "content":"text",
                   }
          ,inplace=True)
df3.head()
### insert value 6 insted of neutral word
df3.label =int(6) 
df3.head()
### Concat the two datasets
df = pd.concat([df1, df3], ignore_index = True, axis = 0)
df.shape
### shuffle the data set 
df=df.sample(frac = 1)
df.head()
#label_dict = {0:'sad', 1:'happy', 2:'love', 3:'angry', 4:'fear', 5:'surprise',6:'neutral'}

### drop the fear and surprise emotions
df = df[df.label != 4] #& 'boredom' & 'enthusiasm' & 'empty'
df = df[df.label != 5]

df = df[df.label != 'sentiment']
df.head()
#label_dict = {0:'sad', 1:'happy', 2:'love', 3:'angry',6:'neutral'}

sns.countplot(df['label'],order = df['label'].value_counts(normalize=True).index)
# Data Preprocessing
# 1- clean any HTML tags in the text
def clean_html(text):
    
    clean = re.compile('<.*?>')
    return re.sub(clean, '',text)
    
df['text']=df['text'].apply(clean_html)
df.head()
# 2- convert all the text into lower case 
def convert_lower(text):
    return text.lower()

df['text']=df['text'].apply(convert_lower)
df.head()
# 3- clean the Tag sign and Tag name (ex:@Paula)
def cleaning_tags(text):
    return ' '.join(re.sub("([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)"," ", text).split())

df['text'] = df['text'].apply(lambda x: cleaning_tags(x))
df['text'].head()
# 4- clean all the punctuations  

english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
df['text']= df['text'].apply(lambda x: cleaning_punctuations(x))
df['text'].head()
#def cleaning_repeating_char(text):
 #   return re.sub(r'([a-z])\1+', r'\1', text)
#df['text'] = df['text'].apply(lambda x: cleaning_repeating_char(x))
#df['text'].head()
# 5- clean the urls founded in the tweets

def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
df['text'] = df['text'].apply(lambda x: cleaning_URLs(x))
df['text'].head()

# 6- clean all numbers 
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
df['text'] = df['text'].apply(lambda x: cleaning_numbers(x))
df['text'].head()
# 7-remove stopwords from data


stop_words = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
df.head()
# 8- stemming words in data
ps= PorterStemmer()
y=[]

def stem_words(text):
    for i in text:
        y.append(ps.stem(i))
    z=y[:]
    y.clear()
    return z
df['text']=df['text'].apply(stem_words)
df.head()
# 9- join back after stemming
def joinback2(list_input):
    return "".join(list_input)
    


df['text']=df['text'].apply(joinback2)
df.head()


### Exploratory data analysis 
# 1- Sad Emotion
txt = ' '.join(text for text in df[df['label']==0]['text'])

wordcloud = WordCloud(
            background_color = 'white',
            max_font_size = 100,
            max_words = 200,
            width = 800,
            height = 500
            ).generate(txt)


plt.imshow(wordcloud,interpolation = 'bilinear')
plt.axis('off')
plt.show()
# 2- Happy Emotion
txt = ' '.join(text for text in df[df['label']==1]['text'])

wordcloud = WordCloud(
            background_color = 'white',
            max_font_size = 100,
            max_words = 200,
            width = 800,
            height = 500
            ).generate(txt)


plt.imshow(wordcloud,interpolation = 'bilinear')
plt.axis('off')
plt.show()
# 3- Love Emotion
txt = ' '.join(text for text in df[df['label']==2]['text'])

wordcloud = WordCloud(
            background_color = 'white',
            max_font_size = 100,
            max_words = 200,
            width = 800,
            height = 500
            ).generate(txt)


plt.imshow(wordcloud,interpolation = 'bilinear')
plt.axis('off')
plt.show()
# 4-Angry Emotion
txt = ' '.join(text for text in df[df['label']==3]['text'])

wordcloud = WordCloud(
            background_color = 'white',
            max_font_size = 100,
            max_words = 200,
            width = 800,
            height = 500
            ).generate(txt)


plt.imshow(wordcloud,interpolation = 'bilinear')
plt.axis('off')
plt.show()
# 5- Neutral Emotion
txt = ' '.join(text for text in df[df['label']==6]['text'])

wordcloud = WordCloud(
            background_color = 'white',
            max_font_size = 100,
            max_words = 200,
            width = 800,
            height = 500
            ).generate(txt)


plt.imshow(wordcloud,interpolation = 'bilinear')
plt.axis('off')
plt.show()
### split data 
X=df['text']
y=df.label
vec = TfidfVectorizer(binary=True, use_idf=True)

tf_model = vec.fit(X)

pickle.dump(tf_model, open("tfidf1.pkl", "wb"))
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3) 
### Transfer data into numerical features using TFIDF
X_train = vec.fit_transform(X_train) 
X_test = vec.transform(X_test)


X_test.shape
### Try first model passive aggressive classfier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(X_train,y_train)
y_pred=pac.predict(X_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

### Try second model : logistic regression
log_reg = LogisticRegression(max_iter=50).fit(X_train, y_train)

y_predicted = log_reg.predict(X_test)
score=accuracy_score(y_test,y_predicted)
print(f'Accuracy: {round(score*100,2)}%')
### Try third model : xgboost 
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100,learning_rate=0.2)
print("Labels:", y_train.unique())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
### Testing

