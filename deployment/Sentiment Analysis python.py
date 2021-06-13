# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Sklearn Libraries
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

## Download stopwords

# nltk.download('stopwords')

"""## Data"""
data = pd.read_csv('Copy of Coronavirus Tweets.csv')
data.head()

data.shape

## get the info of datasets
data.info()

data['Sentiment'].unique()

"""## EDA"""

data.head()

sentiment_count = data['Sentiment'].value_counts().reset_index()
sentiment_count.columns = ['Sentiment', 'count']
sentiment_count

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Sentiment", y='count', data=sentiment_count)
ax.set_title('Proporton of Sentiment', fontsize=20)
ax.set_xlabel("Sentiment", fontsize=15)
ax.set_ylabel("Count", fontsize=15)

replace_values = {"Sentiment": {'Extremely Negative': 'Negative', 'Extremely Positive': 'Positive'}}
data = data.replace(replace_values)

data.head()

sentiment_count2 = data['Sentiment'].value_counts().reset_index()
sentiment_count2.columns = ['Sentiment', 'count']
sentiment_count2

plt.figure(figsize=(6, 6))
ax2 = sns.barplot(x="Sentiment", y='count', data=sentiment_count2)
ax2.set_title('Proporton of Sentiment', fontsize=20)
ax2.set_xlabel("Sentiment", fontsize=15)
ax2.set_ylabel("Count", fontsize=15)

"""## Cleaning data - Tweets"""

df = data.copy()
df.head()

"""### Removing Punctuation"""


## Function to remove punctuation

def remove_punc(text):
    """ function to remove punctuation"""

    import string

    # replacing the punctuations with no space
    translator = str.maketrans('', '', string.punctuation)

    # return the text stripped of punctuation marks
    return text.translate(translator)


df['OriginalTweet'] = df['OriginalTweet'].apply(remove_punc)

df.head()

"""### Remove stopwords"""

## define stopwords

stop_word = stopwords.words('english')


## Function to remove stopwords

def remove_stopwords(text):
    """ function to remove stopwords"""
    ## make text smallcase and then remove stopwords
    text = [word.lower() for word in text.split() if word.lower() not in stop_word]

    # joining the list of words with space separator
    return " ".join(text)


df['OriginalTweet'] = df['OriginalTweet'].apply(remove_stopwords)

df.head()

"""### Top words in Tweets

**Vocabulary count**
"""

# create a CountVectorizer Object
count = CountVectorizer()

# Fit the data in CountVectorizer
count.fit(df['OriginalTweet'])

# collect the vocubulary items used in the vectorizer
dictionary = count.vocabulary_.items()

# lists to store the vocab and counts
vocab = []
count = []
# iterate through each vocab and count append the value to designated lists
for key, value in dictionary:
    vocab.append(key)
    count.append(value)
# store the count in panadas dataframe with vocab as index
vocab_bef_stem = pd.Series(count, index=vocab)
# sort the dataframe
vocab_bef_stem = vocab_bef_stem.sort_values(ascending=False)

"""**Bar plot of top words before Lemmatization**"""

top_vacab = vocab_bef_stem.head(20)
top_vacab.plot(kind='barh', figsize=(5, 10), xlim=(85650, 85700))

"""### Lemmatization

**Function for Lemmetize**
"""

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

nltk.download('wordnet')


# create an object of Lemmatize function

def lemmatize(text):
    """a function which stems each word in the given text"""
    text = [lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(text)


df['OriginalTweet'] = df['OriginalTweet'].apply(lemmatize)

df.head()

"""**Top words after Lemmetization**"""

# create the object of tfid vectorizer
tfid_vectorizer = TfidfVectorizer("english")

# fit the vectorizer using the text data
tfid_vectorizer.fit(df['OriginalTweet'])

# collect the vocabulary items used in the vectorizer
dictionary = tfid_vectorizer.vocabulary_.items()

# lists to store the vocab and counts
vocab1 = []
count1 = []
# iterate through each vocab and count append the value to designated lists
for key, value in dictionary:
    vocab1.append(key)
    count1.append(value)
# store the count in panadas dataframe with vocab as index
vocab_after_lem = pd.Series(count1, index=vocab1)
# sort the dataframe
vocab_after_lem = vocab_after_lem.sort_values(ascending=False)
# plot of the top vocab
top_vacab = vocab_after_lem.head(20)
top_vacab.plot(kind='barh', figsize=(5, 10), xlim=(81980, 82010))

"""# Model Training 1

## Logistic Regression
"""

## defining Independent variable
X = data.OriginalTweet

## defining dependent variable
y = data.Sentiment

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

tvec = TfidfVectorizer()
LR = LogisticRegression(solver='lbfgs', max_iter=10000)

from sklearn.pipeline import Pipeline

model = Pipeline([('vectorizer', tvec), ('classifier', LR)])
model.fit(X_train, y_train)

## model prediction
y_pred = model.predict(X_test)

""" **Performance and Accuracy**"""

from sklearn.metrics import accuracy_score, precision_score, recall_score

print('Accuracy :', accuracy_score(y_pred, y_test))
print('Precision :', precision_score(y_pred, y_test, average='weighted'))
print('Recall :', recall_score(y_pred, y_test, average='weighted'))

"""## Random Forest  """

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

model5 = Pipeline([('vectorizer', tvec), ('classifier', rf)])
model5.fit(X_train, y_train)

y_pred_rf = model5.predict(X_test)

"""### Accuracy and Precision"""

print('Accuracy :', accuracy_score(y_pred_rf, y_test))
print('Precision :', precision_score(y_pred_rf, y_test, average='weighted'))
print('Recall :', recall_score(y_pred_rf, y_test, average='weighted'))

"""## Passive Aggressive Classifier"""

from sklearn.linear_model import PassiveAggressiveClassifier

pac = PassiveAggressiveClassifier()
model2 = Pipeline([('vectorizer', tvec), ('classifier', pac)])
model2.fit(X_train, y_train)

## model prediction

y_pred_pac = model2.predict(X_test)

print('Accuracy :', accuracy_score(y_pred_pac, y_test))
print('Precision :', precision_score(y_pred_pac, y_test, average='weighted'))
print('Recall :', recall_score(y_pred_pac, y_test, average='weighted'))

"""# Trying on new Tweets"""

Tweet = ['hospitals are good', 'boy is sad',
         'he fell sick', 'he is not satisfied',
         'he is angry with his behaviour', 'taste of food bad', 'snacks are good']
result = model.predict(Tweet)

print(result)

"""# Summary

This Project is completed by 3 members - **Sumeet Agrawal, Aayush Kumar and Shafil Ahmad**. We firstly discuss the problem statement and after that look the **dataset (Coronavirus_Tweets.csv)** and start discussing on the information we got from dataset. Then we distributed the work into 3 parts. The first part is  **EDA** which is done by **Aayush Kumar**, second part is **Feature Engineering** which is done by **myself** and the third part is **Model Training and conclusion** which is done by **Shafil Ahmad**.

**Explanation of my part** :-

My part is Feature Engineering. Firstly, when I looked into Sentiment column, I found that there are 5 sentiments given. Different sentiments are – Neutral, Positive, Negative, Extremely Negative and Extremely Positive. I tried to merge Extremely Positive and Positive into Positive sentiment and similarly Extremely Negative and Negative merged into Negative sentiment. In this way we have only 3 sentiments now.
After that next process is cleaning tweets. Tweets contains lots of punctuation and stopwords. So, I removed these words to make tweets cleaner. The reason behind to remove these words because these words doesn’t make any sense in sentiment analysis. Also, I applied stemming operation to bring words into their root word.

# Pickle File
"""

import pickle

filename = 'Sentiment Analysis'
pickle.dump(model, open(filename, 'wb'))

model_load = pickle.load(open(filename, 'rb'))
model_load.predict(X_test)
