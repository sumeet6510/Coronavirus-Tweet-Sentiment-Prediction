# Coronavirus-Tweet-Sentiment-Analysis


### Index
#### 1. Introduction
#### 2. Exploratory Data Analysis./Reviwing Our Dataset
#### 3. Data Preprocessing.
#### 4. Classification- MULTICLASS AND BINARY.
#### 5. Evaluation.
#### 6. Challenges.
#### 7. Conclusion


# 1. Introduction

Our objective is to build a classification model to predict the sentiment of COVID-19 tweets.The tweets have been pulled from Twitter and manual tagging has been done then. The names and usernames have been given codes to avoid any privacy concerns.

![image](https://user-images.githubusercontent.com/83903018/124346241-60aa9b80-dbfb-11eb-97a3-a9251066302c.png)

## We are given the following information:

* Username: Unique user-IDs of the users
* Location: Location of the user
* Tweet At: Date at which the tweet was made
* Original Tweet: The exact tweet
* Sentiment: Sentiment of the tweet

# 2. Exploratory Data Analysis

* The original dataset has 6 columns and 41157 rows.

* In order to analyse various sentiments, We require just two columns named Original Tweet and Sentiment.

* There are five types of sentiments- Extremely Negative, Negative, Neutral, Positive and Extremely Positive.

* The columns such as “UserName” and “ScreenName” does not give any meaningful insights for our analysis.

* All tweets data collected from the months of March and April 2020. Bar plot shows us the number of unique values in each column.

* There are 20.87%(8567) null values in various places of location column. Most of the tweets came from London followed by U.S.

* There are some words like ‘coronavirus’,’grocery store’, having the maximum frequency in our dataset.

* There are various #hashtags in tweets column.But they are almost same in all sentiments.

* Most of the peoples are having positive sentiments about various issues shows us their optimism during pandemic times.

* Very few people are having extremely negatives thoughts about Covid-19.




