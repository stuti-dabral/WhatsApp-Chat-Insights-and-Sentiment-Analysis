import re
import pandas as pd
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sentiments = SentimentIntensityAnalyzer()

def preprocess(data):

    pattern = '\d{1,2}\/\d{1,2}\/\d{2,4},\s\d{1,2}:\d{2}\s[AP]M\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_msgs': messages, 'datetime': dates})

    #converting to date-time format
    df['datetime'] = df['datetime'].str.rstrip(' - ')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%y, %I:%M %p')

    #separating the usernames and messages
    users = []
    messages = []
    for row in df['user_msgs']:
        sep = re.split('\A(.*?[\w\W]+?):\s', row)
        
        #to check if it is a user message or notification
        if (len(sep)>1):
            users.append(sep[1])
            messages.append(sep[2])
        else:
            users.append('notification')
            messages.append(sep[0])

    df['user'] = users
    df['message'] = messages
    df.drop("user_msgs", axis=1, inplace=True)

    df.drop(df[df.user == 'notification'].index, axis=0, inplace=True)

    #Extracting datetime separately
    df['day'] = df['datetime'].dt.day
    df['day_name'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month_name()
    df['month_num'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year

    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append("00" + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    #converting hinglish to english but it takes a lot of time
    #from deep_translator import GoogleTranslator
    # translator = GoogleTranslator(source='auto', target='en')
    # df['en_message'] = df['message'].apply(lambda x: translator.translate(x))

    #to get the sentiment score for each message
    df["compound"] = [sentiments.polarity_scores(i)["compound"] for i in df["message"]]

    #to classify message into positive or negative or neutral
    def classify(compound_score):
        if compound_score > 0.05:       #positive
            return 1
        elif compound_score < -0.05:    #negative
            return -1
        else:                           #neutral
            return 0
    
    df['value'] = df['compound'].apply(classify)

    return df