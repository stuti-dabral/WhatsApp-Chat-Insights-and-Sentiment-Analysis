import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import spacy

nlp = spacy.load("en_core_web_sm")

# #OVERALL SENTIMENT
def overall_senti(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    pos = df['value'].value_counts()[1]
    neu = df['value'].value_counts()[0]
    neg = df['value'].value_counts()[-1]

    print(pos)
    print(neu)
    print(neg)   

    if (pos > neu) and (pos > neg):
       return "Positive 😊 "
    elif (neg > pos) and (neg > neu):
       return "Negative 😠 "
    else:
        return "Neutral 😐"


#MONTHLY ACTIVITY SENTIMENTS
def monthly_sentiments(selected_user, df, n):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['value'] == n]
    return df['month'].value_counts()


#DAY-WISE ACTIVITY SENTIMENTS
def day_sentiments(selected_user, df, n):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['value'] == n]
    return df['day_name'].value_counts()


#WEEKLY SENTIMENTS
def senti_heatmap(selected_user, df, n):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['value'] == n]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


#DAILY TIMELINE
def daily_timeline(selected_user, df, n):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['value'] == n]
    daily_timeline = df.groupby(['year','month_num','month','day']).count()['message'].reset_index()

    time = []
    for i in range(daily_timeline.shape[0]):
        time.append(str(daily_timeline['day'][i]) + "-" + str(daily_timeline['month'][i])[:3] + "-" + str(daily_timeline['year'][i]))

    daily_timeline['time'] = time

    return daily_timeline


#MONTHLY TIMELINE
def monthly_timeline(selected_user, df, n):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['value'] == n]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i][:3] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline


#CONTRIBUTION PERCENTAGE
def percentage(df, n):

    df = round((df['user'][df['value']==n].value_counts() / df[df['value']==n].shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    
    return df


#WORDCLOUD
def create_wordcloud(selected_user, df, n):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['message'] != '<Media omitted>\n']

    #REMOVING STOP WORDS
    def remove_stop_words(message):         
        doc = nlp(message)
        filtered_tokens = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            filtered_tokens.append(token.lemma_)
        
        return " ".join(filtered_tokens)
    
    df['message'] = df['message'].apply(remove_stop_words)
    df['message'] = df['message'][df['value'] == n]

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    img = wc.generate(df['message'].str.cat(sep=" "))
    
    return img
    

#MOST COMMON WORDS
def most_common_words(selected_user, df, n):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['message'] != '<Media omitted>\n']
 
    words = []
    for message in df['message'][df['value'] == n]:
        doc = nlp(message)
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            words.append(token.lemma_)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    
    return most_common_df
