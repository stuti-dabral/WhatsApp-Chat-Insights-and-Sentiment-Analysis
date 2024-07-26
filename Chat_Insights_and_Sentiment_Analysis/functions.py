import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud 
from collections import Counter
import emoji
import spacy

nlp = spacy.load("en_core_web_sm")

#STATISTICS
def get_stats(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    #total number of messages
    total_msgs = df.shape[0]

    #total number of words
    words = []
    for message in df['message']:
        words.extend(message.split()) 

    #total number of media shared
    total_media = df[df['message'] == '<Media omitted>\n'].shape[0]

    #total number of links shared
    url = URLExtract()
    links = []
    for message in df['message']:
        links.extend(url.find_urls(message))

    return total_msgs, len(words), total_media, len(links)


#MONTHLY TIMELINE
def monthly_timeline(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i][:3] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline


#DAILY TIMELINE
def daily_timeline(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby(['year','month_num','month','day']).count()['message'].reset_index()

    time = []
    for i in range(daily_timeline.shape[0]):
        time.append(str(daily_timeline['day'][i]) + "-" + str(daily_timeline['month'][i])[:3] + "-" + str(daily_timeline['year'][i]))

    daily_timeline['time'] = time

    return daily_timeline


#MONTHLY ACTIVITY ANALYSIS
def monthly_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


#WEEKLY ACTIVITY ANALYSIS
def weekly_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


#ACTIVITY MAP
def activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return heatmap


#MOST ACTIVE MEMBERS
def most_active(df):
    var = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'Name', 'user': 'Percentage'})
    
    return var, df.head(10)


#WORDCLOUD
def word_cloud(selected_user, df):

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

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    img = wc.generate(df['message'].str.cat(sep=" "))
    
    return img


#MOST COMMON WORDS
def most_common_words(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['message'] != '<Media omitted>\n']

    words = []
    for message in df['message']:
        doc = nlp(message)
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            words.append(token.lemma_)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    
    return most_common_df


#EMOJIS ANALYSIS
def emojis_used(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df["message"]:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

