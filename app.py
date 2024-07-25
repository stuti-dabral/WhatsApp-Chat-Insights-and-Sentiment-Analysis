import streamlit as st
import preprocess as pre
import functions as func
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sentiments as sent
import numpy as np

st.set_page_config(page_title="Whatsapp Chat Insights", layout="wide")

#main heading
st.markdown("<h1 style='text-align: center; color: lightgreen;'>Whatsapp Chat Insights and Sentiment Analysis</h1>", unsafe_allow_html=True)

st.sidebar.title("Upload your chats here")

#for uploading file
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    #to read file as bytes
    bytes_data = uploaded_file.getvalue()
    #converting byte stream into string
    data = bytes_data.decode("utf-8")
    df = pre.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")

    #dropdown of members name
    selected_user = st.sidebar.selectbox("Select User", user_list)


    #INSIGHTS

    if st.sidebar.button("Show insights"):

        #fetch statistics
        total_msgs, total_words, total_media, total_links = func.get_stats(selected_user , df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("<h3 style='text-align: left; color: white;'>Total Messages</h3>", unsafe_allow_html=True)
            st.subheader(total_msgs)

        with col2:
            st.markdown("<h3 style='text-align: left; color: white;'>Total Words</h3>", unsafe_allow_html=True)
            st.subheader(total_words)

        with col3:
            st.markdown("<h3 style='text-align: left; color: white;'>Media Shared</h3>", unsafe_allow_html=True)
            st.subheader(total_media)

        with col4:
            st.markdown("<h3 style='text-align: left; color: white;'>Links Shared</h3>", unsafe_allow_html=True)
            st.subheader(total_links)


        #monthly timeline
        st.title("Monthly Timeline")
        timeline_month = func.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline_month['time'], timeline_month['message'])
        plt.xlabel("Months")
        plt.ylabel("Number of messages")
        plt.xticks(rotation='vertical')
        st.pyplot(fig)


        #daily timeline
        st.title("Daily Timeline")
        timeline_daily = func.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline_daily['time'], timeline_daily['message'], color='black')
        #to prevent overlapping of dates on xlabel
        myLocator = mticker.MultipleLocator(10)
        ax.xaxis.set_major_locator(myLocator)
        plt.xlabel("Dates")
        plt.ylabel("Number of messages")
        plt.xticks(rotation='vertical')
        st.pyplot(fig)


        #monthly and weekly activity
        st.title('Activity Analysis')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Active Months")
            active_month = func.monthly_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(active_month.index, active_month.values, color='orange')
            plt.xlabel("Months")
            plt.ylabel("Number of messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most Active Days")
            active_day = func.weekly_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(active_day.index, active_day.values, color='purple')
            plt.xlabel("Days")
            plt.ylabel("Number of messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


        #activity map
        st.title("Activity Map")
        activity = func.activity_map(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(activity)
        ax.set(xlabel='Time Period', ylabel='Day of Week')
        st.pyplot(fig)


        #finding the most active members (overall)
        if selected_user == 'Overall':
            var, new_df = func.most_active(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                st.title('Most Active Users')
                ax.bar(var.index, var.values, color='green')
                plt.xticks(rotation=45)
                plt.xlabel("Name")
                plt.ylabel("No. of messages")
                st.pyplot(fig)

            with col2:
                st.title("Top Percentage Contribution")
                st.dataframe(new_df)

    
        #wordcloud
        st.title("Word Cloud")
        img = func.word_cloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(img)
        st.pyplot(fig)


        #most common words
        st.title('Most Commmon Words')
        most_common_df = func.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xlabel("Frequency")
        st.pyplot(fig)


        #emoji analysis
        emoji_df = func.emojis_used(selected_user, df)
        st.title("Emoji Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)

        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)



    #SENTIMENT ANALYSIS

    if st.sidebar.button("Show sentiment analysis"):

        #overall sentiment
        overall = sent.overall_senti(selected_user, df)
        st.markdown(f"<h3 style='text-align: center; color: yellow;'> The overall sentiment is {overall}</h3>", unsafe_allow_html=True)


        #monthly activity sentiments
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Sentiments (Positive)</h3>", unsafe_allow_html=True)
            months = sent.monthly_sentiments(selected_user, df, 1)
            fig, ax = plt.subplots()
            ax.bar(months.index, months.values, color='green')
            plt.xlabel("Months")
            plt.ylabel("No. of positive messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Sentiments (Neutral)</h3>", unsafe_allow_html=True)
            months = sent.monthly_sentiments(selected_user, df, 0)
            fig, ax = plt.subplots()
            ax.bar(months.index, months.values, color='grey')
            plt.xlabel("Months")
            plt.ylabel("No. of neutral messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col3:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Sentiments (Negative)</h3>", unsafe_allow_html=True)
            months = sent.monthly_sentiments(selected_user, df, -1)
            fig, ax = plt.subplots()
            ax.bar(months.index, months.values, color='red')
            plt.xlabel("Months")
            plt.ylabel("No. of negative messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


        #day-wise activity sentiments
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h3 style='text-align: center; color: white;'>Day-Wise Sentiments (Positive)</h3>",unsafe_allow_html=True)
            days = sent.day_sentiments(selected_user, df, 1)
            fig, ax = plt.subplots()
            ax.bar(days.index, days.values, color='green')
            plt.xlabel("Days")
            plt.ylabel("No. of positive messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Day-Wise Sentiments (Neutral)</h3>",unsafe_allow_html=True)
            days = sent.day_sentiments(selected_user, df, 0)
            fig, ax = plt.subplots()
            ax.bar(days.index, days.values, color='grey')
            plt.xlabel("Days")
            plt.ylabel("No. of neutral messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col3:
            st.markdown("<h3 style='text-align: center; color: white;'>Day-Wise Sentiments (Negative)</h3>",unsafe_allow_html=True)
            days = sent.day_sentiments(selected_user, df, -1)
            fig, ax = plt.subplots()
            ax.bar(days.index, days.values, color='red')
            plt.xlabel("Days")
            plt.ylabel("No. of negative messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


        #weekly activity map
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h3 style='text-align: center; color: white;'>Weekly Activity Map (Positive)</h3>",unsafe_allow_html=True)
            heatmp = sent.senti_heatmap(selected_user, df, 1)
            fig, ax = plt.subplots()
            ax = sns.heatmap(heatmp)
            ax.set(xlabel='Time Period', ylabel='Day of Week')
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Weekly Activity Map (Neutral)</h3>",unsafe_allow_html=True)
            heatmp = sent.senti_heatmap(selected_user, df, 0)
            fig, ax = plt.subplots()
            ax = sns.heatmap(heatmp)
            ax.set(xlabel='Time Period', ylabel='Day of Week')
            st.pyplot(fig)
        
        with col3:
            st.markdown("<h3 style='text-align: center; color: white;'>Weekly Activity Map (Negative)</h3>",unsafe_allow_html=True)
            heatmp = sent.senti_heatmap(selected_user, df, -1)
            fig, ax = plt.subplots()
            ax = sns.heatmap(heatmp)
            ax.set(xlabel='Time Period', ylabel='Day of Week')
            st.pyplot(fig)


        #daily timeline
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h3 style='text-align: center; color: white;'>Daily Timeline (Positive)</h3>",unsafe_allow_html=True)
            timeline = sent.daily_timeline(selected_user, df, 1)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            #to prevent overlapping of dates on xlabel
            myLocator = mticker.MultipleLocator(8)
            ax.xaxis.set_major_locator(myLocator)
            plt.xlabel("Dates")
            plt.ylabel("No. of positive messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Daily Timeline (Neutral)</h3>",unsafe_allow_html=True)
            timeline = sent.daily_timeline(selected_user, df, 0)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='grey')
            #to prevent overlapping of dates on xlabel
            myLocator = mticker.MultipleLocator(10)
            ax.xaxis.set_major_locator(myLocator)
            plt.xlabel("Dates")
            plt.ylabel("No. of neutral messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col3:
            st.markdown("<h3 style='text-align: center; color: white;'>Daily Timeline (Negative)</h3>",unsafe_allow_html=True)
            timeline = sent.daily_timeline(selected_user, df, -1)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='red')
            #to prevent overlapping of dates on xlabel
            myLocator = mticker.MultipleLocator(5)
            ax.xaxis.set_major_locator(myLocator)
            plt.xlabel("Dates")
            plt.ylabel("No. of negative messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


        #monthly timeline
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Timeline (Positive)</h3>",unsafe_allow_html=True)
            timeline = sent.monthly_timeline(selected_user, df, 1)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xlabel("Months")
            plt.ylabel("No. of positive messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Timeline (Neutral)</h3>",unsafe_allow_html=True)
            timeline = sent.monthly_timeline(selected_user, df, 0)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='grey')
            plt.xlabel("Months")
            plt.ylabel("No. of neutral messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col3:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Timeline (Negative)</h3>",unsafe_allow_html=True)
            timeline = sent.monthly_timeline(selected_user, df, -1)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='red')
            plt.xlabel("Months")
            plt.ylabel("No. of negative messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


        #percentage contributed
        if selected_user == 'Overall':
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("<h3 style='text-align: left; color: white;'>Most Positive Contribution</h3>", unsafe_allow_html=True)
                x = sent.percentage(df, 1)
                st.dataframe(x)

            with col2:
                st.markdown("<h3 style='text-align: left; color: white;'>Most Neutral Contribution</h3>", unsafe_allow_html=True)
                y = sent.percentage(df, 0)
                st.dataframe(y)

            with col3:
                st.markdown("<h3 style='text-align: left; color: white;'>Most Negative Contribution</h3>", unsafe_allow_html=True)
                z = sent.percentage(df, -1)
                st.dataframe(z)


        #sentiment of members
        if selected_user == 'Overall':
            x = df['user'][df['value'] == 1].value_counts().head(10)
            y = df['user'][df['value'] == -1].value_counts().head(10)
            z = df['user'][df['value'] == 0].value_counts().head(10)

            col1,col2,col3 = st.columns(3)

            with col1:
                st.markdown("<h3 style='text-align: center; color: white;'>Most Positive Members</h3>", unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xlabel("Members")
                plt.ylabel("No. of positive messages")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.markdown("<h3 style='text-align: center; color: white;'>Most Neutral Members</h3>", unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='grey')
                plt.xlabel("Members")
                plt.ylabel("No. of neutral messages")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col3:
                st.markdown("<h3 style='text-align: center; color: white;'>Most Negative Members</h3>", unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='red')
                plt.xlabel("Members")
                plt.ylabel("No. of negative messages")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)


        #wordcloud
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h3 style='text-align: center; color: white;'>Positive WordCloud</h3>", unsafe_allow_html=True)
            # Creating wordcloud of positive words
            df_wc = sent.create_wordcloud(selected_user, df, 1)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)
            
        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Neutral WordCloud</h3>", unsafe_allow_html=True)
            df_wc = sent.create_wordcloud(selected_user, df, 0)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)
            
        with col3:
            st.markdown("<h3 style='text-align: center; color: white;'>Negative WordCloud</h3>", unsafe_allow_html=True)
            df_wc = sent.create_wordcloud(selected_user, df, -1)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)


        #most common words
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<h3 style='text-align: center; color: white;'>Positive Words</h3>", unsafe_allow_html=True)
            most_common_df = sent.most_common_words(selected_user, df, 1)
            fig, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1],color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
            
        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Neutral Words</h3>", unsafe_allow_html=True)
            most_common_df = sent.most_common_words(selected_user, df, 0)  
            fig, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1],color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
            
        with col3:
            st.markdown("<h3 style='text-align: center; color: white;'>Negative Words</h3>", unsafe_allow_html=True)
            most_common_df = sent.most_common_words(selected_user, df, -1)
            fig, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

