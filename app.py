import streamlit as st
import streamlit.components.v1 as components
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import nltk
from nltk.tokenize import sent_tokenize
import ktrain
from ktrain.text.sentiment import SentimentAnalyzer
from preprocess import pre_process
from components import card
from utilities import generate_text, generate_image_link
from youtube_comment_downloader import *
from sentence_transformers import SentenceTransformer, util
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, T5Config, pipeline,
    AutoModelForSequenceClassification, AutoTokenizer
)
from streamlit_option_menu import option_menu
from sentiment_analysis import predict_sentiments
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

pd.set_option('mode.chained_assignment', None)  # Disable pandas SettingWithCopyWarning

st.set_page_config(
    page_title="CommenTube",
    page_icon="📽️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
     <style>
            footer {visibility: hidden;}
     </style>
    """,
     unsafe_allow_html=True
)

with st.sidebar:
    choose = option_menu("CommenTube📽️", ["About", "Classification", "Statistics", "Clustering"],
                         icons=['house', 'camera fill', 'kanban', 'book'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#0F1116"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

#predictor = ktrain.load_predictor('model1')
#classifier = SentimentAnalyzer()


def downloadCommentsFromURL(youtube_url):
    """This function downloads the comments from a YouTube URL.

    Args:
        youtube_url (str): The YouTube URL.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the comments.
    """

    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(youtube_url)
    comments_data = []

    for comment in comments:
        data = {}
        data["text"] = comment["text"]
        data["cid"] = comment["cid"]
        data["time_parsed"] = comment["time"]
        comments_data.append(data)

    df = pd.DataFrame(comments_data)

    return df

def downloadComments():
    df = downloadCommentsFromURL(youtube_url)
    sentences, original = pre_process(df)
    #out = predictor.predict(sentences)
    sentiments = predict_sentiments(sentences)
    for i in range(0, len(sentences)):
        st.write(card(original[i], out[i], list(sentiments[i].keys())[0]), unsafe_allow_html=True)

examples = [
    {
        "url" : "https://www.youtube.com/watch?v=LRJPk9BmJY4",
        "path" : "./examples/example4.csv"
    },
    {
        "url" : "https://www.youtube.com/watch?v=6ZrlsVx85ek",
        "path" : "./examples/example2.csv"
    },
       {
        "url" : "https://www.youtube.com/watch?v=ZMjKp5j1Lt8",
        "path" : "./examples/example3.csv"
    },
    {
        "url" : "https://www.youtube.com/watch?v=l5qU2Yrq_mc",
        "path" : "./examples/example4.csv"
    },

] 
col1, col2, col3 = st.columns([3,2,1])

def show_image(youtube_url):
    col2.image(generate_image_link(youtube_url))
    col2.write(generate_text(youtube_url))

def showCards(originals, sentiments, predictions):
    labels = st.session_state['labels']
    col11, col22 = st.columns(2)
    print(labels)
    for i in range(0, len(originals)):
        if (i % 2 == 0 ):
            col11.write(card(originals[i], predictions[i], sentiments[i]), unsafe_allow_html=True)
        else:
            col22.write(card(originals[i], predictions[i], sentiments[i]), unsafe_allow_html=True)


if 'labels' not in st.session_state:
    st.session_state['labels'] =  ['interrogative', 'imperative', 'corrective', 'miscellaneous', 'others']

def handle_click(selected_labels):
    st.session_state['labels'] = selected_labels

comment_labels = ['interrogative', 'imperative', 'corrective', 'miscellaneous', 'others']
selected_labels = []

youtube_url = col1.text_input('Enter Youtube URL')
col1.write('(or)')
selected = col1.selectbox('Select from Example videos', ('','Example 1', 'Example 2', 'Example 3', 'Example 4'))
if col1.button('Run 🪄'):
    if selected == "":
        show_image(youtube_url)
        downloadComments()
    else:
        if selected == "Example 1":
            show_image(examples[0]["url"])
            df = pd.read_csv(examples[0]["path"])
            originals = df['original'].to_list()
            sentiments = df['sentiments'].to_list()
            predictions = df['predictions'].to_list()
            selected_labels = st.multiselect(
                'Choose Labels',
                options=comment_labels,
                default=st.session_state.labels,
                on_change=handle_click,
                args=(selected_labels,)
            )
            fig1column, fig2column = st.columns([1,1])
            st.session_state['labels'] = selected_labels
            comment_types = df["sentiments"].value_counts()
            df_comment_types = pd.DataFrame({"sentiment": comment_types.index, "Count": comment_types.values})
            fig = px.pie(df_comment_types, values="Count", names="sentiment", title="Sentiment Type Distribution")
            fig1column.plotly_chart(fig)
            fig.update_layout(
                width=800,  # Set the width of the chart
                height=500,  # Set the height of the chart
                margin=dict(l=50, r=50, t=50, b=50),  # Set the margins
                autosize=False,  # Disable autosizing
            )

            prediction_types = df["predictions"].value_counts()
            df_prediction_types = pd.DataFrame({"prediction": prediction_types.index, "Count": prediction_types.values})
            fig = px.pie(df_prediction_types, values="Count", names="prediction", title="Prediction Type Distribution")
            fig2column.plotly_chart(fig)
            
            fig.update_layout(
    width=800,  # Set the width of the chart
    height=500,  # Set the height of the chart
    margin=dict(l=50, r=50, t=50, b=50),  # Set the margins
    autosize=False,  # Disable autosizing
)

            showCards(originals, sentiments, predictions)
            
