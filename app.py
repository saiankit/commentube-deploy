import streamlit as st 
import streamlit.components.v1 as components
from preprocess import pre_process 
from components import card
from utilities import generate_text , generate_image_link
import time
import pandas as pd
from youtube_comment_downloader import *
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, pipeline
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
from sentiment_analysis import sentiment_analyse
import ktrain
from ktrain.text.sentiment import SentimentAnalyzer

st.set_page_config(
    page_title="CommentTube",
    page_icon="💬",
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


def embedd_sentences(sentences, model_name):
    """
    Embeds a list of sentences using a sentence transformer model.

    Args:
        sentences: A list of sentences to be embedded.
        model_name: The name of the sentence transformer model to use.

    Returns:
        A tuple of (corpus_embeddings, sentence_embeddings).
    """

    # Get the sentence transformer model.
    model = SentenceTransformer(model_name)

    # Encode the sentences.
    sentence_embeddings = model.encode(sentences, batch_size=128)

    # Normalize the sentence embeddings.
    corpus_embeddings = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)

    return corpus_embeddings, sentence_embeddings

def cluster_sentences(sentences, sentence_embedding, distance_threshold):
    """
    Clusters a list of sentences using agglomerative clustering.

    Args:
        sentences: A list of sentences to be clustered.
        sentence_embeddings: The sentence embeddings of the sentences to be clustered.
        distance_threshold: The distance threshold for clustering.

    Returns:
        A dictionary of clusters, where the key is the cluster id and the value is a list of sentences in the cluster.
    """

    # Create the clustering model.
    clustering_model = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=distance_threshold)

    # Fit the clustering model to the sentence embeddings.
    clustering_model.fit(sentence_embeddings)

    # Get the cluster assignments.
    cluster_assignment = clustering_model.labels_

    # Create a dictionary of clusters.
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        clustered_sentences[cluster_id].append(sentences[sentence_id])

    return clustered_sentences


#predictor = ktrain.load_predictor('model1')
#classifier = SentimentAnalyzer()

def downloadComments():
    st1 = time.process_time()    
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(youtube_url)
    comments_data = []
    for comment in comments:
        data = {}
        data["text"] = comment["text"]
        data["cid"] = comment["cid"]
        comments_data.append(data)
    json_data = json.dumps(comments_data)
    df = pd.read_json(json_data)
    et = time.process_time()
    res = et - st1
    print('Total time:', res, 'seconds')
    sentences, original = pre_process(df)
    #out = predictor.predict(sentences)
   # sentiments = classifier.predict(sentences)
    for i in range(0, len(sentences)):
        st.write(card(original[i], out[i], list(sentiments[i].keys())[0]), unsafe_allow_html=True)
    # positive_comments_percentage, negative_comments_percentage = sentiment_analyse(comments_df)
    # with col3:
    #     st.markdown(f'<h1 style="color:#33cf33;font-size:24px;">{"Positive Comments "} {positive_comments_percentage} {" %"} </h1>', unsafe_allow_html=True)
    #     st.markdown(f'<h1 style="color:#FF6D60;font-size:24px;">{"Negative Comments "} {negative_comments_percentage} {" %"} </h1>', unsafe_allow_html=True)
    # corpus_embeddings, sentence_embeddings = embedd_sentences(sentences, embedding_model)
    # clustered_sentences = cluster_sentences(sentences, sentence_embeddings, 0.65)
    # col11, col22 = st.columns(2)
    # summaries = []
    # for i, cluster in clustered_sentences.items():
    #     combined_sentence = ""
    #     for sentence in cluster:
    #         combined_sentence += sentence
    #     combined_sentence = combined_sentence.strip().replace("\n", "")
    #     combined_sentence = manage_sentence(combined_sentence)
    #     pipe = pipeline('summarization', model="./t5-small/")
    #     output = pipe(combined_sentence)
    #     summary = '\n'.join(sent_tokenize(output[0]['summary_text']))
    #     element = {}
    #     element["cluster"] = cluster
    #     element["summary"] = summary
    #     summaries.append(element)
    # for element in summaries:
    #     st.write(element)

col1, col2, col3 = st.columns([3,2,1])

def show_image(youtube_url):
    with col2:
        st.image(generate_image_link(youtube_url))
        st.write(generate_text(youtube_url))

def process_sentiments(sentiments):
    sentiments_final = []
    for i in range(0, len(sentiments)):
        if sentiments[i].startswith("{'NEU"):
            sentiments_final.append("Neutral")
        elif sentiments[i].startswith("{'POS"):
            sentiments_final.append("Positive")
        else:
            sentiments_final.append("Negative")
    return sentiments_final

with col1:
    st.header("CommenTube📽️")
    youtube_url = st.text_input('Enter Youtube URL')
    st.write('(or)')
    selected = st.selectbox('Select from Example videos', ('','Example 1', 'Example 2'))
    if st.button('Run 🪄'):
        if selected == "":
            show_image(youtube_url)
            downloadComments()
        else:
            if selected == "Example 1":
                show_image("https://www.youtube.com/watch?v=ZMjKp5j1Lt8")
                df = pd.read_csv("video4-final.csv")
                originals = df['original'].to_list()
                sentiments = process_sentiments(df['sentiments'].to_list())
                predictions = df['predictions'].to_list()
                col11, col22 = st.columns(2)
                for i in range(0, len(originals)):
                    if (i % 2 == 0 ):
                        with col11:
                            st.write(card(originals[i], predictions[i], sentiments[i]), unsafe_allow_html=True)
                    else:
                        with col22:
                             st.write(card(originals[i], predictions[i], sentiments[i]), unsafe_allow_html=True)

