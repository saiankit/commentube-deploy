import re
import pandas as pd
import numpy as np
import nltk
import pickle
import contractions
import unidecode
from nltk.tokenize import sent_tokenize
from autocorrect import Speller
import nltk
from nltk.corpus import stopwords


def removePatterns(sentence): 
    cleaned_text  = re.sub("\\s*\\b(?=\\w*(\\w)\\1{2,})\\w*\\b",' ',sentence)
    return (cleaned_text)

def reducing_incorrect_character_repeatation(text):
    Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)
    Formatted_text = Pattern_alpha.sub(r"\1\1", text) 
    Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')
    Combined_Formatted = Pattern_Punct.sub(r'\1', Formatted_text)
    Final_Formatted = re.sub(' {2,}',' ', Combined_Formatted)
    return Final_Formatted

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U0001F1FF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def remove_number_colon_number(sentence):
  pattern = re.compile(r'(\d+):(\d+)')
  match = pattern.search(sentence)

  if match:
    sentence = sentence.replace(match.group(0), '')
    
  return sentence   
spell = Speller(lang='en')

def pre_process(comments_df):
    comments_df = comments_df[['cid', 'text']]
    comments_df = comments_df.dropna(axis=0, subset=['text'])
    comments_df["original"] = comments_df["text"]
    comments_df['text'] = comments_df['text'].apply(lambda x: x.lower())
    comments_df['text'] = comments_df['text'].apply(lambda x: remove_number_colon_number(x))
    comments_df['text'] = comments_df['text'].apply(lambda x: reducing_incorrect_character_repeatation(x))
    comments_df['text'] = comments_df['text'].str.replace(r'(.)\1{2,}', r'\1')
    comments_df['text'] = comments_df['text'].apply(lambda x: re.sub(r"@\w", "", x))
    comments_df['text'] = comments_df['text'].apply(lambda x: re.sub(r":\w", "", x))
    comments_df['text'] = comments_df['text'].apply(lambda x:re.sub(r"[^a-zA-Z0-9:$-,%.!]+", ' ', x))
    comments_df['text'] = comments_df['text'].apply(lambda x: contractions.fix(x))
    comments_df['text'] = comments_df['text'].apply(lambda x: unidecode.unidecode(x))
    for index, row in comments_df.iterrows():
        row['text'] = remove_emojis(row['text'])
    comments_df['text'] = comments_df['text'].str.normalize('NFKD')    
    comments_df['text'] = comments_df['text'].replace('-', np.nan)
    comments_df['text'] = comments_df['text'].replace('  ', '', regex=True)
    comments_df['text'] = comments_df['text'].replace('"', np.nan)
    comments_df['text'] = comments_df['text'].replace('\n', '', regex=True)
    comments_df['text'] = comments_df['text'].replace(r'[\u200b\xa0]', '', regex=True)
    comments_df = comments_df.drop(index=comments_df[comments_df['text'] == ' '].index)
    comments_df = comments_df.drop(index=comments_df[comments_df['text'] == ''].index)
    comments_df = comments_df.dropna(axis=0, subset=['text'])
    pattern = lambda x: x if pd.isna(x) else re.sub("\\s*\\b(?=\\w*(\\w)\\1{2,})\\w*\\b",' ',x)
    comments_df['text'] = comments_df['text'].apply(pattern)
    comments_df['text'] = comments_df['text'].apply(lambda x: re.sub("^\s+", "", x))
   
    lemmatizer = nltk.stem.WordNetLemmatizer()
    comments_df['text'] = comments_df['text'].astype(str)
    comments_df['text'] = comments_df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    sentences = comments_df['text'].to_list()
    original = comments_df['original'].to_list()
    return sentences 
