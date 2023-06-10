import numpy as np
import pandas as pd
import time
import datetime
import gc
import random
from nltk.corpus import stopwords
import re

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk

import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import AutoTokenizer
import tensorflow as tf
import pickle
import contractions
import unidecode
from nltk.tokenize import sent_tokenize
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

columns = ['text', 'label']
df = pd.read_csv("pre_processed.csv", names=columns)
df["label"] = df["label"].replace({"imperative": 0, "interrogative": 1, "miscellaneous": 2, "corrective": 3, "positive": 4, "negative": 4})

sw = stopwords.words('english')

tweets = df.text.values

max_len = 0
for sent in tweets:
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2)

train_encoded_inputs = tokenizer(train_df['text'].tolist(),
                                 add_special_tokens = True,
                                 padding='max_length', 
                                 truncation=True, 
                                 max_length=max_len, 
                                 return_token_type_ids=False,
                                 return_tensors = 'tf')

train_dataset = tf.data.Dataset.from_tensor_slices((train_encoded_inputs, train_df['label'].values))

# formatting the data as required by bert model
def map_bert(inputs, labels):
  inputs = {'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']}
  
  return inputs, labels
train_dataset = train_dataset.map(map_bert)

dataset = train_dataset.shuffle(100000).batch(64)
DS_LEN = len(dataset)
print(len(dataset))
# take 80% for train and 20% for validation
SPLIT = 0.8
train_ds = dataset.take(round(DS_LEN*SPLIT))
val_ds = dataset.skip(round(DS_LEN*SPLIT))

print(len(train_ds))
print(len(val_ds))

from transformers import TFAutoModel
bert = TFAutoModel.from_pretrained('bert-base-uncased')

# create model architecture
#n_classes = len((train_df.target.unique()))

# Input layers
input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=np.int32, name='input_ids' )
mask = tf.keras.layers.Input(shape=(max_len,), dtype=np.int32, name = 'attention_mask')

# bert embeddings
embeddings = bert([input_ids, mask])[0]
cls_token = embeddings[:,0,:]

# keras layers
#x = tf.keras.layers.GlobalMaxPool1D()(embeddings)
x = tf.keras.layers.BatchNormalization()(cls_token)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)

# output layer
y = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# create the model
model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

model.summary()

# freezing the pretrained bert layer
model.layers[2].trainable = False
model.summary()

import transformers

learning_rate = 5e-5

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.categorical_crossentropy()
metric = tf.keras.metrics.categorical_accuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

callbacks = [tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.1,min_delta=0.001,monitor='val_loss'),
             tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.001, monitor='val_loss')]

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 3,
    callbacks = callbacks
)


model.save_weights('bert_weights.h5')

model.evaluate(val_ds)

test_encoded_inputs = tokenizer(test_df['text'].tolist(),
                                 add_special_tokens = True,
                                 padding='max_length', 
                                 truncation=True, 
                                 max_length=max_len, 
                                 return_token_type_ids=False,
                                 return_tensors = 'tf')

test_dataset = tf.data.Dataset.from_tensor_slices(dict(test_encoded_inputs))

test_ds = test_dataset.shuffle(100000).batch(64)
test_pred = model.predict(test_ds)
test_df.to_csv("test_before.csv")
# create the target labels for test data
test_target = np.round(test_pred).flatten()
test_target

test_df['target'] = test_target.astype('int')

test_df.to_csv("test.csv")
