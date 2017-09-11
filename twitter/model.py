# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 21:56:29 2017

@author: stevewyl
"""

import csv
import os
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
import collections

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import  BatchNormalization

'''
tokenizer = TweetTokenizer()

# pandas cannot load data correctly because there are commas in the tweet content
with open("tweets.csv", "r") as infile, open("quoted.csv", "wb") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for line in reader:
        newline = [','.join(line[:-3])] + line[-3:]
        writer.writerow(newline)
        
df = pd.read_csv('quoted.csv')

df.isnull().sum()
df.Sentiment.value_counts()
# find two error rows, just ingore them
df = df.drop(df.index[[8834,535880]])
df['Sentiment'] = df['Sentiment'].map(int)

df.reset_index(inplace=True, drop=True)
df.drop('index', axis=1, inplace=True)
# delete useless columns
df.drop(['ItemID', 'SentimentSource'], axis=1, inplace=True)
pd.to_pickle(df, 'tweet_dataset.pkl')
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
tokenizer.fit_on_texts(df.SentimentText.values)
vocab = tokenizer.word_index
'''

df = pd.read_pickle('./data/tweet_dataset.pkl')
half = df.sample(200000)
'''
word_freqs = collections.Counter()
text = df.SentimentText.tolist()
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
for tweet in text:
    words = tknzr.tokenize(tweet.lower())
    for word in words:
        word_freqs[word] += 1
'''

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
tokenizer.fit_on_texts(half.SentimentText)
vocab = tokenizer.word_index
Y = half.Sentiment.values
Y = to_categorical(Y)
x_train, x_test, y_train, y_test = train_test_split(half.SentimentText, Y, test_size=0.2, random_state=2017)
x_train_word_ids = tokenizer.texts_to_sequences(x_train)
x_test_word_ids = tokenizer.texts_to_sequences(x_test)
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=40)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=40)

GLOVE_DIR = "D:\python\glove"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt'), encoding = 'utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(vocab) + 1, 200))
for word, i in vocab.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# TextCNNBN
main_input = Input(shape=(40,), dtype='float64')
embedder = Embedding(len(vocab) + 1, 200, input_length = 40, weights = [embedding_matrix], trainable = False)
embed = embedder(main_input)

conv1_1 = Convolution1D(256, 3, padding='same')(embed)
bn1_1 = BatchNormalization()(conv1_1)
relu1_1 = Activation('relu')(bn1_1)
conv1_2 = Convolution1D(128, 3, padding='same')(relu1_1)
bn1_2 = BatchNormalization()(conv1_2)
relu1_2 = Activation('relu')(bn1_2)
cnn1 = MaxPool1D(pool_size=4)(relu1_2)

conv2_1 = Convolution1D(256, 4, padding='same')(embed)
bn2_1 = BatchNormalization()(conv2_1)
relu2_1 = Activation('relu')(bn2_1)
conv2_2 = Convolution1D(128, 4, padding='same')(relu2_1)
bn2_2 = BatchNormalization()(conv2_2)
relu2_2 = Activation('relu')(bn2_2)
cnn2 = MaxPool1D(pool_size=4)(relu2_2)

conv3_1 = Convolution1D(256, 5, padding='same')(embed)
bn3_1 = BatchNormalization()(conv3_1)
relu3_1 = Activation('relu')(bn3_1)
conv3_2 = Convolution1D(128, 5, padding='same')(relu3_1)
bn3_2 = BatchNormalization()(conv3_2)
relu3_2 = Activation('relu')(bn3_2)
cnn3 = MaxPool1D(pool_size=4)(relu3_2)

cnn = concatenate([cnn1,cnn2,cnn3], axis=-1)
flat = Flatten()(cnn)
drop = Dropout(0.5)(flat)
fc = Dense(512)(drop)
bn = BatchNormalization()(fc)
main_output = Dense(2, activation='sigmoid')(drop)
model = Model(inputs = main_input, outputs = main_output)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train_padded_seqs, y_train,
                    batch_size=32,
                    epochs=5,
                    validation_data=(x_test_padded_seqs, y_test))

import matplotlib.pyplot as plt
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="g", label="Train")
plt.plot(history.history["val_acc"], color="b", label="Test")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="g", label="Train")
plt.plot(history.history["val_loss"], color="b", label="Test")
plt.legend(loc="best")

plt.tight_layout()
plt.show()
