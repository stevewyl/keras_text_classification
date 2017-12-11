# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 01:00:31 2017

@author: stevewyl
"""
import pandas as pd
import numpy as np
import os
from keras import backend
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.layers import  BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras import constraints
from keras import regularizers
from keras.engine.topology import Layer

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

df = pd.read_csv('./data/ign.csv')
df = df[df.score_phrase != 'Disaster']
title = df.title
label = df.score_phrase
X_train, X_test, y_train, y_test = train_test_split(title, label, test_size=0.1, random_state=42)

# MultinomialNB Classifier
vect = TfidfVectorizer(stop_words='english', 
                       token_pattern=r'\b\w{2,}\b',
                       min_df=1, max_df=0.1,
                       ngram_range=(1,2))
mnb = MultinomialNB(alpha=2)
svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42)
mnb_pipeline = make_pipeline(vect, mnb)
svm_pipeline = make_pipeline(vect, svm)
mnb_cv = cross_val_score(mnb_pipeline, title, label, scoring='accuracy', cv=10, n_jobs=-1)
svm_cv = cross_val_score(svm_pipeline, title, label, scoring='accuracy', cv=10, n_jobs=-1)
print('\nMultinomialNB Classifier\'s Accuracy: %0.5f\n' % mnb_cv.mean())
# 0.28284
print('\nSVM Classifier\'s Accuracy: %0.5f\n' % svm_cv.mean())
# 0.27684

y_labels = list(y_train.value_counts().index)
le = preprocessing.LabelEncoder()
le.fit(y_labels)
num_labels = len(y_labels)
y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
y_test = to_categorical(y_test.map(lambda x: le.transform([x])[0]), num_labels)

# load glove word embedding data
GLOVE_DIR = "D:\python\glove"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding = 'utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
 
# take tokens and build word-id dictionary     
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
tokenizer.fit_on_texts(title)
vocab = tokenizer.word_index

# Match the word vector for each word in the data set from Glove
embedding_matrix = np.zeros((len(vocab) + 1, 300))
for word, i in vocab.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Match the input format of the model
x_train_word_ids = tokenizer.texts_to_sequences(X_train)
x_test_word_ids = tokenizer.texts_to_sequences(X_test)
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=20)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=20)


# one-hot mlp
x_train = tokenizer.sequences_to_matrix(x_train_word_ids, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test_word_ids, mode='binary')

model = Sequential()
model.add(Dense(512, input_shape=(len(vocab)+1,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=32,
          epochs=15,
          validation_data=(x_test, y_test))
# 0.4557

# RNN model
model = Sequential()
model.add(Embedding(len(vocab)+1, 256, input_length=20))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.1))
model.add(Dense(num_labels, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train_padded_seqs, y_train,
          batch_size=32,
          epochs=12,
          validation_data=(x_test_padded_seqs, y_test))

# Average training time between 280-300 seconds
# GRU-15-256-256-0.1-0.1     0.4262
# GRU-18-256-256-0.15-       0.4364
# GRU-18-256-256-0.2-        0.4305
# GRU-18-256-256-0.2-0.1     0.4359(*best)
# GRU-18-256-256-0.15-0.1    0.4283
# GRU-20-256-256-0.1-0.1     0.4224
# GRU-20-256-256-0.2-0.2     0.4214
# GRU-18-256-256-0.2-0.2     0.4278
# LSTM-18-256-256-0.2-0.2    0.43
# LSTM-18-256-256-0.3-0.2    0.4198
# LSTM-18-256-256-0.2-0.1    0.4235
# epoch 15                   0.4439
# BI                         0.4482
score, acc = model.evaluate(x_test_padded_seqs, y_test,
                            batch_size=32 )

# CNN model
model = Sequential()
model.add(Embedding(len(vocab)+1, 256, input_length=20))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(256, 3, padding='same'))
model.add(MaxPool1D(3,3,padding='same'))
model.add(Convolution1D(128, 3, padding='same'))
model.add(MaxPool1D(3,3,padding='same'))
model.add(Convolution1D(64, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(num_labels,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train_padded_seqs, y_train,
          batch_size=32,
          epochs=12,
          validation_data=(x_test_padded_seqs, y_test))

# Average training time between 40-55 seconds
# 32-15-conv*3(128-64-32)-3-0.2    0.4069(same)
# 32-20-conv*3(128-64-32)-3-0.2    0.4133(same)
# 32-20-conv*3(128-64-32)-3-0.1    0.4165(same)
# 32-20-conv*3(128-64-32)-3-0.15   0.4149(same)
# 64-20-conv*3(128-64-32)-3-0.1    0.4042(same)
# 32-20-conv*3(256-128-64)-3-0.1   0.4219(same)
# 32-20-conv*3(256-128-64)-4-0.1   0.4144(same)
# 32-20-conv*3(256-128-64)-3-0.1   0.4069(valid)
# 32-20-conv*3(256-128-64)-3-0.1   0.4144(same)
# add max-pooling                  0.4257(same)(*best)
# epoch 15                         0.4359(*best)

# CNN+GRU
model = Sequential()
model.add(Embedding(len(vocab)+1, 256, input_length=20))
model.add(Convolution1D(256, 3, padding='same', strides = 1))
model.add(Activation('relu'))
model.add(MaxPool1D(pool_size=2))
model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1, return_sequences = True))
model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1))
model.add(Dense(num_labels, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train_padded_seqs, y_train,
          batch_size=32,
          epochs=12,
          validation_data=(x_test_padded_seqs, y_test))
# 128
# GRU+CONV              0.4412
# GRU*2+CONV*2          0.4359
# GRU*2+CONV*1          0.4337 
# GRU*1+CONV*2          0.4305
# 256
# GRU*1+CONV*1          0.4418
# GRU*1+CONV*2          0.4289
# GRU*2+CONV*1          0.4423

# TextCNN
main_input = Input(shape=(20,), dtype='float64')
embedder = Embedding(len(vocab) + 1, 300, input_length = 20)
embed = embedder(main_input)
cnn1 = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(embed)
cnn1 = MaxPool1D(pool_size=4)(cnn1)
cnn2 = Convolution1D(256, 4, padding='same', strides = 1, activation='relu')(embed)
cnn2 = MaxPool1D(pool_size=4)(cnn2)
cnn3 = Convolution1D(256, 5, padding='same', strides = 1, activation='relu')(embed)
cnn3 = MaxPool1D(pool_size=4)(cnn3)
cnn = concatenate([cnn1,cnn2,cnn3], axis=-1)
flat = Flatten()(cnn)
drop = Dropout(0.2)(flat)
main_output = Dense(num_labels, activation='softmax')(drop)
model = Model(inputs = main_input, outputs = main_output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train_padded_seqs, y_train,
          batch_size=32,
          epochs=12,
          validation_data=(x_test_padded_seqs, y_test))

# conv*3(3,4,5)            0.4546
# with embedding 100d      0.4326
# with embedding 200d      0.4283
# with embedding 200d      0.4332

# TextCNN with GRU
main_input = Input(shape=(20,), dtype='float64')
embed = Embedding(len(vocab)+1, 256, input_length=20)(main_input)
cnn1 = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(embed)
cnn1 = MaxPool1D(pool_size=4)(cnn1)
cnn2 = Convolution1D(256, 4, padding='same', strides = 1, activation='relu')(embed)
cnn2 = MaxPool1D(pool_size=4)(cnn2)
cnn3 = Convolution1D(256, 5, padding='same', strides = 1, activation='relu')(embed)
cnn3 = MaxPool1D(pool_size=4)(cnn3)
cnn = concatenate([cnn1,cnn2,cnn3], axis=-1)
gru = Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1))(cnn)
main_output = Dense(num_labels, activation='softmax')(gru)
model = Model(inputs = main_input, outputs = main_output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train_padded_seqs, y_train,
          batch_size=32,
          epochs=12,
          validation_data=(x_test_padded_seqs, y_test))

# BI-lstm
# 0.4471

# CNN+LSTM concat
main_input = Input(shape=(20,), dtype='float64')
embed = Embedding(len(vocab)+1, 256, input_length=20)(main_input)
cnn = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(embed)
cnn = MaxPool1D(pool_size=4)(cnn)
cnn = Flatten()(cnn)
cnn = Dense(256)(cnn)
rnn = Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1))(embed)
rnn = Dense(256)(rnn)
con = concatenate([cnn,rnn], axis=-1)
main_output = Dense(num_labels, activation='softmax')(con)
model = Model(inputs = main_input, outputs = main_output)

# 0.4434

# C-LSTM
main_input = Input(shape=(20,), dtype='float64')
embed = Embedding(len(vocab)+1, 256, input_length=20)(main_input)
cnn = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(embed)
new = Reshape(target_shape=(cnn.shape[2].value, cnn.shape[1].value))(cnn)

# CNN-char
# Reprocess the input
# get vocab
all_sent = []
for sent in title.tolist():
    new = []
    for word in sent:
        for char in word:
            new.append(word)
        new_sent = " ".join(new)
    all_sent.append(new_sent)
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
tokenizer.fit_on_texts(all_sent)
vocab = tokenizer.word_index
X_train, X_test, y_train, y_test = train_test_split(all_sent, label, test_size=0.1, random_state=42)
X_train_word_ids = tokenizer.texts_to_sequences(X_train)
X_test_word_ids = tokenizer.texts_to_sequences(X_test)
X_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=30)
X_test_padded_seqs = pad_sequences(X_test_word_ids, maxlen=30)

# 0.4063

# DCNN
def KMaxPooling1D():
    pass

def Folding():
    pass

model = Sequential()
model.add(Embedding(len(vocab)+1, 256, input_length=20))
model.add(Convolution1D())
model.add(Folding())
model.add(KMaxPooling1D())
model.add(Activation('tanh'))
model.add()

# GRU with Attention

# Aspect-level attention 

# Hierarchical Model with Attention
class AttLayer(Layer):
    def __init__(self, init='glorot_uniform', kernel_regularizer=None, 
                 bias_regularizer=None, kernel_constraint=None, 
                 bias_constraint=None,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get(init)
        
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(kernel_regularizer)
        
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        super(AttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)
        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        
        self.built = True
        
    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W) # (x, 40, 1)
        uit = K.squeeze(uit, -1) # (x, 40)
        uit = uit + self.b # (x, 40) + (40,)
        uit = K.tanh(uit) # (x, 40)

        ait = uit * self.u # (x, 40) * (40, 1) => (x, 1)
        ait = K.exp(ait) # (X, 1)

        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(x, 40)
            ait = mask*ait #(x, 40) * (x, 40, )

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

inputs = Input(shape=(20,), dtype='float64')
embed = Embedding(len(vocab) + 1,300, input_length = 20)(inputs)
gru = Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(embed)
attention = AttLayer()(gru)
output = Dense(num_labels, activation='softmax')(attention)
model = Model(inputs, output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train_padded_seqs, y_train,
          batch_size=32,
          epochs=12,
          validation_data=(x_test_padded_seqs, y_test))
# 0.4487

# fastetxt model
# Generates the n-gram combination vocabulary for the input text
n_value = 2
def create_ngram_set(input_list, ngram_value=n_value):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))
# Add the new n-gram generated words into the original sentence sequence
def add_ngram(sequences, token_indice, ngram_range=n_value):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

ngram_set = set()
for input_list in x_train_padded_seqs:
    for i in range(2, n_value + 1):
        set_of_ngram = create_ngram_set(input_list, ngram_value=i)
        ngram_set.update(set_of_ngram)
start_index = len(vocab) + 2
token_indice = {v: k + start_index for k, v in enumerate(ngram_set)} # 给bigram词汇编码
indice_token = {token_indice[k]: k for k in token_indice}
max_features = np.max(list(indice_token.keys())) + 1
x_train = add_ngram(x_train_word_ids, token_indice, 3)
x_test = add_ngram(x_test_word_ids, token_indice, 3)
x_train = pad_sequences(x_train, maxlen=64)
x_test = pad_sequences(x_test, maxlen=64)

model = Sequential()
model.add(Embedding(max_features,256,input_length=64))
model.add(GlobalAveragePooling1D())
model.add(Dense(num_labels, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=32,
          epochs=5,
          validation_data=(x_test, y_test))  
# 2-gram 0.4648
# 3-gram 0.4546

# TextRCNN
# RCNN for paper http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745 
left_train_word_ids = [[len(vocab)] + x[:-1] for x in X_train_word_ids]
left_test_word_ids = [[len(vocab)] + x[:-1] for x in X_test_word_ids]
right_train_word_ids = [x[1:] + [len(vocab)] for x in X_train_word_ids]
right_test_word_ids = [x[1:] + [len(vocab)] for x in X_test_word_ids]
left_train_padded_seqs = pad_sequences(left_train_word_ids, maxlen=20)
left_test_padded_seqs = pad_sequences(left_test_word_ids, maxlen=20)
right_train_padded_seqs = pad_sequences(right_train_word_ids, maxlen=20)
right_test_padded_seqs = pad_sequences(right_test_word_ids, maxlen=20)

document = Input(shape = (None, ), dtype = "int32")
left_context = Input(shape = (None, ), dtype = "int32")
right_context = Input(shape = (None, ), dtype = "int32")

embedder = Embedding(len(vocab) + 1, 300, input_length = 20)
doc_embedding = embedder(document)
l_embedding = embedder(left_context)
r_embedding = embedder(right_context)
forward = LSTM(256, return_sequences = True)(l_embedding) # See equation (1)
backward = LSTM(256, return_sequences = True, go_backwards = True)(r_embedding) # See equation (2)
together = concatenate([forward, doc_embedding, backward], axis = 2) # See equation (3)
semantic = TimeDistributed(Dense(128, activation = "tanh"))(together) # See equation (4)
pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (128, ))(semantic) # See equation (5)
output = Dense(10, activation = "softmax")(pool_rnn) # See equations (6) and (7)
model = Model(inputs = [document, left_context, right_context], outputs = output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit([x_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs], y_train,
          batch_size=32,
          epochs=12,
          validation_data=([x_test_padded_seqs, left_test_padded_seqs, right_test_padded_seqs], y_test))

# 0.4439
# 0.4240
# 0.4498

'''
# next step cv for finding better models
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2017)
cvscores = []
Y = np.concatenate((y_train,y_test), axis=0)
X = pd.concat([X_train_padded_seqs, X_test_padded_seqs])
for train, test in kfold.split(X, Y):
    # create model
 	model = best
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
    model.fit(X[train], Y[train], epochs=10, batch_size=32, verbose=0)
	# evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
'''
