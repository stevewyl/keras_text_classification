# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:13:37 2017

@author: stevewyl
"""
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, Input
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D
from keras.layers import  BatchNormalization
from sklearn.model_selection import train_test_split
from data_helper_3c import load_data_and_labels
import matplotlib.pyplot as plt

good_data_file = "./data/good_cut_jieba.txt"
bad_data_file = "./data/bad_cut_jieba.txt"
mid_data_file = "./data/mid_cut_jieba.txt"
x_text, y = load_data_and_labels(good_data_file, bad_data_file, mid_data_file)

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
tokenizer.fit_on_texts(x_text)
vocab = tokenizer.word_index

x_train, x_test, y_train, y_test = train_test_split(x_text, y, test_size=0.2, random_state=2017)

x_train_word_ids = tokenizer.texts_to_sequences(x_train)
x_test_word_ids = tokenizer.texts_to_sequences(x_test)
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=64)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=64)

#TextInception
main_input = Input(shape=(64,), dtype='float64')
embedder = Embedding(len(vocab) + 1, 256, input_length = 64)
embed = embedder(main_input)

block1 = Convolution1D(128, 1, padding='same')(embed)

conv2_1 = Convolution1D(256, 1, padding='same')(embed)
bn2_1 = BatchNormalization()(conv2_1)
relu2_1 = Activation('relu')(bn2_1)
block2 = Convolution1D(128, 3, padding='same')(relu2_1)

conv3_1 = Convolution1D(256, 3, padding='same')(embed)
bn3_1 = BatchNormalization()(conv3_1)
relu3_1 = Activation('relu')(bn3_1)
block3 = Convolution1D(128, 5, padding='same')(relu3_1)

block4 = Convolution1D(128, 3, padding='same')(embed)

inception = concatenate([block1, block2, block3, block4], axis=-1)

flat = Flatten()(inception)
fc = Dense(128)(flat)
drop = Dropout(0.5)(fc)
bn = BatchNormalization()(drop)
relu = Activation('relu')(bn)
main_output = Dense(3, activation='softmax')(relu)
model1 = Model(inputs = main_input, outputs = main_output)

model1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model1.fit(x_train_padded_seqs, y_train,
          batch_size=32,
          epochs=5,
          validation_data=(x_test_padded_seqs, y_test))

# MLP-onehot
x_train = tokenizer.sequences_to_matrix(x_train_word_ids, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test_word_ids, mode='binary')

model2 = Sequential()
model2.add(Dense(512, input_shape=(len(vocab)+1,), activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(3,activation='softmax'))

model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history2 = model2.fit(x_train, y_train,
                    batch_size=32,
                    epochs=5,
                    validation_data=(x_test, y_test))

# plot accuracy and loss
def plot_acc_loss(history):
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

plot_acc_loss(history)
plot_acc_loss(history2)