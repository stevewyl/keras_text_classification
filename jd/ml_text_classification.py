from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from data_helper_ml import load_data_and_labels

import numpy as np

categories = ['good', 'bad', 'mid']

x_text, y = load_data_and_labels("./data/good_cut_jieba.txt", "./data/bad_cut_jieba.txt", "./data/mid_cut_jieba.txt")
x_train, x_test, y_train, y_test = train_test_split(x_text, y, test_size=0.2, random_state=2017)

y = np.argmax(y, axis=-1)
y_train = np.argmax(y_train, axis=-1)
y_test = np.argmax(y_test, axis=-1)

print("Train/Test split: {:d}/{:d}".format(y_train.shape[0], y_test.shape[0]))

""" Naive Bayes classifier """
bayes_clf = Pipeline([('vect', CountVectorizer()),   # Convert a collection of text documents to a matrix of token counts
                      ('tfidf', TfidfTransformer()), # Transform a count matrix to a normalized tf or tf-idf representation
                      ('clf', MultinomialNB())       # Naive Bayes classifier for multinomial models
                      ])
bayes_clf.fit(x_train, y_train)
""" Predict the test dataset using Naive Bayes"""
predicted = bayes_clf.predict(x_test)
print('Naive Bayes correct prediction: {:4.4f}'.format(np.mean(predicted == y_test)))
print(metrics.classification_report(y_test, predicted, target_names=categories))

""" Support Vector Machine (SVM) classifier"""
svm_clf = Pipeline([('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter= 5, random_state=42)),
])
svm_clf.fit(x_train, y_train)
""" Predict the test dataset using Naive Bayes"""
predicted = svm_clf.predict(x_test)
print('SVM correct prediction: {:4.4f}'.format(np.mean(predicted == y_test)))
print(metrics.classification_report(y_test, predicted, target_names=categories))

print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, predicted))
print('\n')

""" 10-fold cross vaildation """
clf_b = make_pipeline(CountVectorizer(), TfidfTransformer(), MultinomialNB())
clf_s= make_pipeline(CountVectorizer(), TfidfTransformer(), SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))

bayes_10_fold = cross_val_score(clf_b, x_text, y, cv=10)
svm_10_fold = cross_val_score(clf_s, x_text, y, cv=10)

print('Naives Bayes 10-fold correct prediction: {:4.4f}'.format(np.mean(bayes_10_fold)))
print('SVM 10-fold correct prediction: {:4.4f}'.format(np.mean(svm_10_fold)))
