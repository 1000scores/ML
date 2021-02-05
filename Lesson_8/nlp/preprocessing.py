# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import pandas as pd
import numpy as np
from sklearn import *
from collections import Counter
import nltk
import string
import re
from nltk import *
from sklearn.preprocessing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import *
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from nltk.corpus import stopwords
import json
from os import listdir
import pickle


def preprocess(path_train='Lesson_8/nlp/train.csv', path_test='Lesson_8/nlp/test.csv'):
    print(listdir())
    df_train = pd.read_csv(path_train, encoding='ISO-8859-1')
    df_test = pd.read_csv(path_test, encoding='ISO-8859-1')
    df = pd.concat([df_train, df_test])
    df.columns = df.columns.str.lower()
    df.set_index('id', inplace=True)
    features = ['hotel_name', 'review_title', 'review_text']
    df[features] = df[features].apply(lambda x: x.str.lower())
    df[features] = df[features].apply(lambda x: x.str.replace('[\W]+', ' '))
    df[features] = df[features].fillna('')
    nltk.download('stopwords')
    sw_eng = set(stopwords.words('english'))

    df[features] = df[features].apply(lambda x: x.apply(lambda y: ' '.join([word for word in y.split(' ') if not(word in sw_eng)])))

    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    def get_wordnet_pos(treebank_tag):
        my_switch = {
            'J': wordnet.wordnet.ADJ,
            'V': wordnet.wordnet.VERB,
            'N': wordnet.wordnet.NOUN,
            'R': wordnet.wordnet.ADV,
        }
        for key, item in my_switch.items():
            if treebank_tag.startswith(key):
                return item
        return wordnet.wordnet.NOUN

    def my_lemmatizer(sent):
        lemmatizer = WordNetLemmatizer()
        tokenized_sent = sent.split()
        pos_tagged = [(word, get_wordnet_pos(tag))
                    for word, tag in pos_tag(tokenized_sent)]
        return ' '.join([lemmatizer.lemmatize(word, tag)
                        for word, tag in pos_tagged])

    df[features] = df[features].apply(lambda x: x.apply(lambda y: ' '.join([my_lemmatizer(word) for word in y.split(' ')])))


    all_text_series = df.hotel_name + df.review_title + df.review_text

    tfidf = TfidfVectorizer()
    tfidf.fit(all_text_series)

    answer = tfidf.transform(all_text_series).todense()

    return (answer[:df_train.shape[0]], df['rating'][:df_train.shape[0]], answer[df_train.shape[0]:])

if __name__ == '__main__':
    print(preprocess())