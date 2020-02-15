#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 13:57:26 2019

@author: poojasuthar
"""

import csv
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd

#Variables
ps = PorterStemmer()
lemm = WordNetLemmatizer()
corpus= []

#FEATURE_LIST_CSV_FILE_PATH = os.getcwd() + "feature_list.csv"
DATASET_FILE_PATH = os.getcwd() + "/dataset.csv"
stopwords = stopwords.words('english')

data_set = pd.read_csv(DATASET_FILE_PATH, header=None, encoding="utf-8", names=["Index", "Label", "Tweet"])

def clean_data(tweet, lemmatize=True, remove_punctuations=True, remove_stop_words=False, stemming=True):
    #tweet=re.sub(r"http\S+", "", tweet)
    tweet=re.sub(r"@\S+", "", tweet)
    #stopwords = nltk.corpus.stopwords.words('english')
    lemm = nltk.stem.wordnet.WordNetLemmatizer()
    tokens = nltk.word_tokenize(tweet)
    if remove_punctuations:
        tokens = [word for word in tokens if word not in string.punctuation]
    if remove_stop_words:
        tokens = [word for word in tokens if word.lower() not in stopwords]
    if lemmatize:
        tokens = [lemm.lemmatize(word) for word in tokens]
    if stemming:
        tokens = [ps.stem(word) for word in tokens]
    tokens= ' '.join(tokens)
    corpus.append(tokens)
    
    
tweets = list(data_set['Tweet'].values)   
for t in tweets:
    clean_data(t)