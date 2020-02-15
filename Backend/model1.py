#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:29:50 2020

@author: poojasuthar

"""

import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sid
import constantly
import pandas as pd

#Variables
ps = PorterStemmer()
lemm = WordNetLemmatizer()
corpus= []


DATASET_FILE_PATH = os.getcwd() + "/Dataset/dataset.csv"
stopwords = stopwords.words('english')

def read_data(filename):
    data = pd.read_csv(filename, header=None, encoding="utf-8", names=["Index", "Label", "Tweet"])
    return data


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
    return corpus
    
def user_mentions(tweet):
    return len(re.findall("@([a-zA-Z0-9]{1,15})", tweet))


def punctuations_counter(tweet, punctuation_list):
    punctuation_count = {}
    for p in punctuation_list:
        punctuation_count.update({p: tweet.count(p)})
    return punctuation_count

def interjections_counter(tweet):
    interjection_count = 0
    for interj in constantly.interjections:
        interjection_count += tweet.lower().count(interj)
    return interjection_count

def intensifier_counter(tokens):
    
    posC, negC = 0, 0
    for index in range(len(tokens)):
        if tokens[index] in constantly.intensifier_list:
            if (index < len(tokens) - 1):
                ss_in = sid.polarity_scores(tokens[index + 1])
                if (ss_in["neg"] == 1.0):
                    negC += 1
                if (ss_in["pos"] == 1.0):
                    posC += 1
    return posC, negC

def repeatLetterWords_counter(tweet):
    repeat_letter_words = 0
    matcher = re.compile(r'(.)\1*')
    repeat_letters = [match.group() for match in matcher.finditer(tweet)]
    for segments in repeat_letters:
        if len(segments) >= 3 and str(segments).isalpha():
            repeat_letter_words += 1
    return repeat_letter_words

def getSentimentScore(tweet):
    return round(sid.polarity_scores(tweet)['compound'], 2)



def main():    
    
    data_set = read_data(DATASET_FILE_PATH)
    label = list(data_set['Label'].values)    
    tweets = list(data_set['Tweet'].values)
    user_mention_count = []
    exclamation_count = []
    questionmark_count = []
    ellipsis_count = []
    print("xxxxxx")
    
    for t in tweets:
        tokens = clean_data(t);
        user_mention_count.append(user_mentions(t))
        p = punctuations_counter(t, ['!', '?', '...'])
        exclamation_count.append(p['!'])
        questionmark_count.append(p['?'])
        ellipsis_count.append(p['...'])
        
if __name__ == "__main__":
    main()