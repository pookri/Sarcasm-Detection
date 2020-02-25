import os
import nltk
import re
import string
import numpy as np
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
ps = PorterStemmer()
sid = SentimentIntensityAnalyzer()
DATASET_FILE_PATH = os.getcwd() + "/Dataset/dataset.csv"
emoji_sentiment = []
emoji_tweet_flip = []
def read_data(filename):
    data = pd.read_csv(filename, header=None, encoding="utf-8", names=["Index", "Label", "Tweet"])
    # data = data[data["Index"] > 76750]
    return data

def getEmojiSentiment(tweet, emoji_count_list=os.getcwd() + "/Dataset/Emoji_list.txt"):
    # Feature - Emoji [Compared with a list of Unicodes and common emoticons]
    emoji_sentiment = 0
    emoji_count_dict = dict(zip(emoji_count_list, np.zeros(len(emoji_count_list))))
    for e in emoji_count_dict.keys():
        if e in tweet:
            if e in emoji_count_list:
                emoji_count_dict.update({e: tweet.count(e)})
            emoji_sentiment += emoji_count_dict[e]*tweet.count(e)
    if sum(emoji_count_dict.values()) > 0:
        emoji_sentiment = (float(emoji_sentiment) / float(sum(emoji_count_dict.values())))
    return emoji_sentiment,emoji_count_dict

def getSentimentScore(tweet):
    return round(sid.polarity_scores(tweet)['compound'], 2)

sentimentscore = []
emoji_count_list = []
           
data_set = read_data(DATASET_FILE_PATH)
label = list(data_set['Label'].values)
tweets = list(data_set['Tweet'].values) 
y={}

for t in tweets:
    #x=user_mentions(i)  lineList = [line.rstrip('\n') for line in open(fileName)]
    #f.write(str(x)+"\n")
    #user_mention_count.append(user_mentions(i))
    x = getEmojiSentiment(t)
    #emoji_count_list.append(x[1])
    emoji_sentiment.append(x[0])
    sentimentscore.append(getSentimentScore(t))
    if (sentimentscore[-1] < 0 and emoji_sentiment[-1] > 0) or (sentimentscore[-1] > 0 and emoji_sentiment[-1] < 0):
            emoji_tweet_flip.append(1)
    else:
            emoji_tweet_flip.append(0)
f=open("Hashtags.txt",'w')
f.write(str(emoji_sentiment))
print(sentimentscore)
"""

def hashtag_sentiment(tweet):
    hash_tag = (re.findall("#([a-zA-Z0-9]{1,25})", tweet))
    f.write(hash_tag)

def user_mentions(tweet):
    return len(re.findall("@([a-zA-Z0-9]{1,20})", tweet))

def normalize( array):
    max = np.max(array)
    min = np.min(array)
    def normalize(x):
        return round(((x-min) / (max-min)),2)
    if max != 0:
        array = [x for x in map(normalize, array)]
    return array

user_mention_count=[]
#hashtag_sentiment(tweets)
f=open("User_Mentions.txt",'w')

print(normalize(user_mention_count) ) 
#f.write(str(user_mention_count))
#print("Labels",label)
#print("Tweets",tweets)


import emoji
import re

test_list=['🤔 🙈 me así,bla es,se 😌 ds 💕👭👙']

## Create the function to extract the emojis
def extract_emojis(a_list):
    emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
    r = re.compile('|'.join(re.escape(p) for p in emojis_list))
    aux=[' '.join(r.findall(s)) for s in a_list]
    return(aux)

## Execute the function
extract_emojis(test_list)

## the output
['🤔 🙈 😌 💕 👭 👙']

import emoji

def extract_emojis(str):
  return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

import emoji
import regex

def split_count(text):

    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)

    return emoji_list


line = ["🤔 🙈 me así, se 😌 ds 💕👭👙 hello 👩🏾‍🎓 emoji hello 👨‍👩‍👦‍👦 how are 😊 you today🙅🏽🙅🏽"]

counter = split_count(line[0])
print(' '.join(emoji for emoji in counter))

import emoji
emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
r = re.compile('|'.join(re.escape(p) for p in emojis_list))
print(' '.join(r.findall(s)))


from emoji import *

EMOJI_SET = set()

# populate EMOJI_DICT
def pop_emoji_dict():
    for emoji in UNICODE_EMOJI:
        EMOJI_SET.add(emoji)

# check if emoji
def is_emoji(s):
    for letter in s:
        if letter in EMOJI_SET:
            return True
    return False

def filter_emojis(sentence):
        return [word for word in sentence.split() if str(word.encode('unicode-escape'))[2] != '\\' ]
"""