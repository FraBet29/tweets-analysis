# -*- coding: utf-8 -*-
"""
Created on Sun May 23 12:30:59 2021

@author: frabe
"""

import pandas as pd
import tweepy
from twitter_keys import APIKey, APISecretKey

auth = tweepy.OAuthHandler(APIKey, APISecretKey)
api = tweepy.API(auth)

# Pinned Tweet
# https://twitter.com/Eurovision/status/1396236320092135425

# https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/tweet
# https://docs.tweepy.org/en/latest/extended_tweets.html

name = 'eurovision'
tweet_id = '1396236320092135425'

pinned_tweet_replies = []

for tweet in tweepy.Cursor(api.search, 
                           q = 'to:'+name, 
                           lang = 'en', 
                           count = 100,  
                           tweet_mode = 'extended').items():
    if hasattr(tweet, 'in_reply_to_status_id_str'):
        if (tweet.in_reply_to_status_id_str == tweet_id):
            pinned_tweet_replies.append([tweet.id_str, tweet.full_text, 
                                  tweet.user.id_str, tweet.user.screen_name])


header = ['ID', 'Text', 'User ID', 'Username']

df = pd.DataFrame.from_records(pinned_tweet_replies, columns = header)

# df.to_csv("pinned_tweet_replies.csv")






