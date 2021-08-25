# -*- coding: utf-8 -*-
"""
Created on Sun May 23 12:23:53 2021

@author: frabe
"""
# %%

import pandas as pd
import nltk
import re
import string
import emojis # https://github.com/alexandrevicenzi/emojis
import pygal
from wordcloud import WordCloud # https://amueller.github.io/word_cloud/
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix

# %%

# Importazione dei dati
df = pd.read_csv('pinned_tweet_replies.csv')
pd.set_option('display.max_columns', None)
print(df.head())

# Lista contenente i testi dei tweet
tweets = df['Text']

# Rimozione degli url

def remove_url(txt):
    return ' '.join(re.sub(r'https:\S+', '', txt).split())

tweets = [remove_url(tweet.lower()) for tweet in tweets]

# Da stringhe a liste di parole
tknzr = nltk.TweetTokenizer() # Lascia integri handle e hashtag
tweets_words = [tknzr.tokenize(tweet) for tweet in tweets]

# Parole contenute in tutti i tweet
# Rimozione di stopwords e punteggiatura
sw = nltk.corpus.stopwords.words("english")
pun = string.punctuation
all_words = [item for tweet in tweets_words for item in tweet 
              if (item not in sw and item not in pun)]

# %%

# Ricerca e conteggio di hashtag
hashtags_freq = nltk.FreqDist([w for w in all_words if re.search('#[A-Za-z0-9_]+', w)])
print(hashtags_freq.most_common(10))

# Ricerca e conteggio di handle
handles_freq = nltk.FreqDist([w for w in all_words if re.search('@[A-Za-z0-9_]+', w)])
print(handles_freq.most_common(3))

# Ricerca e conteggio di emoji
emojis_list = [item for tweet in tweets for item in list(emojis.get(tweet))]
emojis_list = list(set(emojis_list)) # Eliminazione dei duplicati
emojis_freq = nltk.FreqDist([e for e in all_words if e in emojis_list])
print(emojis_freq.most_common(10))

# Identificazione e conteggio delle radici
WNlemma = nltk.WordNetLemmatizer()
all_roots = [WNlemma.lemmatize(w) for w in all_words if w.isalpha()]
all_roots_freq = nltk.FreqDist([r for r in all_roots])
print(all_roots_freq.most_common(20))

# %%

# Visualizzazioni

keys = [i[0] for i in hashtags_freq.most_common(10)]
frequencies = [i[1] for i in hashtags_freq.most_common(10)]
bar_chart = pygal.HorizontalBar()
bar_chart.title = 'Hashtag più frequenti'
bar_chart.add('frequenza', frequencies)
bar_chart.x_labels = keys

# bar_chart.render_to_file('hashtag_frequenti_bar.svg')

keys = [i[0] for i in emojis_freq.most_common(10)]
frequencies = [i[1] for i in emojis_freq.most_common(10)]
radar_chart = pygal.Radar()
radar_chart.title = 'Emoji più frequenti'
radar_chart.add('frequenza', frequencies)
radar_chart.x_labels = keys

# radar_chart.render_to_file('emoji_frequenti_radar.svg')

wordcloud = WordCloud()
wordcloud.generate(' '.join([w for w in all_roots]))
fig = plt.figure()
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('radici_comuni_wordcloud.png', dpi = 200)
plt.show()

# wordcloud_svg = wordcloud.to_svg()
# f = open("radici_comuni_wordcloud.svg", "w+")
# f.write(wordcloud_svg)
# f.close()

# %% 

# Rimozione hashtag e handle

def remove_hashtags(txt):
    return ' '.join(re.sub(r'#\S+', '', txt).split())

tweets = [remove_hashtags(tweet) for tweet in tweets]

def remove_handles(txt):
    return ' '.join(re.sub(r'@\S+', '', txt).split())

tweets = [remove_handles(tweet) for tweet in tweets]

#%%

# Clustering

# Trasformazione del testo in una matrice (sparsa)
vectorizer = TfidfVectorizer(stop_words = {'english'}, strip_accents = 'unicode')
X = vectorizer.fit_transform(tweets)

# Elbow method per la scelta del numero di cluster
sum_of_squared_distances = []
K = range(2,20)
for k in K:
    km = KMeans(n_clusters = k, max_iter = 200, n_init = 10)
    km = km.fit(X)
    # inertia = sum of squared distance for each point to its closest centroid
    sum_of_squared_distances.append(km.inertia_)

fig = plt.figure()
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('sum_of_squared_distances')
# plt.savefig('elbow_method.png', dpi = 200)
plt.show()

# Non c'è un cambiamento significativo della pendenza

# %%

# Analisi del sentiment

# NLTK has a built-in, pretrained sentiment analyzer called VADER

# Hutto, C.J. & Gilbert, E.E. (2014).
# VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
# Eighth International Conference on Weblogs and Social Media (ICWSM-14).
# Ann Arbor, MI, June 2014.

# https://github.com/cjhutto/vaderSentiment#about-the-scoring
# The compound score is computed by summing the valence scores of each word in the lexicon, 
# adjusted according to the rules, and then normalized to be between -1 
# (most extreme negative) and +1 (most extreme positive).
# Typical threshold values:
# Positive sentiment: compound score >= 0.05
# Neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
# Negative sentiment: compound score <= -0.05

sia = SentimentIntensityAnalyzer()

positive_tweets = []
negative_tweets = []

# Alcuni tweet vengono classificati erroneamente (positivi invece di negativi)
# Threshold più alta per i tweet positivi

for tweet in tweets:
    scores = sia.polarity_scores(tweet)
    if scores['compound'] >= 0.5:
        positive_tweets.append(tweet)
    if scores['compound'] <= -0.25:
        negative_tweets.append(tweet)

positive_words = [tknzr.tokenize(tweet) for tweet in positive_tweets]
all_positive_words = [w for pw in positive_words for w in pw if w.isalpha() 
                      and w not in sw and w != 'italy' and w != 'song'and w != 'eurovision']
positive_words_freq = nltk.FreqDist([w for w in all_positive_words])
print(positive_words_freq.most_common(20))

negative_words = [tknzr.tokenize(tweet) for tweet in negative_tweets]
all_negative_words = [w for pw in negative_words for w in pw if w.isalpha() 
                      and w not in sw and w != 'italy' and w != 'song' and w != 'eurovision']
negative_words_freq = nltk.FreqDist([w for w in all_negative_words])
print(negative_words_freq.most_common(20))

# %%

wordcloud.generate(' '.join([w for w in all_positive_words]))
fig = plt.figure()
plt.imshow(wordcloud)
plt.axis('off')
# plt.savefig('positive_wordcloud.png', dpi = 200)
plt.show()

wordcloud.generate(' '.join([w for w in all_negative_words]))
fig = plt.figure()
plt.imshow(wordcloud)
plt.axis('off')
# plt.savefig('negative_wordcloud.png', dpi = 200)
plt.show()

# %%

# Classificazione

labeled_tweets = []

for tweet in positive_tweets:
    labeled_tweets.append([tweet, 1])
for tweet in negative_tweets:
    labeled_tweets.append([tweet, 0])

dati = pd.DataFrame(labeled_tweets, columns = ['Text', 'Label'])

# Divisione del dataset per training e testing
X_train, X_test, y_train, y_test = train_test_split(dati['Text'], 
                                                    dati['Label'], 
                                                    random_state = 0)

# Bag of words
# Conversione del testo in una matrice di conteggi dei token
# Si considerano bigrammi e trigrammi (es. negazioni)
vectorizer = CountVectorizer(ngram_range = (1,3)).fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Training
LogReg = LogisticRegression(max_iter = 10000)
x = LogReg.fit(X_train_vectorized, y_train)

# predictions = LogReg.predict(X_test_vectorized)

# Accuratezza = previsioni corrette / numero dei dati
score = LogReg.score(X_test_vectorized, y_test)
print(score)

# Senza n-grams: 0.8486842105263158
# Con n-grams: 0.8618421052631579

# Confusion matrix
fig = plt.figure()
plot_confusion_matrix(LogReg, X_test_vectorized, y_test)
# plt.savefig('confusion_matrix.png', dpi = 200)
plt.show()
