# Tweets Analysis

### About

Final project of the university optional course "Python per ingegneri matematici" ("Python for mathematical engineers"). The aim of the project is to analyze tweets about a certain topic - in this case, the Eurovision Song Contest 2021 - and experimenting with different Python libraries.

### Dataset

The dataset (about a thousand tweets) has been obtained via the Twitter API, processed with pandas and saved as a .csv file.

### Preprocessing and visual analysis

The tweets in the datasets have been preprocessed by removing URLs, converting the format from string to list of words and removing punctuation. Next, two kind of analysis have been performed:
- Statistical analysis: elements like hashtags, handles, and emojis have been located and their frequencies have been graphically illustrated by using the library Pygal
- Semantical analysis: tweets words have been lemmatized by using the library NLTK and WordNet, a semantical-lexical database, then the most common roots have been represented in a wordcloud

### Classification and sentiment analysis

A first attempt of classification has been performed by using k-means clustering in scikit-learn (unsupervised learning). Sentiment analysis has been performed by using the built-in VADER classifier in the NLTK library, particularly suited for texts from social medias. VADER assigns to each tweet a score between -1 (negative) and 1 (positive); each tweet has been labeled as positive or negative according to its score and the new labeled datased has been used to perform logistic regression. The model has obtained an accuracy of 84%-86% (the latter case with n-grams). 

### Deep learning with TensorFlow

The TensorFlow "Sentiment140" dataset (about a million tweets, only 10% used in the project) has been employed to build a vocabulary and train a deep learning model to perform sentiment analysis. The model includes an embedding layer, allowing to represent words in a vector space where closer words have similar meanings, plus two fully connected layers. The model has been trained for 5 epochs and has obtained a training accuracy of 88% and a validation accuracy of 75%. The model has then been applied to the original dataset to predict the sentiment of random tweets.
