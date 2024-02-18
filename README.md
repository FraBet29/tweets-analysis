# Tweets Analysis

### About

Final project of the university optional course "Python per ingegneri matematici" ("Python for mathematical engineers"). The project aims to analyze tweets about a chosen topic - in this case, the Eurovision Song Contest 2021 - and experiment with different Python libraries.

### Dataset

The dataset (about a thousand tweets) has been obtained via the Twitter API, processed with pandas, and saved as a .csv file.

### Preprocessing and visual analysis

The tweets in the datasets have been preprocessed by removing URLs, converting the format from string to list of words, and removing punctuation. Next, two kinds of analysis have been performed:
- Statistical analysis: elements like hashtags, handles, and emojis have been located and their frequencies have been graphically illustrated by using the library Pygal;
- Semantical analysis: tweets words have been lemmatized by using the library NLTK and WordNet, a semantical-lexical database, then the most common roots have been visualized in a word cloud.

### Classification and sentiment analysis

Being the dataset unlabeled, a first attempt at classification has been performed via an unsupervised learning approach, namely, by using k-means clustering in scikit-learn. Better results have been obtained by using the built-in VADER classifier in the NLTK library, particularly suited for sentiment analysis of texts from social media. VADER assigns a score between -1 (negative) and 1 (positive) to each tweet; because of the high false positive rate, we set a custom threshold penalizing positive scores. The output of VADER has then been used for data annotation and the newly labeled dataset has been used to train a logistic regression model. The model achieved an accuracy of $84%$-$86%$ (the latter case with n-grams). 

### Deep learning with TensorFlow

The TensorFlow "Sentiment140" dataset (about a million tweets, only $10%$ used in the project) has been employed to build a vocabulary and train a deep learning model to perform sentiment analysis. The model includes an embedding layer, allowing to represent words in a vector space where closer words have similar meanings, plus two fully connected layers. The model has been trained for 5 epochs and has obtained a training accuracy of 88% and a validation accuracy of 75%. The model has then been applied to the original dataset to predict the sentiment of random tweets.
