import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))  # list of stopwords in english


def tokenize(inp_text):
    return nltk.word_tokenize(inp_text)


def remv_stop_words(tokenized_words):
    filtered_words = []
    for word in tokenized_words:
        if word.casefold() not in stop_words:
            filtered_words.append(word)
    return filtered_words


def stemming(f_w):
    stemmed_words = [stemmer.stem(word) for word in f_w]
    return stemmed_words


def b_o_w(text, s_w):
    bow = []
    for one_sen in nltk.sent_tokenize(text):
        vector = []
        for word in s_w:
            if word in nltk.word_tokenize(one_sen):
                vector.append(1)
            else:
                vector.append(0)
        bow.append(vector)
    return np.asarray(bow)


def nlp(inp_text):
    tokenized_words = tokenize(inp_text)  # list of tokens
    print(tokenized_words)
    # removing of stopwords from tokenized_words
    filtered_words = remv_stop_words(tokenized_words)
    print(filtered_words)
    # stemming the filtered_words
    stemmed_words = stemming(filtered_words)
    print(stemmed_words)
    # creating bag of words
    bag = b_o_w(inp_text, stemmed_words)

    return bag


