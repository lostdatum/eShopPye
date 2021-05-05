# This file contains functions used in the data_viz_texte.ipynb file.

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import random
import math as m
import string
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer


def word_count(x):
    cd=pd.isnull(x)
    if cd==True:
        return np.nan
    else :
        return len(str(x).split()) 
    
def unique_words(x):
    if pd.isnull(x):
        return np.nan
    else :
        return len(set(str(x).split()))
def stop_words_count(x):
    stop_words = set(stopwords.words('french'))
    if pd.isnull(x):
        return np.nan
    else :
        return len([w for w in str(x).lower().split() if w in stop_words])
def chart_count(x):
    if pd.isnull(x):
        return np.nan
    else :
        return len(str(x))
def mean_word_length(x):
    if pd.isnull(x):
        return np.nan
    else :
        return len(set(str(x).split()))
    
def punctuation_count(x):
    if pd.isnull(x):
        return np.nan
    else :
        return len([c for c in str(x) if c in string.punctuation])
    
def remplace_accent(x):
    cd=pd.isnull(x)
    if cd==True:
        return x
    accent = ['é', 'è', 'ê', 'à', 'ù', 'û', 'ç', 'ô', 'î', 'ï', 'â','&acirc;','&agrave;','&eacute;','&ecirc;','&egrave;','&euml;','&icirc;','&iuml;','&ocirc;','&oelig;','&ucirc;','&ugrave;','&uuml;','&ccedil;','&lt;','&gt;','&szlig;','&oslash;','&Omega;','&ETH;','&Oslash;','&THORN;','&thorn;','&Aring;']
    sans_accent = ['e', 'e', 'e', 'a', 'u', 'u', 'c', 'o', 'i', 'i', 'a','a','a','e','e','e','e','i','i','o','oe','u','u','u','c',' ',' ','',' ',' ',' ',' ',' ',' ','A']

    for c, s in zip(accent, sans_accent):
        x = x.replace(c, s)
    return x



def cleaning_data(x):
    cd=pd.isnull(x)
    if cd==True:
        return x
    else:
        sentences = nltk.sent_tokenize(x)
        stemmer =  FrenchStemmer()

        # Lemmatization
        for i in range(len(sentences)):
            words = nltk.word_tokenize(sentences[i])
            words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('french'))]
            sentences[i] = ' '.join(words) 
        sentences=' '.join(sentences)
        # on retire les balises html
        sentences=re.sub('<[^<]+?>', '', sentences)  
        return sentences
    
def corpus(x,y):
    cx=pd.isnull(x)
    cy=pd.isnull(y)
    if cx==True:
        return y
    elif cy==True:
        return x
    else :
        return x+y 

def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(' ') if token != '' if token not in set(stopwords.words('french'))]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]
    
    
def plot_ngrams(dict_ngram,size=50, title="Les 50 unigrammes les plus présents dans les données textuelles", color="C3", figsize=(20,10)):
    keys = list(dict_ngram.keys())
    vals = [dict_ngram[k] for k in keys]
    ngrams=pd.DataFrame( columns=['ngrams', "count"])
    ngrams['ngrams']=keys
    ngrams['count']=vals
    ngrams = ngrams.sort_values(['count'], ascending=False).reset_index(drop=True)
    plt.figure(figsize=figsize)
    sns.barplot(y=ngrams['ngrams'][:size], x=ngrams['count'][:size],color = color)
    plt.legend(fontsize=15)
    plt.title(title, fontsize=20)
    plt.xlabel("Count", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
     