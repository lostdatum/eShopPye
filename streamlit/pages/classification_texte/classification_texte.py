# ================================== IMPORTS ==================================

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle
import re
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
import os
from PIL import Image

# ==================================== FUNC ===================================

@tf.function
def macro_soft_f1(y, y_hat):
    
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    # on multiplie la proba prédite d'une classe (y_hat) par son label => Uniquement les probas des vrais positifs seront non nuls
    tp = tf.reduce_sum(y_hat * y, axis=0) 
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    #  calcul du F1 score , 1e-16 pour ne pas diviser par 0
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    # comme on cherche a maximiser F1_score , et qu'il nous faut une fonction coût à minimiser on calcul le cout= 1 - soft-f1 
    cost = 1 - soft_f1 
    # on fait la moyenne pour tous les labels du batch
    macro_cost = tf.reduce_mean(cost) 
    return macro_cost


@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')
    

def remplace_accent(x):
    cd=pd.isnull(x)
    if cd==True:
        return x
    accent = ['é', 'è', 'ê', 'à', 'ù', 'û', 'ç', 'ô', 'î', 'ï', 'â','&acirc;','&agrave;','&eacute;','&ecirc;','&egrave;','&euml;','&icirc;','&iuml;','&ocirc;','&oelig;','&ucirc;','&ugrave;','&uuml;','&ccedil;','&lt;','&gt;','&szlig;','&oslash;','&Omega;','&ETH;','&Oslash;','&THORN;','&thorn;','&Aring;']
    sans_accent = ['e', 'e', 'e', 'a', 'u', 'u', 'c', 'o', 'i', 'i', 'a','a','a','e','e','e','e','i','i','o','oe','u','u','u','c',' ',' ','',' ',' ',' ',' ',' ',' ','A']
    for c, s in zip(accent, sans_accent):
        x = x.replace(c, s)
        return x


def preprocess_sentence(w):
    STOP_WORDS = set(stopwords.words('french'))
    w = unicode_to_ascii(w.lower().strip())
    w=remplace_accent(w)
    #removing html tags
    w=re.sub('<[^<]+?>', '', w)
    # creating a space between a word and the punctuation following it
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    #Remove words of length less than 1
    w=re.sub(r'\b\w{,1}\b', '', w)
     # remove stopword
    mots = word_tokenize(w.strip())
    stemmer =  FrenchStemmer()
    mots = [stemmer.stem(mot) for mot in mots if mot not in STOP_WORDS]
    return ' '.join(mots).strip()


@st.cache
def tokenize_text(text, script_dir):
    TOKENIZER_FILENAME = 'tokenizer.pickle'
    with open(scriptpath(script_dir, TOKENIZER_FILENAME), 'rb') as handle:
        tokenizer = pickle.load(handle)
    tokenized_text=tokenizer.texts_to_sequences([preprocess_sentence(text)])
    prepro_text=tf.keras.preprocessing.sequence.pad_sequences(tokenized_text, maxlen=17, padding='post')[0]
    prepro_text=prepro_text.reshape(1, 17)
    return prepro_text


@st.cache
def preprocess_text(text):
    return preprocess_sentence(text)


@st.cache 
def predict(text, script_dir):
    bi_lstm_200 = load_model(
        scriptpath(script_dir, "rnn.h5"),
        custom_objects={'macro_soft_f1': macro_soft_f1, "macro_f1":macro_f1}
        )
    y_pred=bi_lstm_200.predict(text)
    classe=np.argmax(y_pred,axis = 1)
    return classe


# Handle paths issues
def scriptpath(script_dir, relpath):
    return script_dir + os.path.normpath('/' + relpath)

# Page
def write():
    """Used to write the page in the app.py file"""
    
    
    # ==================================== INIT ===================================
        
    # Get this script's directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # ==================================== PAGE ===================================
    
    html_temp = """
    <div >
    <h1 style="color:rgb(191,0,0);text-align:center;">Classification à partir des données textes </h1>
    </div>
    """
    html_temp1 = """
    <div >
    <h2 style="color:black;font-weight: bold;">Word cloud des données</h2>
    </div>
    """
    html_temp2 = """
    <div >
    <h2 style="color:black;font-weight: bold;">Désignation de produit</h2>
    </div>
    """
    html_temp3 = """
    <div >
    <h2 style="color:black;font-weight: bold;">Prétraitement</h2>
    </div>
    """
    html_temp4 = """
    <div >
    <h2 style="color:black;font-weight: bold;">Classification</h2>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")
    st.markdown(html_temp1,unsafe_allow_html=True)
    st.text("")
    wc1 = st.empty()
    st.text("")
    wc2 = st.empty()
    
    st.markdown(html_temp2,unsafe_allow_html=True)
    designation = st.beta_container()
    st.markdown(html_temp3,unsafe_allow_html=True)
    prepro = st.beta_container()
    st.markdown(html_temp4,unsafe_allow_html=True)
    classif = st.beta_container()
    
    
    # ------------------------------- BACKEND -------------------------------------
    
    # Word cloud
    wc1.image(Image.open(scriptpath(SCRIPT_DIR, 'avant_prepro.png')), use_column_width=True, caption="Avant preprocessing")
    wc2.image(Image.open(scriptpath(SCRIPT_DIR, 'apres_prepro.png')), use_column_width=True, caption="Après preprocessing")

    # Entrer designation
    texte = designation.text_input("Entrez la désignation","Collez ou tapez votre texte")
    
    # Preprocessing
    prepro_text=preprocess_text(texte)
    prepro.subheader("Après préprocessing nous obtenons :")
    prepro.markdown(prepro_text)
    prepro.subheader("Après tokenisation est padding nous obtenons :")
    embedding=tokenize_text(texte, SCRIPT_DIR)
    prepro.write(embedding)

    # Classif
    classif.subheader("Architecture du meilleur modèle")
    path_modele=scriptpath(SCRIPT_DIR, 'archi_model.png')
    classif.image(Image.open(path_modele),  use_column_width=True,caption=["BiLSTM 100 units"])
    
    # Show subprocess and progress
    classe=""
    with st.spinner("Prédiction en cours..."):
        if st.button("Predire la classe du produit"):
            classe=predict(embedding, SCRIPT_DIR)
    st.success('La classe de ce produit est  {}'.format(classe))