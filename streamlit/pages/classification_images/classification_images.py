# ================================ IMPORTS ====================================

# Srtreamlit
import streamlit as st

# Essential
import numpy as np
import pandas as pd

# System
import os
import time

# Image
from PIL import Image
import cv2

# Deep learning
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import vgg16


# Page
def write():
    """Used to write the page in the app.py file"""
    
    # =============================== CONSTANTS ===================================
    
    LABELS = range(27)
    
    # ================================= PATHS =====================================
    
    # Get this script's directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Set paths
    PATH_DECODE = SCRIPT_DIR + os.path.normpath("/data/csv/class_index_decoding.csv")
    PATH_PRODUCT = SCRIPT_DIR + os.path.normpath("/data/csv/prdtype_decode_enhanced.csv")
    PATH_DUMMY = SCRIPT_DIR + os.path.normpath("/assets/images/blank.png")
    PATH_TABLE = SCRIPT_DIR + os.path.normpath("/assets/images/table_model_2.png")
    PATH_SUMMARY = SCRIPT_DIR + os.path.normpath("/assets/images/vgg16_summary.png")
    PATH_MODEL = SCRIPT_DIR + os.path.normpath("/data/models/vgg16_fullset_v2.h5")
    
    
    # ================================ CODE =======================================
    
    st.title("Classification des images")
    st.header("Choix de l'image")
    # Create file uploader
    img_io = st.file_uploader( # returns UploadedFile (subclass of BytesIO)
                label = "Choisissez une image à tester:",
                type = ['png', 'jpg', 'jpeg'],
                accept_multiple_files = False
                )
    # Create image slot
    slot_img_raw = st.empty()
    # img = cv2.imread(r"C:\Users\basti\ANACONDA\eShopPye\bastien\streamlit\pages\classification_images\assets\images\test_img.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imgslot1 = st.image(img/255, caption="Image sélectionnée")
    # Show selected image
    if img_io:
        img = np.array(Image.open(img_io), dtype='float')
        slot_img_raw.image(img/255, caption="Image sélectionnée")
    
    st.header("Prétraitement")
    # Partitioning
    prepro = st.beta_container()
    col1, col2 = prepro.beta_columns(2)
    # Image (filled with placeholder at first)
    imgslot = col1.image(Image.open(PATH_DUMMY))
    # Description
    col2.text(
        "Avant de pouvoir être traitée par notre\n"
        "modèle, l'image doit être prétraitée.\n"
        "Lors du prétraitement, l'image est d'abord\n"
        "redimensionnée sur 256 x 256 pixels, puis\n"
        "traitée par la fonction preprocess_input()\n"
        "du module vgg16 de Tensorflow.\n"
        "Cette fonction inverse l'ordre des canaux\n"
        "de couleur (RGB vers BGR), puis recentre\n"
        "chaque pixel de chaque canal de couleur\n"
        "sur la moyenne du jeu d'images ImageNet."
        )
    # Preprocess
    if img_io:
        img_resz = cv2.resize(img, (256, 256))
        img_prepro = vgg16.preprocess_input(img_resz)
        imgslot.image(np.clip(img_prepro/255, 0, 1), caption="Image prétraitée")
    
    
    st.header("Modèle utilisé")
    model = load_model(PATH_MODEL)
    # Introduce
    st.text(
        "Le modèle de classification est un réseau de neurones convolutif (CNN).\n"
        "Les couches convolutives ont été extraites d'un modèle VGG16 pré-entraîné "
        "sur ImageNet.\n"
        "Ces couches convolutives de base sont restées figées pendant l'entraînement.\n"
        "A cette base, nous avons ajouté notre propre couche dense prédictive, "
        "précédée de deux\ncouches denses cachées, que nous avons entrainées.\n"
        "Les caractéristiques détaillées de ce modèle sont résumées ci-dessous.\n"
        )
    # Partitioning
    showmodel = st.beta_container()
    col3, col4 = showmodel.beta_columns(2)
    # Show images
    col3.image(Image.open(PATH_SUMMARY), caption="Architecture du modèle")
    col4.image(Image.open(PATH_TABLE), caption="Caractéristiques du modèle")
    
    
    st.header("Classification")
    # Partitioning
    classif = st.beta_container()
    col5, col6 = classif.beta_columns(2)
    # Description
    col5.text(
        "Pour lancer la classification de\n"
        "l'image sélectionnée, veuillez\n"
        "cliquer sur le bouton ci-contre."
        )
    # Add button
    gopredict = col6.button("Prédire")
    # Prediction
    if gopredict:
        with st.spinner("Classification en cours..."):
            # Predict probabilities
            y_probs = model.predict(np.reshape(img_prepro, (1, 256, 256, 3))).squeeze()
            # Decode class indexes into labels
            decoding = pd.read_csv(PATH_DECODE, index_col='label').sort_index().squeeze()
            # Reorder probabilities (according to labels)
            y_probs = [y_probs[decoding[label]] for label in LABELS]
            # Decision
            y_pred = np.argmax(y_probs)
            # Find product definition
            products = pd.read_csv(PATH_PRODUCT, index_col='label')['prdtype'].squeeze()
            # Display
            st.success(f"Classe prédite : [{y_pred}] '{products[y_pred]}'")
            # Emulate heavy computing (for user experience!)
            time.sleep(0.2)