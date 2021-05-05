import streamlit as st


def write():
    """Used to write the page in the app.py file"""

    st.title("Présentation du projet")
    st.text(
		"Le projet eShopPye consiste à classifier des produits de e-commerce\n"
        "à partir du texte et d'une image associés.\n"
        "Réalisé dans le cadre du challenge de data science\n"
		"Rakuten France Multimodal Product Data Classification\n"
		"pendant la formation Data Scientist de DataScientest.\n"
	)