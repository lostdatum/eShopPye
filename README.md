Projet eShopPye  
===============

Contexte  
---------

Ce projet a été réalisé dans le cadre de la [formation Data Scientist de Datascientest](https://datascientest.com/formation-data-scientist) (promotion Bootcamp Décembre 2020) et du challenge [Rakuten France Multimodal Product Data Classification](https://challengedata.ens.fr/participants/challenges/35/).  


Objectif  
--------

Le problème consiste, à partir des informations textuelles et de l’image associées à chaque article du catalogue, à classifier automatiquement chacun d’eux dans l’une des catégories issues de la taxonomie produit de *Rakuten*, en commettant le moins d’erreurs possible.  

En situation de production, la taxonomie *Rakuten* comprend plus de 1000 catégories, mais dans le cadre de ce challenge, on se limite à seulement 27 d’entre elles.  


Jeux de données  
---------------

Les données utilisées sont la propriété de *Rakuten Institute of Technology* et sont fournies au participants lors de l’inscription au challenge.  

Le jeu de données disponible pour l’entraînement (et la validation) est constitué de 84916 observations.  

Chaque observation représente un article du catalogue de la plateforme de e-commerce de *Rakuten France*, et comporte nécessairement une image et une désignation textuelle, et éventuellement une description textuelle plus détaillée.  


Description des fichiers  
------------------------

- **data_exploration.ipynb** : première exploration des données,  
- **data_viz_image.ipynb** : exploration et visualisation des images  
- **data_viz_texte.ipynb** : exploration et visualisation du texte,  
- **classification_image.ipynb** : classification des images (VGG16),
- **classification_texte_partie1.ipynb** : classification du texte (Conv-1D, LSTM, GRU),  
- **classification_texte_partie2.ipynb** : classification du texte (Bi-LSTM),  
- **classification_bimodale.ipynb** : classification bimodale avec texte et image (Bi-LSTM + MobileNet),  
- **imgtools.py** : fonctions utilisées dans les notebooks data_exploration.ipynb et data_viz_image.ipynb,  
- **preprocessing.py** : fonctions utilisées dans le notebook data_viz_texte.ipynb,  
- **doc** : rapport et diaporama de présentation du projet,  
- **streamlit** : démonstrateur *Streamlit* (sauf modèles, cf. infra).  


Google Drive  
------------

Le [Google Drive](https://drive.google.com/drive/folders/1c-dc-QARWRhZ912M2AY9iFppn5QyLi7L?usp=sharing) du projet contient les fichiers de sauvegarde de nos modèles de classication entraînés, trop volumineux pour être hébergés sur *GitHub*. Ces fichiers sont nécessaires au démonstrateur *Streamlit*.  

Certains fichiers CSV utilisés dans nos *Jupyter* notebooks sont également présents. 


Réalisation  
-----------

**Réalisé par :**  

* Nada STAOUITE ([LinkedIn](https://www.linkedin.com/in/nada-staouite-330720a5/))  
* Bastien PIQUEREAU ([LinkedIn](https://www.linkedin.com/in/bastien-p-4661331b0/))  
* Lucas GANDY ([LinkedIn](https://www.linkedin.com/in/lucas-gandy/))  

**Supervisé par :**  

* Chloé GUIGA ([LinkedIn](https://www.linkedin.com/in/chloeguiga/))  
