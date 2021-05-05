import streamlit as st
import awesome_streamlit as ast

import pages.presentation.presentation
import pages.classification_images.classification_images
import pages.classification_texte.classification_texte
import pages.classification_bimodale.classification_bimodale


# Load ast services
ast.core.services.other.set_logging_format()


# Dictionary of pages modules
PAGES = {
    "Présentation": pages.presentation.presentation,
    "Classification d'image": pages.classification_images.classification_images,
    "Classification de texte": pages.classification_texte.classification_texte,
    "Classification bimodale": pages.classification_bimodale.classification_bimodale
}



def main():
    """Main function of the App"""
    
    # Create sidebar and set title
    st.sidebar.title("Projet eShopPye")
    
    # Page selection
    selection = st.sidebar.radio("Barre de navigation", list(PAGES.keys()))
    page = PAGES[selection]
    
    # Show loading
    with st.spinner("Chargement de {}...".format(selection)):
        ast.shared.components.write_page(page)
    
    # Info box
    st.sidebar.info(
		"Réalisé par: Nada STAOUITE, Bastien PIQUEREAU, Lucas GANDY. "
        "Sous la supervision de: Chloé GUIGA de Datascientest."
        )



if __name__ == "__main__":
    main()