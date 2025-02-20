import streamlit as st

def main():
    st.set_page_config(
        page_title="My Personal Site",
        page_icon=":wave:",
        layout="centered"
    )
    st.title("Welcome to my Capstone Website")
    st.write(
        "Use the sidebar to navigate: "
        "Biography, Resume, Projects, and the deployed model for my capstone project."
    )
    st.write("""
             Hi! My name is Han and I currently work as a Data Analyst 
             for a cancer center at MSKCC. 
             
             My capstone project at Eastern University
             as a Masters Student in Data Science showcases a neural network 
             machine learning model trained on over 80k+ medical images across 30k+ patients
             to classify images from 14 different thoracic pathologies, including cardiomegaly,
             masses, and hernias. 

             The deployed model is meant to be an interactive demo that allows 
             YOU, the viewer, to upload your own chest X-ray image to get diagnosed with 
             a pathology. I've also incorporated a Grad-CAM function to create a 
             heatmap that illustrates how the model 'sees' features in the image to
             make a classification. 

             I hope you enjoy! For more of my work, you can visit my [website](https://hanlu.dev).
             """)

if __name__ == "__main__":
    main()