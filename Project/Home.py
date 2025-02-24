import streamlit as st

def main():
    st.set_page_config(
        page_title="My Personal Site",
        page_icon=":wave:",
        layout="centered"
    )
    st.title("Welcome to my Capstone Website")
    st.write(
        "_Use the sidebar to navigate:_ "
        "_Biography, Resume, Projects, and the deployed capstone model._"
    )
    st.write("""
             Hi!
             
            My name is Han and I currently work as a Data Analyst 
             for a cancer center at MSK. 
             
             My capstone project at Eastern University
             as a Data Science Masters student demonstrates a convolutional neural network (CNN)
             machine learning model trained on over 80k+ medical images of chest X-rays across 30k+ patients
             to classify images from 14 different thoracic pathologies, including cardiomegaly,
             masses, and hernias. 

             The deployed model is meant to be an interactive demo that allows 
             YOU, the viewer, to upload your own chest X-ray image to get diagnosed with 
             a pathology. I've also incorporated a Grad-CAM function to create a 
             heatmap that illustrates how the model 'sees' features in the image to
             make a classification. Please note that this deployed model is NOT to be used for anything 
             except demonstration and research purposes!

             I hope you enjoy! For more of my work, you can visit my [website](https://hanlu.dev).
             If you'd like to reach out, you can contact me at my [email](mailto:han.lu.122@gmail.com)
             """)

if __name__ == "__main__":
    main()