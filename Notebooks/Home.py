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
        "_Biography, Resume, Projects, and the deployed ML model._"
    )
    st.write("""
             Hi! :wave:


            I'm Han. Welcome to my website--here I'll be providing 
            a brief bio, my resume, links to my other works, and a demonstration of a
            machine learning project I've been working on for months! :books:
            """)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image("Notebooks/pages/transparent_background_profpic.png", caption="Nice to meet you!", width = 200)

    st.write("""
             My capstone project at Eastern University
             as a Data Science Masters student demonstrates a convolutional neural network (CNN)
             machine learning model trained on over 80k+ medical images of chest X-rays across 30k+ patients
             to classify images from 14 different thoracic pathologies, including cardiomegaly,
             masses, and hernias. :medical_symbol:

             The deployed model is meant to be an interactive demo that allows
             YOU, the viewer, to upload your own chest X-ray image to get diagnosed with
             a pathology. I've also incorporated a Grad-CAM function to create a
             heatmap that illustrates how the model 'sees' features in the image to
             make a classification. 
             
             :warning: Please note that this deployed model is NOT to be used for anything
             except demonstration and research purposes! :warning:

             I hope you enjoy! For more of my work, you can visit my [website](https://hanlu.dev).
             If you'd like to reach out, you can contact me at my :email: [email](mailto:han.lu.122@gmail.com), or 
              send me a message on [LinkedIn](https://www.linkedin.com/in/-han-lu-)! :zap:
             """)

    

if __name__ == "__main__":
    main()
