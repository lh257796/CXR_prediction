import streamlit as st
def main():
    st.title("About me")
    st.subheader("Academic Background")

    st.write("""
        Thank you for checking out my about me page. I'm currently a Master's student in Data Science at Eastern University. 
        I'm passionate about building machine learning models for healthcare 
        applications. One example is using computer vision--like this capstone, where I build medical imaging 
        models to create predictions for different chest-related pathologies.
        
        - **Interests:** Machine Learning, Computer Vision, Healthcare Tech
        - **Hobbies:** Pickleball, Concerts, Fashion, Cooking, Reading
        - **Unique Aspect:** I have previously started my own streetwear company and attended med school during the pandemic.
    """)

if __name__ == "__main__":
    main()