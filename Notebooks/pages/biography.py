import streamlit as st
def main():
    st.title("About me")
    st.subheader("Academic Background")

    st.write("""
        Hello! I'm Han, currently a Master's student in Data Science. 
        I'm passionate about building machine learning models for healthcare 
        applications. Computer vision is one example, using medical imaging 
        models to create predictions for different pathologies.
        
        - **Interests:** Machine Learning, Computer Vision, Healthcare Tech
        - **Hobbies:** Pickleball, Concerts, Fashion, Cooking, Reading
        - **Unique Aspect:**I have previously started my own streetwear company and attended med school during the pandemic.
    """)

if __name__ == "__main__":
    main()