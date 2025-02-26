import streamlit as st
def main():
    st.title("About me")
    st.subheader("Academic Background")

    st.write("""
        Thank you for checking out my about me page!
        """)
    
    st.image('Notebooks/pages/selfie.jpg', width = 500)

    st.write("""
             
        I'm currently a Master's student in Data Science at Eastern University. 
        I'm passionate about building machine learning models for healthcare 
        applications. One example is using computer vision--like this capstone, where I build medical imaging 
        models to create predictions for different chest-related pathologies.
             
        I've always been interested in technology from a young age. As a middle schooler, I would procrastinate 
        studying by creating games and programs using BASIC on a TI-84 calculator, and sell popular games for money
        instead of doing my precalc homework. As an undergrad at NYU, I studied premed coursework and majored in Chemistry
        but still enjoyed my two courses in Python and web design. Most recently, I placed 3rd in a hackathon 
        creating a React Native application that utilizes NFC middleware to distribute meal-cards and enable
        restaurants/food pantries to efficiently provide meals to the needy and underserved populations. In my free time,
        I enjoy volunteering with New York Cares to give back to my commmunity here in NYC! 
             
        I would love to work on creating machine learning models to improve our healthcare system. It's a dream of mine to 
        renovate American medicine in a way that provides medical accessibility (literacy and care access particularly) 
        to every American, and eventually everyone in the world. I believe ML models are a means of providing this access, 
        and I'm grateful for my time at Eastern for giving me the tools to learn how to apply such models for incredible usecases 
        such as the one in my capstone project!
        
        - **Interests:** Machine Learning, Computer Vision, Healthcare Tech
        - **Hobbies:** Pickleball, Concerts, Fashion, Cooking, Reading
        - **Unique Aspect:** I have previously started my own streetwear company and attended med school during the pandemic.
             
        Thank you for reading and checking out my website and biography. If you're interested in learning more about me, check out 
        the sidebar to keep browsing!
        
    """)

if __name__ == "__main__":
    main()