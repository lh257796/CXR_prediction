import streamlit as st

def main():
    st.title("Résumé")
    st.subheader("Education")
    st.write("""
    - **M.S. in Data Science** - Eastern University (Grad: Mar 2025)
    - B.A. in Chemistry - New York University (Grad: May 2019)
    """)

    st.subheader("Work Experience")
    st.write("""
    1. **Data Analyst**, Memorial Sloan Kettering Cancer Center (2025 - Present)
       - Built chatbots and worked on MLOps 
       - Developed internal Python apps for data cleaning

    2. **Fullstack Software Engineer**, GigFinesse (2022 - 2024)
       - Created Tableau dashboards for venues and internal teams presenting KPIs
       - Built reservation systems in Python/Django for venues
       - Constructed React components and performed UI overhauls
             
    """)

    st.image("resume_graphic.pdf", width=200)

if __name__ == "__main__":
    main()