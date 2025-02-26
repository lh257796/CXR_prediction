import streamlit as st

def main():
    st.title("Resume: _Brief summary_")
    st.subheader("Education")
    st.write("""
    - **M.S. in Data Science** - Eastern University (Est. Grad: Mar 2025)
    - B.A. in Chemistry - New York University (Grad: May 2019)
    """)

    st.subheader("Work Experience")
    st.write("""

   - **Python / Data Analytics Tutor**, Self-Employed (2024 - Current)
       - Taught python fundamentals, ie syntax/control flow/algos
       - Introduced analytical techniques for exploratory data analysis and visualizations
   - **Fullstack Software Engineer**, GigFinesse (2022 - 2024)
       - Created Tableau dashboards to visualize KPIs
       - Built reservation systems in Python/Django for venues
       - Constructed React components and performed UI overhauls
             
    """)
    st.write("_For my full resume (updated 1/2025), see the attached image below:_")

    st.image("Notebooks/pages/Resume.png", caption="Han's Resume, updated Jan 2025")

if __name__ == "__main__":
    main()