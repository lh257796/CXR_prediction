import streamlit as st

def main():
    st.title("General Projects")
    st.write("Most of my other completed projects can be found at my [personal website](https://hanlu.dev).")
    st.write("Examples of things I've worked on:")
    st.markdown("""
         - IoTs (Smart Mirror with Raspberry Pi  (_picture below_!)) 
         - React Native application incorporating NFT middleware for a Hackathon placing 3rd
         - 3D Platformer game designed in Unreal Engine 5
         - Reservation system built in Python + Django for music venues
         - Dog walking website built in React (PERN stack) with functioning database
    """)

    st.write("For other work in progress projects or to see some of my repos in more depth, including the one for this project, check out my [Github](https://github.com/lh25779)!")
    st.image("Notebooks/pages/mirror.jpg")
if __name__ == "__main__":
    main()