import streamlit as st
import os
from streamlit_extras.buy_me_a_coffee import button

class MathsForML:
    def __init__(self):
        pass
    def run(self):
        st.title("Mathematics for Machine Learning :books:")
        st.subheader("This app is designed to help you understand the mathematics behind machine learning.\
                     I think this sources will be helpful for you... :computer:")
        st.write("Written by: [Atalay Erdogan] :writing_hand:")
        with st.form(key='my_form'):
            box = st.selectbox("Select the topic", ["Linear Algebra", "Calculus"])
            if box == "Linear Algebra" and st.form_submit_button("Go to the page"):
                st.switch_page(os.getcwd() + "/pages/LinearAlgebra.py")
            elif box == "Calculus" and st.form_submit_button("Go to the page"):
                st.switch_page(os.getcwd() + "/pages/Calculus.py")
        st.write("It will be grateful if you buy me a coffee :coffee:")
        button(username="atalayerdgn", floating=False, width=221)
    
def main():
    m = MathsForML()
    m.run()

if __name__ == "__main__":
    main()