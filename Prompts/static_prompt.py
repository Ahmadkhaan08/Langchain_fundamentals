from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

st.header("Research Tool")
user_input=st.text_input("Write something here")

if st.button("Summarize"):
    response=model.invoke(user_input)
    st.write(response.content[0]['text'])