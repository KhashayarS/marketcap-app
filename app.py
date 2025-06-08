import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain_groq.chat_models import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup
import pandas as pd

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# STEP 1: Set your Groq API Key
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# STEP 2: Use WebBaseLoader to load CoinMarketCap
def load_coinmarketcap_text():
    loader = WebBaseLoader("https://coinmarketcap.com/")
    docs = loader.load()
    return docs[0].page_content

# STEP 3: Load LLaMA 2 from Groq
def get_llm():
    llm = ChatGroq(model='llama-3.3-70b-versatile')
    return llm

# STEP 4: Market analysis
def analyze_market(page_content, llm):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""
    You are a crypto market analyst. Below is the scraped text from CoinMarketCap.com.

    Analyze the current state of the crypto market. Identify trends, bullish or bearish sentiment, any notable changes, and suggest potential trading opportunities.

    Scraped Page:
    {content}

    Market Analysis:
    """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"content": page_content})

# STEP 5: Question answering
def ask_question(page_content, llm, question):
    prompt = PromptTemplate(
        input_variables=["content", "question"],
        template="""
You are a crypto market assistant. Based on the page content below, answer the user's question.

CoinMarketCap Data:
{content}

User Question: {question}

Answer:
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"content": page_content, "question": question})

# Using methods to create streamlit app
page_text = load_coinmarketcap_text()
llm = get_llm()

st.title('Coinmarketcap Analyzer')
question = st.text_input('What do you want to know about the market today \n(regarding data derived from coinmarketcap.com)')
if question:
    response = ask_question(page_text, llm, question)
    st.write(response)

# # MAIN
# if __name__ == "__main__":
#     print("📥 Loading CoinMarketCap page...")
#     page_text = load_coinmarketcap_text()

#     print("⚙️ Loading Groq LLaMA 2...")
#     llm = get_llm()

#     print("\n📊 Market Analysis:")
#     print(analyze_market(page_text, llm))

#     print("\n❓ Example Q&A:")
#     question = "Which coins are rising rapidly?"
#     print(f"Q: {question}")
#     print("A:", ask_question(page_text, llm, question))