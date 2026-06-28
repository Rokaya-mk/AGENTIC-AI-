import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def get_llm(temperature: float = 0.0):
    return ChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=GROQ_API_KEY)
