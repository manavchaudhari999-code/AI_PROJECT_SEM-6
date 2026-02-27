from langchain_community.llms import Ollama

def load_llm():
    return Ollama(model="llama3")
