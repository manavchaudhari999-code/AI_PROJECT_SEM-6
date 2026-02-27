from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def ingest_docs():
    documents = []
    folder = "data/medical_papers"

    if not os.path.exists(folder) or not os.listdir(folder):
        print("⚠️ No PDFs found in data/medical_papers/")
        return

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("data/vectorstore")

    print("✅ Medical knowledge base created successfully")

if __name__ == "__main__":
    ingest_docs()
