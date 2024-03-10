import sys
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

import time

import os
from dotenv import load_dotenv

load_dotenv()

def build_rag_pipeline(pdf_dir):
    # Load the PDF
    loader = PyPDFLoader(pdf_dir)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Select which embeddings we want to use
    embeddings = OpenAIEmbeddings()

    # Create the vectorstore to use as the index
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss_index")

    # Expose this index in a retriever interface
    retriever = db.as_retriever()

    # Create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )

    return qa


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    print("Building RAG pipeline for file:", file_path)
    start = time.time()
    # Build the RAG pipeline
    qa = build_rag_pipeline(file_path)
    print("Building took {0}s".format(time.time()-start))
    print("Done building RAG pipeline")