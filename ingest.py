from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

def load_and_chunk_document(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks):
    # Step 1: Load the embedding model (calls OpenAI API to convert text → vectors)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Step 2: Embed all chunks and store them in a FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Step 3: Save the vector store to disk so we don't re-embed every time
    vector_store.save_local("faiss_index")
    print("Vector store saved to faiss_index/")

    return vector_store


if __name__ == "__main__":
    pdf_path = os.path.join("data", "sample.pdf")

    print("Step 1: Loading and chunking document...")
    chunks = load_and_chunk_document(pdf_path)
    print(f"Total chunks: {len(chunks)}")

    print("\nStep 2: Creating embeddings and vector store...")
    vector_store = create_vector_store(chunks)
    print("Done! Your chunks are now embedded and stored in FAISS.")