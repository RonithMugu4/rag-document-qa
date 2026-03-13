from dotenv import load_dotenv
load_dotenv()



from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def load_vector_store():
    # Load the FAISS index we saved to disk in Phase 2
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True  # Required by LangChain for local files
    )
    return vector_store


def retrieve_relevant_chunks(query: str, k: int = 4):
    vector_store = load_vector_store()

    # Embeds the query and finds the k most similar chunks using cosine similarity
    results = vector_store.similarity_search(query, k=k)

    return results


if __name__ == "__main__":
    # Test with a question related to your sample PDF
    query = "What is social media mining?"

    print(f"Query: {query}")
    print(f"\nTop 4 most relevant chunks:\n")

    chunks = retrieve_relevant_chunks(query)

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content)
        print()
