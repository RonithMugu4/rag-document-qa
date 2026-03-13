import os
import streamlit as st
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

# On Streamlit Cloud, secrets are not env vars automatically — inject them
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

from ingest import load_and_chunk_document, create_vector_store
from generator import generate_answer
import tempfile

# Page configuration
st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG Document Q&A System")
st.write("Upload a PDF and ask it anything.")

# Step 1: File uploader widget
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Save uploaded file to a temp location so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Step 2: Process the PDF only once using session state
    if "vector_store_ready" not in st.session_state:
        with st.spinner("Processing your PDF..."):
            chunks = load_and_chunk_document(tmp_path)
            create_vector_store(chunks)
            st.session_state.vector_store_ready = True  # Remember it's been processed
        st.success(f"Ready! {len(chunks)} chunks indexed.")

    # Step 3: Question input
    question = st.text_input("Ask a question about your document:")

    if question:
        with st.spinner("Searching and generating answer..."):
            answer = generate_answer(question)
        st.markdown("### Answer")
        st.write(answer)

    # Clean up temp file
    os.unlink(tmp_path)