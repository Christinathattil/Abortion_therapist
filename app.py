#!pip install -q python-dotenv langchain langchain-core langchain-docling langchain-huggingface langchain-community faiss-cpu sentence-transformers transformers docling streamlit

import streamlit as st
import os
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_docling.loader import ExportType
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker

# Helper to get environment variables (from Colab or OS)
def _get_env_from_colab_or_os(key):
    try:
        from google.colab import userdata
        try:
            return userdata.get(key)
        except userdata.SecretNotFoundError:
            pass
    except ImportError:
        pass
    return os.getenv(key)

# Load environment variables
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
HF_TOKEN = _get_env_from_colab_or_os("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN environment variable not set. Please set it in your .env file (local) or Streamlit secrets (deployment).")
    st.stop()

# Adjust this to your local path for the PDF
FILE_PATH = [str(Path(__file__).parent / "data" / "Trent Horn - 20 Answers To Abortion.pdf")]

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
EXPORT_TYPE = ExportType.DOC_CHUNKS
PROMPT = PromptTemplate.from_template(
    """You are a supportive and non-judgmental therapist. Your purpose is to provide information and understanding related to abortion, drawing solely from the provided document, "Trent Horn - 20 Answers To Abortion.pdf".

When responding, maintain an empathetic, neutral, and respectful tone. Do not offer personal opinions, medical advice, psychological counseling, or make judgments. Your role is to calmly and clearly present information as found in the document to help the user understand the topic better from the perspective of this resource.

Context information is below.
---------------------
{context}
---------------------
Based on the provided context from the document, please answer the query in a non-judgmental and informative manner:
Query: {input}
Answer:
""",
)
TOP_K = 3
FAISS_INDEX_PATH = "faiss_index_abortion_pdf"

def clip_text(text, threshold=350):
    return f"{text[:threshold]}..." if len(text) > threshold else text

@st.cache_resource
def setup_rag_pipeline():
    """
    Sets up the RAG pipeline. This function is cached to avoid
    re-running heavy computations (like loading documents and creating embeddings)
    every time the Streamlit app reruns.
    """
    st.spinner("Loading document and building knowledge base... This might take a moment.")
    print(f"[{datetime.now()}] Loading document from: {FILE_PATH[0]}")

    # Check if the PDF file exists
    if not Path(FILE_PATH[0]).exists():
        st.error(f"PDF file not found at {FILE_PATH[0]}. Please ensure it's in the 'data/' directory relative to app.py.")
        st.stop()

    loader = DoclingLoader(
        file_path=FILE_PATH,
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
    )
    docs = loader.load()

    if EXPORT_TYPE == ExportType.DOC_CHUNKS:
        splits = docs
    elif EXPORT_TYPE == ExportType.MARKDOWN:
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
    else:
        raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")

    print(f"[{datetime.now()}] Loaded {len(splits)} document splits.")

    print(f"[{datetime.now()}] Initializing embedding model...")
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

    if Path(FAISS_INDEX_PATH).exists():
        print(f"[{datetime.now()}] Loading FAISS index from {FAISS_INDEX_PATH}...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding, allow_dangerous_deserialization=True)
    else:
        print(f"[{datetime.now()}] Creating FAISS index from documents...")
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embedding,
        )
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"[{datetime.now()}] FAISS index saved to {FAISS_INDEX_PATH}")

    print(f"[{datetime.now()}] Initializing generative model and RAG chain...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    endpoint_llm = HuggingFaceEndpoint(
        repo_id=GEN_MODEL_ID,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.7,
    )
    llm = ChatHuggingFace(llm=endpoint_llm)

    question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Chatbot for '20 Answers to Abortion'", layout="wide")

st.title("📚 RAG Chatbot: 20 Answers to Abortion")
st.markdown("Ask questions about the document 'Trent Horn - 20 Answers To Abortion.pdf'.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
            with st.expander("Show Sources"):
                for i, doc in enumerate(message["context"]):
                    st.write(f"**Source {i + 1}:**")
                    st.code(clip_text(doc.page_content), language='text')

                    st.write("**Metadata:**")
                    # Extract and display relevant metadata
                    file_name = None
                    if 'file_path' in doc.metadata and doc.metadata['file_path']:
                        file_name = Path(doc.metadata['file_path']).name
                    elif 'source' in doc.metadata and doc.metadata['source']:
                        file_name = Path(doc.metadata['source']).name

                    if file_name:
                        st.write(f"  *File Name*: {file_name}")

                    if 'page' in doc.metadata:
                        st.write(f"  *Page Number*: {doc.metadata['page']}")

                    # Check for headings (if present from a different chunking strategy or DoclingLoader)
                    # DoclingLoader for PDFs often provides 'Header_X' if it detects them
                    for header_key in ['Header_1', 'Header_2', 'Header_3']:
                        if header_key in doc.metadata and doc.metadata[header_key]:
                            # Format the header key nicely (e.g., "Header 1")
                            display_header_key = header_key.replace('_', ' ')
                            st.write(f"  *{display_header_key}*: {doc.metadata[header_key]}")

                    # You can add other specific metadata fields here if needed, e.g., 'title'
                    if 'title' in doc.metadata and doc.metadata['title']:
                        st.write(f"  *Title*: {doc.metadata['title']}")

# Main chat input
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            rag_chain = setup_rag_pipeline() # Get the cached RAG pipeline
            response_dict = rag_chain.invoke({"input": prompt})
            full_response = response_dict["answer"]
            sources = response_dict["context"]

            message_placeholder.markdown(full_response)
            with st.expander("Show Sources"):
                for i, doc in enumerate(sources):
                    st.write(f"**Source {i + 1}:**")
                    st.code(clip_text(doc.page_content), language='text')

                    st.write("**References from Document:**")
                    # Extract and display relevant metadata
                    file_name = None
                    if 'file_path' in doc.metadata and doc.metadata['file_path']:
                        file_name = Path(doc.metadata['file_path']).name
                    elif 'source' in doc.metadata and doc.metadata['source']:
                        file_name = Path(doc.metadata['source']).name

                    if file_name:
                        st.write(f"  *File Name*: {file_name}")

                    if 'page' in doc.metadata:
                        st.write(f"  *Page Number*: {doc.metadata['page']}")

                    for header_key in ['Header_1', 'Header_2', 'Header_3']:
                        if header_key in doc.metadata and doc.metadata[header_key]:
                            display_header_key = header_key.replace('_', ' ')
                            st.write(f"  *{display_header_key}*: {doc.metadata[header_key]}")

                    if 'title' in doc.metadata and doc.metadata['title']:
                        st.write(f"  *Document Title*: {doc.metadata['title']}")

            st.session_state.messages.append({"role": "assistant", "content": full_response, "context": sources})

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})

if __name__ == "__main__":
    pass
