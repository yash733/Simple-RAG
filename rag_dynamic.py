import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from streamlit_pdf_viewer import pdf_viewer

# Load API Keys
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

st.title("📄 RAG - Why You Should Hire Me!")

# Upload PDF
uploaded_file = st.sidebar.file_uploader("Upload Your Resume (PDF)", type=["pdf"])

if uploaded_file:
    # Load PDF and Process
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    #from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader("temp_resume.pdf")
    docs = loader.load()

    #from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)
    docs = text_splitter.split_documents(docs)

    #from langchain_huggingface import HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #texts = [doc.page_content for doc in docs] 

    #from langchain_community.vectorstores import FAISS 
    vectorstore = FAISS.from_documents(docs, embedding_function)
    #vectorstore = FAISS.from_texts(texts, embedding_function)      !! if loading text !!


    with st.sidebar:
        pdf_viewer(input=uploaded_file.getvalue(), width=700)

    input_text = st.text_input("Ask a question (e.g., Your experience in AI)")
    if input_text:
        result = vectorstore.similarity_search(input_text)
        for i, doc in enumerate(result, start=1):
            st.write(f"🔹 **Document {i}**")
            st.write(f"📄 **Content:**\n{doc.page_content}\n")
            st.write("=" * 80)