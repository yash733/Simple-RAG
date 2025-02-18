import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

#from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from streamlit_pdf_viewer import pdf_viewer


os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

resume = r"C:\Users\yashg\Downloads\yash-gupta.pdf"
loader = PyPDFLoader(resume)
docs = loader.load()

#from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 120)
docs = text_splitter.split_documents(docs)

#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.vectorstores import FAISS

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#texts = [doc.page_content for doc in docs]  

vectorstore = FAISS.from_documents(docs, embedding_function)

st.title("Rag-Why you should hire ME!")
input_text = st.text_input("Ask question like : Your experience in the foeld of AI")
if input_text:
    result = vectorstore.similarity_search(input_text)
    for i, doc in enumerate(result, start=1):
        st.write(f"🔹 **Document {i}**")
        st.write(f"📌 **ID:** {doc.id}")
        st.write(f"📄 **Content:**\n{doc.page_content}\n")
        st.write("=" * 80) 


#from streamlit_pdf_viewer import pdf_viewer
with st.sidebar:
    pdf_viewer(input = resume, width = 900)