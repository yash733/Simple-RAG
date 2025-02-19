import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load API Keys
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

from langchain_groq import ChatGroq

model = ChatGroq(model="llama3-70b-8192")



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
    #__ vectorstore = FAISS.from_texts(texts, embedding_function)      !! if loading text !!
    retriever = vectorstore.as_retriever()

    #from langchian_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system","Act like my assistant, use the provided context and answer the question's asked by recruter. Specify my capable in number. Also any other details that can help me in getting shortlisted in the role of Machine Learning Engineer, Artificial Intelligance Engineer, Data Analytics, Data Scientist or any other role or job discription related to my skill set. \n{context}"),
            ("user","{input}")
        ]
    )

    #from langchain.chains.combine_documents import create_stuff_documents_chain
    document_chain = create_stuff_documents_chain(model,prompt)
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    with st.sidebar:
        pdf_viewer(input=uploaded_file.getvalue(), width=700)

    input_text = st.text_input("Ask a question (e.g., Your experience in AI)")
    if input_text:
        result = retriever_chain.invoke({"input":input_text, "context":"context"})
        st.write(result['answer'])        
        #result = vectorstore.similarity_search(input_text)
    
        with st.expander("Extra Info"):
            for i, doc in enumerate(result['context'], start=1):
                st.write(f"🔹 **Document {i}**")
                st.write(f"📄 **Content:**\n{doc}\n")
                st.write("=" * 80)