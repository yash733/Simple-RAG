import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
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

#Store History in "chat_history" a list[]
if "chat_history" not in st.session_state:
   st.session_state.chat_history = []

# Load context from context.txt
context_text = ""
with open("D:/krish/agent/RAG/contex.txt", "r") as file:
    context_text = file.read()


if uploaded_file:
    # Load PDF and Process
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader("temp_resume.pdf")
    docs = loader.load()
    #print("docs ",docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)
    docs = text_splitter.split_documents(docs)

    # Split context text into chunks and create Document objects
    context_docs = text_splitter.split_text(context_text)

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #texts = [doc.page_content for doc in docs] 
     
    vectorstore = FAISS.from_documents(docs, embedding_function)
    vectorstore = FAISS.from_documents(context_docs, embedding_function)
    #__ vectorstore = FAISS.from_texts(texts, embedding_function)      !! if loading text !!
    retriever = vectorstore.as_retriever()

    #from langchian_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate(
        [
            ("system","Act as my assistant and representative during recruitment discussions. Your role is to answer questions posed by recruiters using my resume as context. Ensure that responses highlight my compatibility with their requirements, particularly for roles related to Machine Learning, Data Science, Data Analytics, Computer Vision, NLP, AI Engineering, or any other relevant fields."
            "Back up every claim with specific examples from my experience, showcasing my skills, achievements, certifications and projects to make me a strong candidate. Dont over advertize"
            "If asked about AI-related projects exaplin a bit of my expriaence gained or my current work, provide them with my X and LinkedIn profile links, featched from context and additonal data {data}. \n{context}"),
            MessagesPlaceholder(variable_name = "chat_history_"),
            ("user","{input}")
        ]
    )

    #from langchain.chains.combine_documents import create_stuff_documents_chain
    document_chain = create_stuff_documents_chain(model,prompt)
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    #Trim Chat_History
    trimmer = trim_messages(
        max_tokens = 200,
        strategy = 'last',
        token_counter = model,
        include_system = True,
        allow_partial = False,
        start_on = 'human'
    )
        
    with st.sidebar:
        pdf_viewer(input=uploaded_file.getvalue(), width=700)

    input_text = st.text_input("Ask a question (e.g., Your experience in AI)")
    
       
    if input_text:
        #store user input
        #st.session_state.chat_history.add_user_message(input_text) 
        st.session_state.chat_history.append(HumanMessage(content=input_text))

        trimmed_message = trimmer.invoke(st.session_state.chat_history)

        result = retriever_chain.invoke({
            "input":input_text,
            "data":context_docs, 
            "context":docs, 
            "chat_history_":trimmed_message})
        
        #store ai message
        #st.session_state.chat_history.add_ai_message(result['answer'])
        st.session_state.chat_history.append(AIMessage(content=result['answer']))
        
        st.write(result['answer'])        
        #result = vectorstore.similarity_search(input_text)
    
        with st.expander("Extra Info"):
            for i, doc in enumerate(result['context'], start=1):
                st.write(f"🔹 **Document {i}**")
                st.write(f"📄 **Content:**\n{doc}\n")
                st.write("=" * 80)
            
            st.subheader('Chat History')
            for msg in st.session_state.chat_history:
                if isinstance(msg, HumanMessage):
                    st.write(f"👤 **You:** {msg.content}")
                elif isinstance(msg, AIMessage):
                    st.write(f"🤖 **AI:** {msg.content}")
            
            st.write(st.session_state.chat_history)

    if st.button('Clear ChatHistory'):
        st.session_state.chat_history = []
