# RAG - Why You Should Hire Me!

## Overview
This project is a **Retrieval-Augmented Generation (RAG) system** built using **Streamlit, LangChain, FAISS, and Hugging Face embeddings**. It allows users to upload their **resume (PDF)** and interactively answer questions about their skills and experience. The system helps candidates highlight their qualifications in an **AI-driven and structured manner**, making it useful for job interviews.

## Features
- 📄 **PDF Upload & Processing**: Users can upload their resume in PDF format.
- 🔍 **Semantic Search with FAISS**: Efficient retrieval of relevant resume content based on queries.
- 🧠 **Hugging Face Embeddings**: Converts text into vector representations for better search accuracy.
- 🤖 **Chat Model Integration**: Uses **Llama 3 (70B)** via Groq API for generating structured answers.
- 🎯 **Job-Specific Responses**: Answers are tailored to Machine Learning, AI, Data Science, and related job roles.
- 📜 **Expandable Extra Info Section**: Provides additional details retrieved from the resume.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/rag-hire-me.git
   cd rag-hire-me
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables (Create a `.env` file):
   ```
   LANGCHAIN_API_KEY=your_langchain_api_key
   GROQ_API_KEY=your_groq_api_key
   LANGCHAIN_PROJECT=your_project_name
   ```
5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## How It Works
1. Upload your **resume PDF**.
2. The system **extracts text** from the resume and **embeds it using Hugging Face models**.
3. The **FAISS vector database** enables fast and efficient retrieval.
4. A **retrieval chain with Llama 3** processes your queries and generates structured responses.
5. You can **ask any question**, such as "Tell me about my experience in AI," and get a detailed answer.
6. The **Extra Info section** provides more relevant details from the resume.

## Technologies Used
- **Streamlit**: Web UI framework for interaction
- **LangChain**: Framework for LLM-based applications
- **FAISS**: Fast vector search engine for efficient document retrieval
- **Hugging Face Embeddings**: All-MiniLM-L6-v2 for text vectorization
- **ChatGroq API (Llama 3 - 70B)**: For generating intelligent responses
- **PyPDFLoader**: Extracting text from PDFs
- **Python Dotenv**: Managing API keys securely

## Future Improvements
- ✅ Support for **multiple document uploads**
- ✅ Better prompt engineering for improved responses
- ✅ Integration with **more LLMs** for comparison
- ✅ Option to **save and export** responses

## 📌 Data Flow

### **1. User Upload (Dynamic Mode)**
- User uploads a **PDF file** via the **Streamlit sidebar**.
- The uploaded file is converted into **binary format** (`getvalue()`) and stored temporarily.
- **PyPDFLoader** extracts text from the uploaded file.
- Text is split into **chunks** (500 characters with 120 overlap) using **RecursiveCharacterTextSplitter**.
- Text chunks are embedded using **HuggingFaceEmbeddings (MiniLM-L6-v2)**.
- The **FAISS vectorstore** is created from these embeddings.
- The user can query the resume, and FAISS retrieves the most relevant responses.


---

## 🔥 Why Do We Use `getvalue()` in Dynamic Mode?
When users upload a file in **Streamlit**, it is stored in memory as a **BytesIO object**. To process the file, we need to:
- Convert it into **binary format** using `getvalue()`.
- Write the binary data into a temporary file (`temp_resume.pdf`).
- Load this temporary file using **PyPDFLoader**.

💡 **In contrast, Static Mode does not require this step** because we already have a file path available.

🔹 **Built with Python, Streamlit, LangChain, and FAISS!** 🚀

### *Dynamic 
![image](https://github.com/user-attachments/assets/8e0eed5a-12c1-4cbf-b2cd-233f83768c4a)


![image](https://github.com/user-attachments/assets/c19a1191-317a-4b51-85fb-c7577e57e691)


