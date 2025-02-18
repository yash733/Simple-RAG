# RAG-Based Resume Question-Answering Application

This application implements **Retrieval-Augmented Generation (RAG)** to answer questions about a resume. The system processes a **PDF resume**, extracts text, converts it into embeddings, and enables semantic search using **FAISS**. The user can upload their resume dynamically or use a predefined static file.

---

## Features
- 📂 **Dynamic Upload Mode**: Users can upload their resume in real-time.
- 📄 **Static Mode**: The application reads a predefined resume file.
- 🔍 **Semantic Search**: Retrieves relevant sections of the resume based on user queries.
- 🤖 **AI-Powered Retrieval**: Uses **HuggingFace embeddings** for high-quality search results.
- 🖼️ **PDF Viewer**: Displays the uploaded resume alongside the search interface.

---

## 📌 Data Flow

### **1. User Upload (Dynamic Mode)**
- User uploads a **PDF file** via the **Streamlit sidebar**.
- The uploaded file is converted into **binary format** (`getvalue()`) and stored temporarily.
- **PyPDFLoader** extracts text from the uploaded file.
- Text is split into **chunks** (500 characters with 120 overlap) using **RecursiveCharacterTextSplitter**.
- Text chunks are embedded using **HuggingFaceEmbeddings (MiniLM-L6-v2)**.
- The **FAISS vectorstore** is created from these embeddings.
- The user can query the resume, and FAISS retrieves the most relevant responses.

### **2. Static Mode (Predefined Resume Path)**
- The resume file path is directly provided (`C:\Users\yashg\Downloads\yash-gupta.pdf`).
- **PyPDFLoader** extracts text from the specified file.
- Text is processed and embedded **as in dynamic mode**.
- The system allows querying **without requiring user upload**.

---

## 🚀 Key Differences: Dynamic vs Static Mode

| Feature | Dynamic Mode | Static Mode |
|---------|-------------|-------------|
| **Resume Source** | Uploaded by user | Predefined file path |
| **File Handling** | Uses `getvalue()` to extract binary data | Directly loads file using its path |
| **Temporary Storage** | Saves PDF as `temp_resume.pdf` | No extra file writing needed |
| **Flexibility** | Works with any uploaded resume | Limited to a single predefined resume |
| **PDF Viewer** | Uses `getvalue()` to display PDF | Directly passes file path |

---

## 🔥 Why Do We Use `getvalue()` in Dynamic Mode?
When users upload a file in **Streamlit**, it is stored in memory as a **BytesIO object**. To process the file, we need to:
- Convert it into **binary format** using `getvalue()`.
- Write the binary data into a temporary file (`temp_resume.pdf`).
- Load this temporary file using **PyPDFLoader**.

💡 **In contrast, Static Mode does not require this step** because we already have a file path available.

---

## 🛠️ How to Run the Application
### **1. Install Dependencies**
```sh
pip install -r requirements.txt
```
### **2. Run the Application**
```sh
streamlit run app.py
```
---

## 📝 Future Enhancements
- ✅ Improve UI for better user interaction.
- 📊 Add visualization of extracted resume sections.
- 🎯 Fine-tune search accuracy with better embeddings.

---

🔹 **Built with Python, Streamlit, LangChain, and FAISS!** 🚀

### *Dynamic 
![image](https://github.com/user-attachments/assets/963d673a-4707-44ea-a75c-a4228781cfa8)

### *Static
![image](https://github.com/user-attachments/assets/7d963d60-3924-416a-b890-1d9577ebcb72)

