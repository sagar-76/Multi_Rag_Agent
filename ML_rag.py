import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import requests

class PDFRAGQA:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.vector_store = None

    def load_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF: {e}")

    def create_vector_store(self, text: str):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = splitter.create_documents([text])
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(documents, embeddings)

    def retrieve_context(self, query: str, k: int = 4) -> str:
        if not self.vector_store:
            raise ValueError("Vector store is not initialized.")
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        results = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in results)
        return context[:3000]  # truncate long context if needed

    def call_groq_llama(self, prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"API Error: {response.status_code} - {response.text}"

    def ask(self, query: str) -> str:
        context = self.retrieve_context(query)
        prompt = f"""You are a helpful assistant.
Answer ONLY from the provided context.
If the context is insufficient, just say you don't know.

Context: {context}
Question: {query}

Answer:"""
        return self.call_groq_llama(prompt)
