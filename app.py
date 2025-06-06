# Imports
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *  # Your prompt template
import os

# Initialize Flask app
app = Flask(__name__)

# Load .env variables
load_dotenv()

# Load API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set env vars for used SDKs
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Load embeddings (from HuggingFace or whatever you use)
embeddings = download_hugging_face_embeddings()

# Define Pinecone index
index_name = "medicalbot"

# Load existing Pinecone index into LangChain VectorStore
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Set up retriever (k = 3 most relevant chunks)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or gemini-1.5-pro if needed
    temperature=0.4,
    max_output_tokens=500
)

# Build RAG chain components
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Question answering chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Retrieval augmented generation chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Flask Routes
@app.route("/")
def index():
    return render_template('chat.html')  # Your chat UI page

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)
    
    response = rag_chain.invoke({"input": msg})
    
    print("Response:", response["answer"])
    return str(response["answer"])

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
