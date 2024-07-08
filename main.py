import pinecone
import json
import os
import shutil
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, File, UploadFile
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from utils import initialize_model_and_embeddings, create_vector_store, load_and_split_pdf, construct_chain
from request import ask_question
from fastapi import FastAPI, HTTPException, File, UploadFile
from dotenv import load_dotenv
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from pinecone import Pinecone, ServerlessSpec


# Initialize FastAPI app and load environment variables
app = FastAPI()
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load necessary configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"
# MODEL = "llama3"
INDEX_NAME = "rag"

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

try:
    pc = Pinecone(api_key=pinecone_api_key)

    index_list = pc.list_indexes().names()
    if INDEX_NAME not in index_list:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )
    logger.info(f"Pinecone index '{INDEX_NAME}' is ready.")
except Exception as e:
    logger.error(f"Error initializing Pinecone: {e}")
    raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Pinecone initialization failed")

# Model and embedding objects
model, embeddings, vector_store = None, None, None


@app.get("/")
def read_root():
    return {"Welcome to": "RAG System Q&A Chatbot"}


@app.on_event("startup")
def startup_event():
    global model, embeddings
    model, embeddings = initialize_model_and_embeddings(MODEL)
    logger.info("Model and embeddings initialized.")


@app.post("/uploader/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        save_path = "YOLO.pdf"
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and split the PDF into pages
        pages = load_and_split_pdf(save_path)

        # Create vector store with embeddings
        global vector_store
        vector_store = create_vector_store(pages, embeddings)
        logger.info("PDF uploaded and vector store created.")

        return {"filename": file.filename}
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="File processing failed")


@app.post("/ask-question/")
async def ask_question_endpoint(question: str):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST,
                            detail="Vector store is not initialized")
    try:
        response = ask_question(model, vector_store, question)
        return {"question": question, "answer": response}
    except Exception as e:
        logger.error(f"Error asking question: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="Question processing failed")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
