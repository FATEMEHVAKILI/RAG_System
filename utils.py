import os
from dotenv import load_dotenv
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "rag"

def initialize_model_and_embeddings(model_name):
    if model_name.startswith("gpt"):
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=model_name)
        embeddings = OpenAIEmbeddings()
    else:
        model = Ollama(model=model_name)
        embeddings = OllamaEmbeddings(model=model_name)
    return model, embeddings

def load_and_split_pdf(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    return pages

def create_vector_store(pages, embeddings):
    pinecone = PineconeVectorStore.from_documents(
        pages, embeddings, index_name=INDEX_NAME
    )
    retriever = pinecone.as_retriever()
    return retriever

def construct_chain(model, retriever):
    template = """
    Context: {context}

    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    parser = StrOutputParser()

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | parser
    )
    return chain
