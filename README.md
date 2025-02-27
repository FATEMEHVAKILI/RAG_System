<div dir="rtl"> 


# [RAG_System](https://github.com/FATEMEHVAKILI/RAG_System) 
For more info:

- [GitHub](https://github.com/FatemehVakili/)

- [Resume](https://fatemehvakili.github.io/)

------------------
The repository contains a project based on RAG systems. 
To set up the required dependencies for the code, use the command below:
 ```bash
pip install -r requirements.txt
```
You can use the RAG system by running the file RAGwithoutFastAPI.py :
 ```bash
python RAGwithoutFastAPI.py
```
If you want to use it with Fast API, follow these instructions:
 ```bash
 uvicorn main:app --reload
```
Also, you need a .env file for OPENAI_API_KEY and PINECOINE_API_KEY:
 ```bash
OPENAI_API_KEY = 'YOUR_API_KEY'
PINECONE_API_KEY = 'YOUR_API_KEY'
PINECONE_ENVIRONMENT = 'YOUR_ENVIRONMENT'
 ```
