from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def ask_question(model, retriever, question: str):
    template_str = """
    Context: {context}

    Question: {question}
    """
    prompt = PromptTemplate.from_template(template_str)
    parser = StrOutputParser()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )
    response = chain.invoke(question)
    return response
