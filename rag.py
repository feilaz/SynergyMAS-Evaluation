from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import tool 

vectorstores = {}

def create_vectorstore(directory: str, name: str):
    """Create a vectorstore if it doesn't exist, otherwise return the existing one."""
    if name not in vectorstores:
        docs = []
        for file_path in [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]:
            loader = TextLoader(file_path)
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs)

        vectorstores[name] = Chroma.from_documents(
            documents=doc_splits,
            collection_name=f"rag-{name}-chroma",
            embedding=OpenAIEmbeddings(),
        )
    
    return vectorstores[name].as_retriever()


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def check_keyword_relevance(question: str, document: str) -> bool:
    """
    Check if the document contains keywords from the question.
    """
    question_words = set(question.lower().split())
    document_words = set(document.lower().split())
    common_words = question_words.intersection(document_words)
    return len(common_words) > 1  # Adjust this threshold as needed

def rag(retriever, question: str, llm_model: str = "gpt-3.5-turbo") -> str:
    """
    Core RAG function that performs the retrieval and generation.
    """
    docs = retriever.get_relevant_documents(question)
    
    if not docs:
        return "RETRIEVAL_FAILED: No relevant documents found. Consider rephrasing the question or exploring a different topic."
    
    relevant_docs = [doc for doc in docs if check_keyword_relevance(question, doc.page_content)]
    
    if not relevant_docs:
        return "IRRELEVANT_ANSWER: The retrieved information doesn't seem directly relevant to the question. Consider rephrasing or asking a different question."
    
    llm = ChatOpenAI(model_name=llm_model, temperature=0, streaming=True)
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = prompt | llm | StrOutputParser()
    
    formatted_docs = "\n\n".join(doc.page_content for doc in relevant_docs)
    response = rag_chain.invoke({"context": formatted_docs, "question": question})
    
    print(f"RAG RESPONSE: {response}")
    print()

    return f"RELEVANT_INFORMATION: {response}"

michael_data_path = "C:/Users/adam/Desktop/langchain"

@tool
def rag_michael(question: str) -> str:
    """
    Perform Retrieval-Augmented Generation (RAG) to answer a question.
    If the question is not suitable for retrieval or the answer is irrelevant, inform the agent.
    
    Args:
        question (str): The question to be answered.
    
    Returns:
        str: The generated answer or an information message.
    """
    retriever = create_vectorstore(michael_data_path, "michael-jackson")
    return rag(retriever, question)

