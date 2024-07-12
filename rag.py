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

CHROMA_DB_DIR = "C:/Users/adam/Desktop/langchain"
MRA_data_path = "C:/Users/adam/Desktop/product team/rag/MRA"
PD_data_path = "C:/Users/adam/Desktop/product team/rag/PD"
SM_data_path = "C:/Users/adam/Desktop/product team/rag/SM"


def create_vectorstore(directory: str, name: str):
    """Create a vectorstore if it doesn't exist, otherwise return the existing one."""
    chroma_db_path = os.path.join(CHROMA_DB_DIR, f"{name}_chroma_db")
    
    if name not in vectorstores:
        # Check if the vectorstore already exists on disk
        if os.path.exists(chroma_db_path) and os.listdir(chroma_db_path):
            print(f"Loading existing vectorstore for {name}")
            vectorstores[name] = Chroma(
                persist_directory=chroma_db_path,
                embedding_function=OpenAIEmbeddings(),
                collection_name=f"rag-{name}-chroma"
            )
        else:
            print(f"Creating new vectorstore for {name}")
            docs = []
            for file_path in [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]:
                loader = TextLoader(file_path, encoding="utf-8")
                docs.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=100, chunk_overlap=50
            )
            doc_splits = text_splitter.split_documents(docs)

            # Set the custom directory for Chroma's database files
            os.makedirs(CHROMA_DB_DIR, exist_ok=True)

            vectorstores[name] = Chroma.from_documents(
                documents=doc_splits,
                collection_name=f"rag-{name}-chroma",
                embedding=OpenAIEmbeddings(),
                persist_directory=chroma_db_path
            )
            
            # Persist the data
            vectorstores[name].persist()

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

@tool
def rag_MRA(question: str) -> str:
    """
    Perform Retrieval-Augmented Generation (RAG) for the Market Research Analyst (MRA) agent.
    This function accesses a specialized knowledge base containing market analysis data, customer needs, and industry trends.
    
    Use this function to answer questions related to:
    - Market trends and analysis
    - Customer needs and preferences
    - Competitive landscape
    - Industry forecasts and projections

    Args:
        question (str): A market research related question to be answered.
    
    Returns:
        str: The generated answer based on the MRA's knowledge base, or an information message if retrieval fails or the answer is irrelevant.
    """
    retriever = create_vectorstore(MRA_data_path, "MRA")
    return rag(retriever, question)

@tool
def rag_PD(question: str) -> str:
    """
    Perform Retrieval-Augmented Generation (RAG) for the Product Designer (PD) agent.
    This function accesses a specialized knowledge base containing information on product design, usability, and aesthetics.
    
    Use this function to answer questions related to:
    - Product design principles and best practices
    - User experience (UX) and user interface (UI) design
    - Design trends and innovations
    - Ergonomics and human factors in design

    Args:
        question (str): A product design related question to be answered.
    
    Returns:
        str: The generated answer based on the PD's knowledge base, or an information message if retrieval fails or the answer is irrelevant.
    """
    retriever = create_vectorstore(PD_data_path, "PD")
    return rag(retriever, question)

@tool
def rag_SM(question: str) -> str:
    """
    Perform Retrieval-Augmented Generation (RAG) for the Sales Manager (SM) agent.
    This function accesses a specialized knowledge base containing information on sales strategies, team management, and customer relationships.
    
    Use this function to answer questions related to:
    - Sales strategies and techniques
    - Customer relationship management
    - Sales team management and motivation
    - Market positioning and product pricing

    Args:
        question (str): A sales management related question to be answered.
    
    Returns:
        str: The generated answer based on the SM's knowledge base, or an information message if retrieval fails or the answer is irrelevant.
    """
    retriever = create_vectorstore(SM_data_path, "SM")
    return rag(retriever, question)