from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
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

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def create_retrieval_grader():
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    system = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    return grade_prompt | structured_llm_grader

def create_question_rewriter():
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    
    system = """You a question re-writer that converts an input question to a better version that is optimized
    for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
    
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    
    return re_write_prompt | llm | StrOutputParser()

def rag(retriever, question: str, llm_model: str = "gpt-3.5-turbo") -> str:
    """
    Modified RAG function incorporating CRAG elements with web search fallback.
    """
    retrieval_grader = create_retrieval_grader()
    question_rewriter = create_question_rewriter()
    web_search_tool = TavilySearchResults(k=3)
    
    docs = retriever.get_relevant_documents(question)
    
    relevant_docs = []
    used_web_search = False
    
    if docs:
        for doc in docs:
            grade = retrieval_grader.invoke({"question": question, "document": doc.page_content})
            if grade.binary_score == "yes":
                relevant_docs.append(doc)
    
    if not relevant_docs:
        used_web_search = True
        improved_question = question_rewriter.invoke({"question": question})
        web_results = web_search_tool.invoke({"query": improved_question})
        web_content = "\n".join([d["content"] for d in web_results])
        relevant_docs.append(Document(page_content=web_content))
    
    if not relevant_docs:
        return "IRRELEVANT_ANSWER: No relevant information found. Please try rephrasing your question or asking about a different topic."
    
    llm = ChatOpenAI(model_name=llm_model, temperature=0, streaming=True)
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = prompt | llm | StrOutputParser()
    
    formatted_docs = "\n\n".join(doc.page_content for doc in relevant_docs)
    response = rag_chain.invoke({"context": formatted_docs, "question": question})
    
    source_message = "WEB SEARCH" if used_web_search else "RAG DATABASE"
    print(f"RAG RESPONSE (Source: {source_message}): {response}")
    print()

    return f"RELEVANT_INFORMATION: {response}"

@tool
def rag_MRA(question: str) -> str:
    """
    Perform Corrective Retrieval-Augmented Generation (RAG) for the Market Research Analyst (MRA) agent.
    This function first attempts to access a specialized knowledge base containing market analysis data, customer needs, and industry trends.
    If relevant information is not found in the database, it will search for information online.

    Use this function to answer questions related to:
    - Market trends and analysis
    - Customer needs and preferences
    - Competitive landscape
    - Industry forecasts and projections

    Args:
        question (str): A market research related question to be answered.

    Returns:
        str: The generated answer based on the MRA's knowledge base or online search, or an information message if retrieval fails or the answer is irrelevant.
    """
    retriever = create_vectorstore(MRA_data_path, "MRA")
    return rag(retriever, question)

@tool
def rag_PD(question: str) -> str:
    """
    Perform Corrective Retrieval-Augmented Generation (RAG) for the Product Designer (PD) agent.
    This function first attempts to access a specialized knowledge base containing information on product design, usability, and aesthetics.
    If relevant information is not found in the database, it will search for information online.

    Use this function to answer questions related to:
    - Product design principles and best practices
    - User experience (UX) and user interface (UI) design
    - Design trends and innovations
    - Ergonomics and human factors in design

    Args:
        question (str): A product design related question to be answered.

    Returns:
        str: The generated answer based on the PD's knowledge base or online search, or an information message if retrieval fails or the answer is irrelevant.
    """
    retriever = create_vectorstore(PD_data_path, "PD")
    return rag(retriever, question)

@tool
def rag_SM(question: str) -> str:
    """
    Perform Corrective Retrieval-Augmented Generation (RAG) for the Sales Manager (SM) agent.
    This function first attempts to access a specialized knowledge base containing information on sales strategies, team management, and customer relationships.
    If relevant information is not found in the database, it will search for information online.

    Use this function to answer questions related to:
    - Sales strategies and techniques
    - Customer relationship management
    - Sales team management and motivation
    - Market positioning and product pricing

    Args:
        question (str): A sales management related question to be answered.

    Returns:
        str: The generated answer based on the SM's knowledge base or online search, or an information message if retrieval fails or the answer is irrelevant.
    """
    retriever = create_vectorstore(SM_data_path, "SM")
    return rag(retriever, question)