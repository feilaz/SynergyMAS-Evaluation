import os
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import Tool

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class RAGSystem:
    def __init__(self, chroma_db_dir, mra_data_path, pd_data_path, sm_data_path, model_name="gpt-3.5-turbo"):
        self.CHROMA_DB_DIR = chroma_db_dir
        self.MRA_data_path = mra_data_path
        self.PD_data_path = pd_data_path
        self.SM_data_path = sm_data_path
        self.vectorstores = {}
        self.llm = ChatOpenAI(model=model_name)
        
    def create_vectorstore(self, directory: str, name: str):
        """Create a vectorstore if it doesn't exist, otherwise return the existing one."""
        chroma_db_path = Path(self.CHROMA_DB_DIR) / f"{name}_chroma_db"
        
        if name not in self.vectorstores:
            if chroma_db_path.exists() and any(chroma_db_path.iterdir()):
                print(f"Loading existing vectorstore for {name}")
                self.vectorstores[name] = Chroma(
                    persist_directory=str(chroma_db_path),
                    embedding_function=OpenAIEmbeddings(),
                    collection_name=f"rag-{name}-chroma"
                )
            else:
                print(f"Creating new vectorstore for {name}")
                docs = []
                for file_path in Path(directory).glob('*.txt'):
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    docs.extend(loader.load())

                text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=100, chunk_overlap=50
                )
                doc_splits = text_splitter.split_documents(docs)

                chroma_db_path.mkdir(parents=True, exist_ok=True)

                self.vectorstores[name] = Chroma.from_documents(
                    documents=doc_splits,
                    collection_name=f"rag-{name}-chroma",
                    embedding=OpenAIEmbeddings(),
                    persist_directory=str(chroma_db_path)
                )
                
                self.vectorstores[name].persist()

        return self.vectorstores[name].as_retriever()

    def create_retrieval_grader(self):
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        
        system = """You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])
        
        return grade_prompt | structured_llm_grader

    def create_question_rewriter(self):
        system = """You a question re-writer that converts an input question to a better version that is optimized
        for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
        
        re_write_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ])
        
        return re_write_prompt | self.llm | StrOutputParser()

    def rag(self, retriever, question: str) -> str:
        """
        Modified RAG function incorporating CRAG elements with web search fallback.
        """
        retrieval_grader = self.create_retrieval_grader()
        question_rewriter = self.create_question_rewriter()
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
        
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = prompt | self.llm | StrOutputParser()
        
        formatted_docs = "\n\n".join(doc.page_content for doc in relevant_docs)
        response = rag_chain.invoke({"context": formatted_docs, "question": question})
        
        source_message = "WEB SEARCH" if used_web_search else "RAG DATABASE"
        print(f"RAG RESPONSE (Source: {source_message}): {response}")
        print()

        return f"RELEVANT_INFORMATION: {response}"

    def rag_MRA(self, question: str) -> str:
        """
        Perform Corrective Retrieval-Augmented Generation (RAG) for the Market Research Analyst (MRA) agent.
        This function first attempts to access a specialized knowledge base containing market analysis data, customer needs, and industry trends.
        If relevant information is not found in the database, it will search for information online.

        Use this function to answer questions related to:
        - Market trends and analysis
        - Customer needs and preferences
        - Competitive landscape
        - Industry forecasts and projections
        """
        retriever = self.create_vectorstore(self.MRA_data_path, "MRA")
        return self.rag(retriever, question)

    def rag_PD(self, question: str) -> str:
        """
        Perform Corrective Retrieval-Augmented Generation (RAG) for the Product Designer (PD) agent.
        This function first attempts to access a specialized knowledge base containing information on product design, usability, and aesthetics.
        If relevant information is not found in the database, it will search for information online.

        Use this function to answer questions related to:
        - Product design principles and best practices
        - User experience (UX) and user interface (UI) design
        - Design trends and innovations
        - Ergonomics and human factors in design
        """
        retriever = self.create_vectorstore(self.PD_data_path, "PD")
        return self.rag(retriever, question)

    def rag_SM(self, question: str) -> str:
        """
        Perform Corrective Retrieval-Augmented Generation (RAG) for the Sales Manager (SM) agent.
        This function first attempts to access a specialized knowledge base containing information on sales strategies, team management, and customer relationships.
        If relevant information is not found in the database, it will search for information online.

        Use this function to answer questions related to:
        - Sales strategies and techniques
        - Customer relationship management
        - Sales team management and motivation
        - Market positioning and product pricing
        """
        retriever = self.create_vectorstore(self.SM_data_path, "SM")
        return self.rag(retriever, question)
    
def create_rag_tool(func: callable) -> Tool:
    return Tool(
        name=func.__name__,
        description=func.__doc__,
        func=func
    )