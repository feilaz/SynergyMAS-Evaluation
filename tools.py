# tools.py
import json
import re
import os
from typing import Dict, Any
from pydantic import Field, BaseModel
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains.graph_qa.prompts import CYPHER_QA_PROMPT
from clingo import Control, MessageCode
from prompts import ASP_TRANSLATION_PROMPT

class KnowledgeBaseSystem:
    def __init__(self, neo4j_url, neo4j_username, neo4j_password):
        self.graph = Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password)
        self.graph.query("MATCH (n) DETACH DELETE n")
        
        self.asp_llm = ChatOpenAI(model="gpt-4o")
        self.qa_llm = ChatOpenAI(model="gpt-4o")
        self.graph_transformer_llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        
        self.llm_transformer = LLMGraphTransformer(
            llm=self.graph_transformer_llm, 
            node_properties=["description"],
            relationship_properties=["description"]
        )
        
        self.cypher_chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            cypher_llm=ChatOpenAI(temperature=0, model="gpt-4o"),
            qa_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
            validate_cypher=True,
            return_direct=True,
            verbose=True,
            top_k=20
        )

    def process_text(self, text: str):
        doc = Document(page_content=text)
        graph_documents = self.llm_transformer.convert_to_graph_documents([doc])
        return graph_documents[0] if graph_documents else None

    def add_to_kb(self, text):
        graph_document = self.process_text(text)
        self.graph.add_graph_documents([graph_document], baseEntityLabel=True, include_source=True)
        return graph_document

    def query_knowledge_base(self, query: str):
        """
        Query the knowledge base to retrieve data and process complex logical questions.
        """
        self.graph.refresh_schema()
        graph_data = self.cypher_chain.invoke({"query": query})

        if not graph_data["result"]:
            result = {"result": "No relevant data was found in the database.", "source": "graph_database"}
        else:
            result_complexity = self._assess_result_complexity(graph_data["result"])
            if result_complexity == "simple":
                qa_chain = CYPHER_QA_PROMPT | self.qa_llm
                answer = qa_chain.invoke({"question": query, "context": graph_data})
                result = {"result": answer, "source": "graph_database"}
            else:
                clingo_result = self._solve_with_clingo(query, graph_data)
                result = {"result": clingo_result, "source": "clingo_solver with graph_database"}

        self._save_to_file({
            "question": query,
            "retrieved_data": graph_data["result"],
            "result_complexity": result_complexity,
            "answer": result["result"],
            "source": result["source"]
        })

        return result

    def _assess_result_complexity(self, result: str) -> str:
        return "complex" if len(result) >= 10 else "simple"

    def _save_to_file(self, data: dict):
        filename = "C:/Users/adam/Desktop/knowledge_base_queries.json"
        try:
            with open(filename, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = []
        existing_data.append(data)
        with open(filename, 'w') as file:
            json.dump(existing_data, file, indent=2)
        print(f"Query data saved to {filename}")

    def _solve_with_clingo(self, query: str, graph_data: Dict[str, Any]) -> str:
        asp_parser = PydanticOutputParser(pydantic_object=ASPInput)
        try:
            asp_representation = self._generate_asp_representation(query, graph_data, asp_parser)
        except ValueError as e:
            return str(e)

        print(f"Final ASP representation:\n{asp_representation}")

        ctl = Control()
        error_handler = ClingoErrorHandler()
        
        max_clingo_attempts = 3
        for attempt in range(max_clingo_attempts):
            try:
                ctl.register_observer(error_handler)
                ctl.add("base", [], asp_representation)
                ctl.ground([("base", [])])
                break
            except RuntimeError as e:
                print(f"Attempt {attempt + 1}: Error in Clingo processing: {str(e)}")
                if attempt < max_clingo_attempts - 1:
                    asp_representation = self._improve_asp_representation(asp_representation, error_handler.error_messages)
                else:
                    return "Error in processing the ASP representation after maximum attempts"

        solution = []
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                solution.append(str(model))
        
        print(f"Solutions found: {solution}")

        if not solution or solution == [""]:
            return "No solution found"

        return self._interpret_solution(query, solution)

    def _generate_asp_representation(self, query: str, graph_data: Dict[str, Any], asp_parser, max_attempts: int = 3) -> str:
        asp_prompt = ChatPromptTemplate.from_messages([
            ("system", ASP_TRANSLATION_PROMPT),
            ("human", """Query: {query}\n\nRelevant Knowledge: {relevant_kb}\n\nProvide an ASP representation for this query. 
             Ensure each statement is on a separate line and validate the ASP syntax before finalizing the output.\n\n{format_instructions}""")
        ])

        for attempt in range(max_attempts):
            asp_chain = asp_prompt | self.asp_llm
            asp_result = asp_chain.invoke({
                "query": query,
                "relevant_kb": json.dumps(graph_data, indent=2),
                "format_instructions": asp_parser.get_format_instructions()
            })
            
            try:
                parsed_asp_result = asp_parser.parse(asp_result.content)
                if self._validate_asp_syntax(parsed_asp_result.asp_representation):
                    return parsed_asp_result.asp_representation
                else:
                    print(f"Attempt {attempt + 1}: Invalid ASP syntax. Retrying...")
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error parsing ASP output: {e}")
        
        raise ValueError("Failed to generate valid ASP representation after maximum attempts")

    def _improve_asp_representation(self, asp_representation: str, error_messages: list) -> str:
        improvement_prompt = ChatPromptTemplate.from_messages([
            ("system", "Improve the ASP representation to fix the Clingo error. Ensure each statement is on a separate line and follows ASP syntax rules."),
            ("human", f"Original ASP:\n{asp_representation}\nClingo Errors:\n{', '.join(error_messages)}\n\nProvide an improved ASP representation that addresses these errors.")
        ])
        improvement_chain = improvement_prompt | self.asp_llm
        improvement_result = improvement_chain.invoke({})
        return self._preprocess_asp(improvement_result.content)

    def _interpret_solution(self, query: str, solution: list) -> str:
        interpretation_prompt = ChatPromptTemplate.from_messages([
            ("system", "Interpret the Clingo solution and provide a human-readable answer to the original query."),
            ("human", "Original query: {query}\nClingo solution: {solution}\n\nPlease provide a clear, concise interpretation of this solution as an answer to the original query.")
        ])
        
        interpretation_chain = interpretation_prompt | self.qa_llm
        interpretation_result = interpretation_chain.invoke({
            "query": query,
            "solution": "; ".join(solution)
        })

        return interpretation_result.content

    def _validate_asp_syntax(self, asp_representation: str) -> bool:
        statements = asp_representation.strip().split('\n')
        fact_pattern = r'^[a-z_]+\([a-z0-9_, ]+\)\.$'
        rule_pattern = r'^[a-z_]+\([a-z0-9_, ]+\)\s*:-\s*[a-z_]+\([a-z0-9_, ]+\)(\s*,\s*[a-z_]+\([a-z0-9_, ]+\))*\.$'
        constraint_pattern = r'^:-\s*[a-z_]+\([a-z0-9_, ]+\)(\s*,\s*[a-z_]+\([a-z0-9_, ]+\))*\.$'
        
        for statement in statements:
            statement = statement.strip()
            if not (re.match(fact_pattern, statement) or 
                    re.match(rule_pattern, statement) or 
                    re.match(constraint_pattern, statement)):
                print(f"Invalid ASP statement: {statement}")
                return False
        return True

    def _preprocess_asp(self, asp_representation: str) -> str:
        asp_representation = re.sub(r'([^.\s])(\s*\n)', r'\1.\2', asp_representation)
        asp_representation = re.sub(r'\.\.', '.', asp_representation)
        asp_representation = re.sub(r'\s*:-\s*', ' :- ', asp_representation)
        return asp_representation

class ASPInput(BaseModel):
    """Input for generating ASP representation."""
    asp_representation: str = Field(description="ASP query representation")

class ClingoErrorHandler:
    def __init__(self):
        self.error_messages = []

    def on_message(self, code, msg):
        if code in [MessageCode.RuntimeError, MessageCode.SyntaxError, MessageCode.LogicError]:
            self.error_messages.append(f"{code}: {msg}")

def create_kb_tool(kb_system: KnowledgeBaseSystem) -> Tool:
    return Tool(
        name="query_knowledge_base",
        description="Query the knowledge base to retrieve data and process complex logical questions.",
        func=kb_system.query_knowledge_base
    )

