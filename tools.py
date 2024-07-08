# tools.py
from clingo import Control, Symbol
import json
from langchain.tools import tool 
from typing import Annotated, List
from rag import rag_michael
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

KB_FILE = "C:/Users/adam/Desktop/langchain/knowledge_base.txt"


class ASPCategorizedTranslation(BaseModel):
    """ASP translation of natural language categorized into different components."""
    concepts: str = Field(default="", description="ASP sentences containing concepts or classes")
    relationships: str = Field(default="", description="ASP sentences containing relationships")
    attributes: str = Field(default="", description="ASP sentences containing attributes or properties")
    instances: str = Field(default="", description="ASP sentences containing instances")
    rules: str = Field(default="", description="ASP sentences containing rules and constraints")

class ASPInput(BaseModel):
    """ASP input for Clingo solver."""
    facts: str = Field(default="", description="ASP facts relevant to the query as a single string, separated by semicolons")
    rules: str = Field(default="", description="ASP rules relevant to the query as a single string, separated by semicolons")
    query: str = Field(default="", description="ASP representation of the query")
    



asp_input_parser = PydanticOutputParser(pydantic_object=ASPInput)
asp_categorized_parser = PydanticOutputParser(pydantic_object=ASPCategorizedTranslation)


@tool
def add_to_kb(knowledge: Annotated[str, "The knowledge to add to the database"]) -> str:
    """
    Add knowledge to the knowledge database.
    
    Args:
    knowledge (str): The knowledge to add, in natural language.
    
    Returns:
    str: A confirmation message.
    """
    # First, translate the natural language to categorized ASP
    categorized_asp = translate_to_categorized_ASP(knowledge)
    
    # Create a dictionary to store the categorized ASP
    entry = {
        "original": knowledge,
        "asp": {
            "concepts": categorized_asp.concepts,
            "relationships": categorized_asp.relationships,
            "attributes": categorized_asp.attributes,
            "instances": categorized_asp.instances,
            "rules": categorized_asp.rules
        }
    }
    
    # Append the entry to the knowledge base file
    with open(KB_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    return f"Knowledge added to database: {knowledge}"

def translate_to_categorized_ASP(knowledge: str) -> ASPCategorizedTranslation:
    """
    Translate the knowledge to a categorized ASP program.
    
    Args:
    knowledge (str): The knowledge to translate.
    
    Returns:
    ASPCategorizedTranslation: The categorized ASP program generated from the knowledge.
    """
    result_str = asp_translator_categorized.invoke({"knowledge": knowledge})
    return asp_categorized_parser.parse(result_str)

# LLM with function call
llm_to_categorized_asp = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
to_categorized_asp_translator = llm_to_categorized_asp.with_structured_output(ASPCategorizedTranslation)

# Prompt for categorized ASP translation
asp_to_categorized_system = """You are an expert in translating natural language to Answer Set Programming (ASP) for use with the Clingo solver.
Follow these guidelines:
1. Use lowercase for predicate names and constants.
2. Capitalize variables.
3. End each rule with a period.
4. Use :- for implications (if).
5. Use commas for conjunctions (and).
6. Use semicolons for disjunctions (or).
7. Use not for negation.

Translate the given knowledge into a valid ASP program, categorizing the output into:
1. Concepts (or Classes): Fundamental entities or categories.
2. Relationships: How concepts are related to each other.
3. Attributes (or Properties): Characteristics of the concepts.
4. Instances: Specific examples of the concepts.
5. Rules and Constraints: Logic and restrictions within the domain.

Provide ASP sentences for each category separately."""

to_asp_translator_prompt = ChatPromptTemplate.from_messages([
    ("system", asp_to_categorized_system),
    ("human", "Translate this to categorized ASP: {knowledge}")
])

asp_translator_categorized = to_asp_translator_prompt | to_categorized_asp_translator | asp_categorized_parser


# LLM with function call
llm_asp_input = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm_asp_input = llm_asp_input.with_structured_output(ASPInput)

# Prompt for retrieving relevant ASP and formulating Clingo input
retrieval_system = """You are an expert in Answer Set Programming (ASP) and using the Clingo solver. Your task is to:
1. Analyze the given query and the knowledge base content.
2. Extract relevant ASP facts and rules from the knowledge base.
3. Formulate the query as an ASP query.
4. Return the facts, rules, and query in a format suitable for input to the Clingo solver.

For facts and rules, separate multiple items with semicolons (;).

Ensure that the ASP input you provide is directly relevant to answering the query and is in correct ASP syntax for Clingo."""

retrieval_prompt = ChatPromptTemplate.from_messages([
    ("system", retrieval_system),
    ("human", "Query: {query}\n\nKnowledge Base:\n{kb_content}\n\nPlease provide the ASP input for Clingo, categorized into facts, rules, and the query.")
])

retrieval_agent = retrieval_prompt | structured_llm_asp_input | asp_input_parser

llm_aps_to_natural_language = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Function to translate ASP output to natural language
def translate_to_natural_language(asp_result: str) -> str:
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in translating Answer Set Programming (ASP) results to natural language. Your task is to interpret the given ASP output and explain it in clear, concise natural language."),
        ("human", "ASP Result: {asp_result}\n\nPlease translate this to natural language.")
    ])
    
    translation_chain = translation_prompt | llm_aps_to_natural_language
    
    result = translation_chain.invoke({"asp_result": asp_result})
    return result.content

@tool
def solve_with_clingo(query: Annotated[str, "The query to solve using Clingo"]) -> str:
    """
    Solve a query using Clingo with the knowledge from the database and translate the result to natural language.
    
    Args:
    query (str): The query to solve.
    
    Returns:
    str: The solution from Clingo in natural language.
    """
    # Read the entire knowledge base
    with open(KB_FILE, "r") as f:
        kb_content = f.read()
    
    # Use the retrieval agent to get ASP input for Clingo
    asp_input = retrieval_agent.invoke({
        "query": query,
        "kb_content": kb_content
    })
    
    # Create a Clingo control object
    ctl = Control()
    
    # Add facts and rules from the knowledge base
    for fact in asp_input.facts.split(';'):
        fact = fact.strip()
        if fact:
            ctl.add("base", [], fact)
    
    for rule in asp_input.rules.split(';'):
        rule = rule.strip()
        if rule:
            ctl.add("base", [], rule)
    
    # Add the query
    ctl.add("base", [], asp_input.query)
    
    # Ground the program
    ctl.ground([("base", [])])
    
    # Solve
    solution = []
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            solution.append(str(model))
    
    asp_result = "\n".join(solution) if solution else "No solution found."
    
    # Translate ASP result to natural language
    nl_result = translate_to_natural_language(asp_result)
    
    return nl_result

