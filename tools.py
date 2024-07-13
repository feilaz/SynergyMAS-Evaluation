# tools.py
from clingo import Control
import json
from langchain.tools import tool 
from typing import Annotated
from rag import rag_MRA, rag_PD, rag_SM
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from nltk.stem import WordNetLemmatizer
import nltk
from typing import Dict, Union, Literal
import re
from clingo import Control, MessageCode
from langchain.output_parsers import PydanticOutputParser

KB_FILE = "C:/Users/adam/Desktop/langchain/knowledge_base.json"
    
VALID_CONTEXTS = ["Client", "Product", "Market", "Design", "Sales", "Strategy"]

class Context(BaseModel):
    """Context for categorizing sentences."""
    context: Literal["Client", "Product", "Market", "Design", "Sales", "Strategy"]

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

def categorize_sentences(sentence: str) -> str:
    llm_categorizer = llm.with_structured_output(Context)
    categorize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Categorize the given sentence into the appropriate context."),
        ("human", "Given Sentence: {sentence}\n\nPlease categorize this sentence into one of the following contexts: Client, Product, Market, Design, Sales, Strategy")
    ])
    categorize_chain = categorize_prompt | llm_categorizer
    result = categorize_chain.invoke({"sentence": sentence})
    return result['context']

@tool
def add_to_kb(input_string: Annotated[str, "Semicolon-separated sentences"]) -> Annotated[str, "Confirmation message"]:
    """
    Adds knowledge to the Product Development Team's database from a string of sentences.
    Categorizes each sentence into appropriate contexts. Use for new information, decisions,
    or insights. Maintains shared understanding among agents. Use regularly to keep the
    knowledge base current and comprehensive.
    """
    sentences = [s.strip() for s in input_string.split(';') if s.strip()]
    
    # Process and lemmatize sentences
    lemmatizer = WordNetLemmatizer()
    processed_entries = {context: [] for context in VALID_CONTEXTS}

    for sentence in sentences:
        # Categorize each sentence individually
        category = categorize_sentences(sentence)
        
        words = nltk.word_tokenize(sentence)
        lemmatized_words = []
        for word, tag in nltk.pos_tag(words):
            if tag.startswith('NN'):
                lemmatized_words.append(lemmatizer.lemmatize(word, pos='n'))
            elif tag.startswith('VB'):
                lemmatized_words.append(lemmatizer.lemmatize(word, pos='v'))
            elif tag.startswith('JJ'):
                lemmatized_words.append(lemmatizer.lemmatize(word, pos='a'))
            elif tag.startswith('RB'):
                lemmatized_words.append(lemmatizer.lemmatize(word, pos='r'))
            else:
                lemmatized_words.append(word)
        
        processed_sentence = ' '.join(lemmatized_words)
        processed_entries[category].append(processed_sentence)

    # Load existing knowledge base
    try:
        with open(KB_FILE, "r") as f:
            kb = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        kb = {context: [] for context in VALID_CONTEXTS}

    # Update knowledge base with new processed entries
    for context, sentences in processed_entries.items():
        kb[context].extend(sentences)

    # Save updated knowledge base
    with open(KB_FILE, "w") as f:
        json.dump(kb, f, indent=2)
    
    return f"Processed and categorized {len(sentences)} sentences and added to knowledge base."


class QueryCategories(BaseModel):
    """Query categories for identifying relevant knowledge."""
    categories: str = Field(description="Comma-separated list of relevant categories (can be empty)")

class ASPInput(BaseModel):
    """Input for generating ASP representation."""
    asp_representation: str = Field(description="ASP query representation")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def validate_asp_syntax(asp_representation: str) -> bool:
    # Split the representation into individual statements
    statements = asp_representation.strip().split('\n')
    
    # Define regex patterns for different ASP constructs
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

class ClingoErrorHandler:
    def __init__(self):
        self.error_messages = []

    def on_message(self, code, msg):
        if code in [MessageCode.RuntimeError, MessageCode.SyntaxError, MessageCode.LogicError]:
            self.error_messages.append(f"{code}: {msg}")

def preprocess_asp(asp_representation: str) -> str:
    """Preprocess ASP representation to fix common issues."""
    # Ensure each statement ends with a period
    asp_representation = re.sub(r'([^.\s])(\s*\n)', r'\1.\2', asp_representation)
    
    # Remove any double periods
    asp_representation = re.sub(r'\.\.', '.', asp_representation)
    
    # Ensure proper spacing around ':-'
    asp_representation = re.sub(r'\s*:-\s*', ' :- ', asp_representation)
    
    return asp_representation

def generate_asp_representation(query: str, relevant_kb: dict, llm, asp_parser, max_attempts: int = 3) -> str:
    asp_prompt = ChatPromptTemplate.from_messages([
        ("system", asp_translation_prompt),
        ("human", """Query: {query}\n\nRelevant Knowledge: {relevant_kb}\n\nProvide an ASP representation for this query. 
         Ensure each statement is on a separate line and validate the ASP syntax before finalizing the output.\n\n{format_instructions}""")
    ])

    for attempt in range(max_attempts):
        asp_chain = asp_prompt | llm
        asp_result = asp_chain.invoke({
            "query": query,
            "relevant_kb": json.dumps(relevant_kb, indent=2),
            "format_instructions": asp_parser.get_format_instructions()
        })
        
        try:
            parsed_asp_result = asp_parser.parse(asp_result.content)
            if validate_asp_syntax(parsed_asp_result.asp_representation):
                return parsed_asp_result.asp_representation
            else:
                print(f"Attempt {attempt + 1}: Invalid ASP syntax. Retrying...")
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error parsing ASP output: {e}")
    
    raise ValueError("Failed to generate valid ASP representation after maximum attempts")

@tool
def solve_with_clingo(query: str) -> str:
    """Solve a logical query using Clingo and the knowledge database."""
    # Step 1: Determine relevant categories
    category_parser = PydanticOutputParser(pydantic_object=QueryCategories)
    category_prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the query and determine relevant categories. The list can be empty if no external knowledge is needed."),
        ("human", "Query: {query}\n\nPossible Categories: Client, Product, Market, Design, Sales, Strategy\n\nReturn relevant categories as a comma-separated list (or empty string if none apply).\n\n{format_instructions}")
    ])
    
    category_chain = category_prompt | llm
    category_result = category_chain.invoke({
        "query": query,
        "format_instructions": category_parser.get_format_instructions()
    })
    
    try:
        parsed_category_result = category_parser.parse(category_result.content)
        categories = [cat.strip() for cat in parsed_category_result.categories.split(',') if cat.strip()]
    except Exception as e:
        print(f"Error parsing category output: {e}")
        print(f"Raw category output: {category_result.content}")
        return "Error in processing the query categories"

    print(f"Relevant categories: {categories}")

    # Step 2: Generate ASP representation
    with open(KB_FILE, "r") as f:
        kb_data = json.load(f)
    
    relevant_kb = {cat: kb_data.get(cat, []) for cat in categories}
    
    asp_parser = PydanticOutputParser(pydantic_object=ASPInput)

    try:
        asp_representation = generate_asp_representation(query, relevant_kb, llm, asp_parser)
        print(relevant_kb)
        print(asp_representation)
    except ValueError as e:
        return str(e)

    print(f"Final ASP representation:\n{asp_representation}")

    # Step 3: Use Clingo to solve the ASP representation
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
                print("Attempting to improve ASP representation...")
                improvement_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Improve the ASP representation to fix the Clingo error. Ensure each statement is on a separate line and follows ASP syntax rules."),
                    ("human", f"Original ASP:\n{asp_representation}\nClingo Errors:\n{', '.join(error_handler.error_messages)}\n\nProvide an improved ASP representation that addresses these errors.")
                ])
                improvement_chain = improvement_prompt | llm
                improvement_result = improvement_chain.invoke({})
                asp_representation = preprocess_asp(improvement_result.content)
                print(f"Improved ASP representation:\n{asp_representation}")
                error_handler.error_messages.clear()
            else:
                return "Error in processing the ASP representation after maximum attempts"

    solution = []
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            solution.append(str(model))
    
    print(f"Solutions found: {solution}")

    if not solution or solution == [""]:
        return "No solution found"

    # Step 4: Interpret the solution
    interpretation_prompt = ChatPromptTemplate.from_messages([
        ("system", "Interpret the Clingo solution and provide a human-readable answer to the original query."),
        ("human", "Original query: {query}\nClingo solution: {solution}\n\nPlease provide a clear, concise interpretation of this solution as an answer to the original query.")
    ])
    
    interpretation_chain = interpretation_prompt | llm
    interpretation_result = interpretation_chain.invoke({
        "query": query,
        "solution": "; ".join(solution)
    })

    return interpretation_result.content

asp_translation_prompt = """
Generate an ASP (Answer Set Programming) representation for the given query using the provided knowledge.
Follow these guidelines strictly to ensure proper syntax:
1. Use lowercase for predicate names and constants.
2. Use underscores to separate words in predicate names and constants.
3. End each statement with a period (.).
4. Use 'not' for negation.
5. Use ':-' for constraints.
6. Put each statement on a separate line.
7. Validate the ASP syntax before finalizing the output.

Examples:

Example 1:
Query: Who is responsible for product design?
Relevant Knowledge:
- Taylor is responsible for product design.
ASP:
responsible_for_product_design(taylor).

Example 2:
Query: What are the key features of our product?
Relevant Knowledge:
- The product has user-friendly interface, high performance, and extended battery life.
ASP:
product_feature(user_friendly_interface).
product_feature(high_performance).
product_feature(extended_battery_life).

Example 3:
Query: Is our product more cost-effective than our competitors?
Relevant Knowledge:
- Our product costs $199.
- Competitor A's product costs $249.
- Competitor B's product costs $299.
ASP:
product_price(our_product, 199).
competitor_price(competitor_a, 249).
competitor_price(competitor_b, 299).
more_cost_effective_than(X, Y) :- product_price(X, PX), competitor_price(Y, PY), PX < PY.

Example 4:
Query: What are the constraints on our product features?
Relevant Knowledge:
- The product should not exceed 200 grams.
- The product should have a battery life of at least 24 hours.
ASP:
product_weight_constraint(200).
battery_life_constraint(24).
:- product_weight(W), W > 200.
:- battery_life(H), H < 24.

Example 5:
Query: Can we list the features that are not related to the user interface?
Relevant Knowledge:
- The product has user-friendly interface, high performance, and extended battery life.
ASP:
product_feature(user_friendly_interface).
product_feature(high_performance).
product_feature(extended_battery_life).
not_user_interface_related(X) :- product_feature(X), not user_friendly_interface(X).

Ensure the ASP representation adheres to the given guidelines and examples.
"""