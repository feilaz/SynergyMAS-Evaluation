# tools.py
from clingo import Control, Symbol
import json
from langchain.tools import tool 

@tool
def hidden_message(first_number: int, second_number: int):
    """returns hidden message"""
    return "kotek na plotek"

@tool
def use_clingo(program, query=None, opt=False):
    """
    Execute a Clingo program and return the results.
    
    Args:
    program (str): The Clingo program to execute.
    query (str, optional): A specific predicate to query from the results.
    opt (bool): If true, only optimal answer sets are returned.
    
    Returns:
    str: A JSON string containing the parsed results of the Clingo execution.
    """
    ctl = Control(['0', '--warn=none', '--opt-mode=optN', '-t', '4'])
    
    try:
        ctl.add("base", [], program)
        ctl.ground([("base", [])])
        
        models = []
        if opt:
            ctl.solve(on_model=lambda model: models.append(model.symbols(shown=True)) if model.optimality_proven else None)
        else:
            ctl.solve(on_model=lambda model: models.append(model.symbols(shown=True)))
        
        results = [[str(atom) for atom in model] for model in models]
        
        if query:
            filtered_results = []
            for result in results:
                filtered_result = [atom for atom in result if atom.startswith(query)]
                if filtered_result:
                    filtered_results.append(filtered_result)
            return json.dumps(filtered_results, indent=2)
        
        return json.dumps(results, indent=2)
    except RuntimeError as e:
        return json.dumps({"error": str(e)}, indent=2)