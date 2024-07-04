from clingo import Control, Symbol
from clingo.symbol import SymbolType
import json

class ClingoTool:
    def __init__(self):
        self.ctl = Control(['0', '--warn=none', '--opt-mode=optN', '-t', '4'])
        self.knowledge_base = ""

    def add_to_knowledge_base(self, facts_and_rules):
        """Add facts and rules to the knowledge base."""
        self.knowledge_base += facts_and_rules + "\n"

    def run(self, program, query=None, opt=False):
        full_program = self.knowledge_base + "\n" + program
        try:
            self.ctl.cleanup()
            self.ctl.add("base", [], full_program)
            self.ctl.ground([("base", [])])
            
            models = []
            if opt:
                self.ctl.solve(on_model=lambda model: models.append(model.symbols(shown=True)) if model.optimality_proven else None)
            else:
                self.ctl.solve(on_model=lambda model: models.append(model.symbols(shown=True)))
            
            results = [[str(atom) for atom in model] for model in models]
            
            if query:
                filtered_results = []
                for result in results:
                    filtered_result = [atom for atom in result if atom.startswith(query)]
                    if filtered_result:
                        filtered_results.append(filtered_result)
                return filtered_results
            
            return results
        except RuntimeError as e:
            return f"Clingo Error: {str(e)}"

    def parse_result(self, result):
        if isinstance(result, str) and result.startswith("Clingo Error"):
            return result
        elif isinstance(result, list):
            return [self.parse_result(item) for item in result]
        else:
            return str(result)

clingo_tool = ClingoTool()

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
    print(f"Clingo input:\n{program}")
    result = clingo_tool.run(program, query, opt)
    parsed_result = clingo_tool.parse_result(result)
    print(f"Clingo output:\n{json.dumps(parsed_result, indent=2)}")
    return json.dumps(parsed_result, indent=2)

def add_to_kb(facts_and_rules):
    """
    Add facts and rules to the Clingo knowledge base.
    """
    clingo_tool.add_to_knowledge_base(facts_and_rules)
    return "Knowledge base updated successfully."
