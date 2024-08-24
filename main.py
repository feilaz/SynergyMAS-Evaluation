import os
import argparse
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START

from agents import create_agent, agent_node, AgentState
from config_loader import load_config, get_all_config_values
from rag import RAGSystem, create_rag_tool
from tools import KnowledgeBaseSystem, create_kb_tool
from prompts import INITIAL_MESSAGE_BASE, PM_PERSONALITY, MRA_PERSONALITY, PD_PERSONALITY, SM_PERSONALITY

class WorkflowManager:
    def __init__(self, config):
        self.config = config
        self.llm = self._setup_llm()
        self.kb_system = self._setup_kb_system()
        self.rag_system = self._setup_rag_system()
        self.toolbox = self._setup_toolbox()
        self.workflow = self._setup_workflow()

    def _setup_llm(self):
        return ChatOpenAI(model=self.config['OPENAI_MODEL'])

    def _setup_kb_system(self):
        return KnowledgeBaseSystem(
            neo4j_url=self.config['NEO4J_URL'],
            neo4j_username=self.config['NEO4J_USERNAME'],
            neo4j_password=self.config['NEO4J_PASSWORD']
        )

    def _setup_rag_system(self):
        return RAGSystem(
            chroma_db_dir=self.config['CHROMA_DB_DIR'],
            mra_data_path=self.config['MRA_DATA_PATH'],
            pd_data_path=self.config['PD_DATA_PATH'],
            sm_data_path=self.config['SM_DATA_PATH'],
            model_name=self.config['OPENAI_MODEL']
        )

    def _setup_toolbox(self):
        return [
            create_kb_tool(self.kb_system),
            create_rag_tool(self.rag_system.rag_MRA),
            create_rag_tool(self.rag_system.rag_PD),
            create_rag_tool(self.rag_system.rag_SM)
        ]

    def _setup_workflow(self):
        members = ["Sam", "Jamie", "Taylor"]
        options = ["FINISH"] + members

        boss_chain = self._create_boss_chain(options)

        workflow = StateGraph(AgentState)

        workflow.add_node("Sam", self._create_agent_node("Sam", MRA_PERSONALITY))
        workflow.add_node("Jamie", self._create_agent_node("Jamie", PD_PERSONALITY))
        workflow.add_node("Taylor", self._create_agent_node("Taylor", SM_PERSONALITY))
        workflow.add_node("Alex", boss_chain)

        for member in members:
            workflow.add_edge(member, "Alex")

        conditional_map = {k: k for k in members}
        conditional_map["FINISH"] = END
        conditional_map["Alex"] = "Alex"

        workflow.add_conditional_edges(
            "Alex",
            self._route_next,
            conditional_map
        )

        workflow.add_edge(START, "Alex")

        return workflow.compile()

    def _create_boss_chain(self, options):
        function_def = {
            "name": "assign_task",
            "description": "Select the next role and assign a task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "next": {
                        "type": "string",
                        "enum": options,
                        "description": "The next worker to act or FINISH if done."
                    },
                    "task": {
                        "type": "string",
                        "description": "The task to be performed by the selected worker."
                    }
                },
                "required": ["next", "task"],
            },
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", PM_PERSONALITY),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next and what task should they perform?"
                " Or should we FINISH? Select one of: {options} and provide a task description."
                " Make sure the current phase is completed thoroughly before finishing."
            ),
        ]).partial(options=str(options))

        return (
            prompt
            | self.llm.bind_functions(functions=[function_def], function_call="assign_task")
            | JsonOutputFunctionsParser()
            | (lambda x: {
                "next": x["next"],
                "messages": [HumanMessage(content=f"Task: {x.get('task', 'Respond to the current situation.')}", name="Alex")]
            })
        )

    def _create_agent_node(self, name: str, personality: str):
        agent = create_agent(
            self.llm, 
            self.toolbox, 
            INITIAL_MESSAGE_BASE.format(agent_personality=personality, agent_name=name)
        )
        return lambda state: agent_node(state, agent, name, self.kb_system)

    @staticmethod
    def _route_next(state):
        return state["next"]

    def run_phase(self, problem: str):
        initial_state = {
            "messages": [HumanMessage(content=problem)],
        }
        
        for s in self.workflow.stream(initial_state, {"recursion_limit": 50}):
            if "__end__" not in s:
                print(s)
                print("----")

def main():
    parser = argparse.ArgumentParser(description="Run multi-agent system for product development phases.")
    parser.add_argument("problem", help="The problem statement for the phase")
    args = parser.parse_args()

    config = load_config()
    config_values = get_all_config_values(config)

    # Set environment variables
    os.environ["OPENAI_API_KEY"] = config_values['OPENAI_API_KEY']
    if config_values['LANGCHAIN_TRACING_V2']:
        os.environ["LANGCHAIN_API_KEY"] = config_values['LANGCHAIN_API_KEY']
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = config_values['LANGCHAIN_PROJECT']
    else:
        # Remove LangChain environment variables if tracing is disabled
        for var in ["LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT"]:
            os.environ.pop(var, None)

    workflow_manager = WorkflowManager(config_values)
    workflow_manager.run_phase(args.problem)

if __name__ == "__main__":
    main()