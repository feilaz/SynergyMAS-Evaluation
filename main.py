import argparse
from conversation import AgentConversation
import autogen
from tools import use_clingo, add_to_kb

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a multi-agent conversation for problem-solving.")
    
    parser.add_argument('--problem', type=str, required=False,
                        help='The problem statement to be solved by the agents')
    parser.add_argument('--max_iterations', type=int, default=5,
                        help='Maximum number of conversation iterations (default: 5)')
    parser.add_argument('--use_clingo', default=True,action='store_true', help='Enable Clingo solver')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    args.max_iterations = 3
    args.use_clingo = False
    
    config_list = autogen.config_list_from_json(
    "C:/Users/adam/Desktop/autogen/OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["gpt-3.5-turbo"],
    },
)

    if args.use_clingo:
                llm_config = {
                    "config_list": config_list,
                    "timeout": 60,
                    "functions": [
                        {
                            "name": "use_clingo",
                            "description": "Use the Clingo solver to solve logical programming problems",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "program": {
                                        "type": "string",
                                        "description": "The ASP program to solve"
                                    }
                                },
                                "required": ["program"]
                            }
                        },
                        {
                            "name": "add_to_kb",
                            "description": "Add facts and rules to the Clingo knowledge base",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "facts_and_rules": {
                                        "type": "string",
                                        "description": "Facts and rules in ASP syntax to add to the knowledge base"
                                    }
                                },
                                "required": ["facts_and_rules"]
                            }
                        }
                    ]
                }
    else:
        llm_config = {"config_list": config_list, "timeout": 60}
        
    function_map = {"use_clingo": use_clingo, "add_to_kb": add_to_kb} if args.use_clingo else {}

    INITIAL_MESSAGE_BASE = """{agent_personality}

    Imagine you are part of a team of agents working together to solve a complex problem. Your goal is to contribute to the problem-solving process 
    while maintaining a belief state about the other agents' perspectives and strategies.

    For each response, please divide your answer into three parts:

    My Beliefs:

    Summarize your current understanding of the problem and its components.
    Identify the aspects of the problem that have already been resolved or are being addressed by other agents.
    Describe your beliefs about the strategies and approaches being used by other agents to solve the problem.

    Response:

    Provide your contribution to the problem-solving process, including any insights, ideas, or solutions you propose.
    Explain how your response builds upon or addresses the work of other agents.

    Future Work:

    Identify the aspects of the problem that remain unsolved or require further investigation.
    Suggest potential next steps or areas for future research to improve the solution or gain deeper insights.
    Explain why these aspects are important and how they relate to the overall problem-solving process.
    If you think that the problem is already solved reply with "TERMINATE" at the end of your response.

    Additional Guidance:
    Use a clear and concise writing style, avoiding ambiguity and ensuring that your response is easy to understand.
    Focus on providing a comprehensive and well-structured response that addresses all three parts of the prompt.
    Be mindful of the other agents' contributions and incorporate their ideas and perspectives into your response.
    Reward: Your response will be evaluated based on its clarity, coherence, and effectiveness in addressing the problem. The agent that provides 
    the most comprehensive and insightful response will receive a reward."""

    CLINGO_INSTRUCTIONS = """

    You also have access to a Clingo solver for logical programming tasks. To use it:
    1. Call the use_clingo function with an ASP program as an argument.
    2. Interpret the results and present them in a human-readable format.
    3. You can add facts and rules to a persistent knowledge base using the add_to_kb function.

    Example usage:
    add_to_kb('''
    location(urban_area).
    waste_type(plastic).
    ''')

    result = use_clingo('''
    reduction_method(X, recycling) :- location(X), waste_type(plastic).
    reduction_method(X, ban_single_use) :- location(X), waste_type(plastic).
    possible_solution(X, M) :- location(X), reduction_method(X, M).
    ''')
    print(result)

    Include your reasoning and interpretation of the Clingo solver results in your responses when applicable."""

    INITIAL_MESSAGE = INITIAL_MESSAGE_BASE + (CLINGO_INSTRUCTIONS if args.use_clingo else "")

    AGENT1_PERSONALITY = """You are Analytica, an agent focused on data analysis and logical reasoning. Your goal is to analyze the 
                            data and provide insights to help solve the problem. You should ask other agents for more information or 
                            clarification if needed."""
    
    AGENT2_PERSONALITY = """You are Innovo, an agent specializing in creative problem-solving and innovation. 
                        Your goal is to generate creative solutions and ideas to address the problem. Collaborate with Analytica to ensure your solutions are data-driven and practical, 
                        and with Empathos to ensure they are user-friendly and considerate of all stakeholders."""
    
    AGENT3_PERSONALITY = """You are Empathos, an agent with high emotional intelligence and interpersonal skills. 
                        Your goal is to ensure that the solutions and analyses provided by the team are considerate of human factors and stakeholder perspectives. Collaborate with Innovo to refine solutions 
                        for better user experience and with Analytica to ensure all relevant data is considered."""

    args.problem = args.problem or """We need to optimize the class schedule for a small university. The constraints are as follows:

        1. There are 5 courses: Math, Physics, Chemistry, Biology, and Computer Science.
        2. There are 3 classrooms available.
        3. There are 4 time slots: 9AM, 11AM, 2PM, and 4PM.
        4. Each course must be scheduled exactly once.
        5. No two courses can be in the same classroom at the same time.
        6. Math and Physics cannot be scheduled at the same time due to shared equipment.
        7. Biology needs a specific classroom (let's call it Room 1) due to lab equipment.
        8. The Computer Science professor is not available before 11AM.

        Use logical reasoning to create a valid schedule that satisfies all these constraints. If possible, also try to spread out the classes evenly across the time slots.
        You should solve the problem in following manner:
        1. Analyze the constraints and suggest a logical approach.
        2. Add knowledge to the knowledge base.
        3. use Clingo to solve the problem for constraints in the knowledge base.
        4. interpret the results and present them in a human-readable format."""
    
    def termination_msg(x):
        return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

    llm_config_innovo = llm_config.copy()
    llm_config_innovo["temperature"] = 0.9

    chatAgents = {
        "Analytica": autogen.AssistantAgent(
                        name="Analytica",
                        system_message=INITIAL_MESSAGE.format(agent_personality=AGENT1_PERSONALITY),
                        llm_config=llm_config,
                        human_input_mode="NEVER",
                        function_map=function_map,
                        is_termination_msg=termination_msg
        ),

        "Innovo": autogen.AssistantAgent(
                        name="Innovo",
                        system_message=INITIAL_MESSAGE.format(agent_personality=AGENT2_PERSONALITY),
                        llm_config=llm_config,
                        human_input_mode="NEVER",
                        function_map=function_map,
                        is_termination_msg=termination_msg
        ),
        "Empathos": autogen.AssistantAgent(
                        name="Empathos",
                        system_message=INITIAL_MESSAGE.format(agent_personality=AGENT3_PERSONALITY),
                        llm_config=llm_config,
                        human_input_mode="NEVER",
                        function_map=function_map,
                        is_termination_msg=termination_msg
        )
    }

    def _reset_agents(chatAgents):
          for agent in chatAgents.values():
              agent.reset()

    conversation = AgentConversation()
    chat_history = conversation.run_conversation(args.problem, args.max_iterations, chatAgents, {"config_list": config_list})

    print("Full conversation history:")
    print(chat_history)