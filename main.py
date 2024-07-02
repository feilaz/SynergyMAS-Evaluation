import argparse
from conversation import AgentConversation
from agent import Agent
import autogen

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a multi-agent conversation for problem-solving.")
    
    # !!! change required for --problem to True after the program is build !!!

    parser.add_argument('--problem', type=str, required=False,
                        help='The problem statement to be solved by the agents')
    parser.add_argument('--max_iterations', type=int, default=5,
                        help='Maximum number of conversation iterations (default: 5)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    config_list = autogen.config_list_from_json(
    "C:/Users/adam/Desktop/autogen/OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["gpt-3.5-turbo"],
    },
)
    
    INITIAL_MEASSAGE = """{agent_personality}

    Maintain a belief state about other agents based on their contributions to the conversation. 
    Use this belief state to inform your responses and avoid repeating information or suggestions already provided by others.

    When responding:
    1. Consider the problem at hand and your specific expertise.
    2. Review your belief state about other agents' knowledge and contributions.
    3. Provide unique insights or suggestions that complement what others have already shared.
    4. If you believe the problem has been sufficiently addressed, end your message with 'TERMINATE'.

    Your goal is to contribute meaningfully to the problem-solving process while maintaining an efficient and non-redundant conversation."""

    AGENT1_PERSONALITY = """You are Analytica, an agent focused on data analysis and logical reasoning. Your goal is to analyze the 
                            data and provide insights to help solve the problem. You should ask other agents for more information or 
                            clarification if needed."""
    
    AGENT2_PERSONALITY = """You are Innovo, an agent specializing in creative problem-solving and innovation. 
                        Your goal is to generate creative solutions and ideas to address the problem. Collaborate with Analytica to ensure your solutions are data-driven and practical, 
                        and with Empathos to ensure they are user-friendly and considerate of all stakeholders."""
    
    AGENT3_PERSONALITY = """You are Empathos, an agent with high emotional intelligence and interpersonal skills. 
                        Your goal is to ensure that the solutions and analyses provided by the team are considerate of human factors and stakeholder perspectives. Collaborate with Innovo to refine solutions 
                        for better user experience and with Analytica to ensure all relevant data is considered."""

    chatAgents = {
        "Analytica": Agent("Analytica", 
                            INITIAL_MEASSAGE.format(agent_personality=AGENT1_PERSONALITY, use_belief_state=True), 
                            config_list),
        "Innovo": Agent("Innovo",
                        INITIAL_MEASSAGE.format(agent_personality=AGENT2_PERSONALITY, use_belief_state=True),
                        config_list,
                        temperature=0.9),
        "Empathos": Agent("Empathos",
                        INITIAL_MEASSAGE.format(agent_personality=AGENT3_PERSONALITY, use_belief_state=True),
                        config_list=config_list)
                        }

    args.problem = "How can we reduce plastic waste in urban areas?"

    conversation = AgentConversation()
    args.max_iterations = 2
    chat_history = conversation.run_conversation(args.problem, args.max_iterations, chatAgents)

    print(chat_history)
    print(len(chat_history))