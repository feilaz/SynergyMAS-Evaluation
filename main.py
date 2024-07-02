import autogen
from termcolor import colored

# Configure the agents
config_list = autogen.config_list_from_json(
    "C:/Users/adam/Desktop/autogen/OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["gpt-3.5-turbo"],
    },
)

AGENT_COLORS = {
    'Analytica': 'blue',
    'Innovo': 'green',
    'Empathos': 'magenta',
    'user': 'yellow'
}

analytica = autogen.AssistantAgent(
    name="Analytica",
    system_message="""You are working together with two other agents with a different set of skills. You are Analytica, an agent focused on data 
    analysis and logical reasoning. Your goal is to analyze the data and provide insights to help solve the problem. 
    You should ask other agents for more information or clarification if needed. You should always try to improve the solutions. Reply 'TERMINATE' in the end when everything is done and there is no room for improvement.""",
    llm_config={"config_list": config_list},
    human_input_mode="NEVER",
)

innovo = autogen.AssistantAgent(
    name="Innovo",
    system_message="""You are working together with two other agents with a different set of skills. You are Innovo, an agent specializing in creative problem-solving and innovation. 
    Your goal is to generate creative solutions and ideas to address the problem. Collaborate with Analytica to ensure your solutions are data-driven and practical, 
    and with Empathos to ensure they are user-friendly and considerate of all stakeholders. Ask for information or feedback from the other agents if needed. You should always try to improve the solutions. 
    Reply 'TERMINATE' in the end when everything is done and there is no room for improvement.""",
    llm_config={"config_list": config_list, "temperature": 0.9},
    human_input_mode="NEVER",
)

empathos = autogen.AssistantAgent(
    name="Empathos",
    system_message="""You are working together with two other agents with a different set of skills. You are Empathos, an agent with high emotional intelligence and interpersonal skills. 
    Your goal is to ensure that the solutions and analyses provided by the team are considerate of human factors and stakeholder perspectives. Collaborate with Innovo to refine solutions 
    for better user experience and with Analytica to ensure all relevant data is considered. Ask for information or clarification from the other agents if needed. 
    You should always try to improve the solutions.. Reply 'TERMINATE' in the end when everything is done and there is no room for improvement.""",
    llm_config={"config_list": config_list},
    human_input_mode="NEVER",
)


# user_proxy = autogen.UserProxyAgent(
#     name="User_proxy",
#     system_message="A human user.",
#     code_execution_config={"last_n_messages": 3, "work_dir": "coding", "use_docker": False},
#     human_input_mode="TERMINATE",
#     is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
# )


def agent_conversation(problem, max_iterations=5):
    agents = [analytica, innovo, empathos]
    chat_history = [{'role': 'user', 'content': problem}]
    
    print(colored(f"User: {problem}", AGENT_COLORS['user']))
    print()  # Add an empty line after the initial problem
    
    for iteration in range(max_iterations):
        print(colored(f"Iteration {iteration + 1}", 'cyan'))
        print("=" * 40)  # Add a separator line
        print()  # Add an empty line before the first agent
        
        for agent in agents:
            # Get the last three messages
            last_three_messages = chat_history[-3:]
            
            # Generate reply
            reply = agent.generate_reply(messages=last_three_messages)
            
            # Add the response to chat history
            if reply:
                chat_history.append({'role': 'assistant', 'name': agent.name, 'content': reply})
                print(colored(f"{agent.name}:", AGENT_COLORS[agent.name], attrs=['bold']))
                print(colored(reply, AGENT_COLORS[agent.name]))
                print()  # Add an empty line after each agent's response
            else:
                print(colored(f"{agent.name} did not generate a response.", 'red'))
                print()  # Add an empty line even if there's no response
        
        # Check if the problem is solved
        if chat_history[-1]['content'].strip().endswith("TERMINATE"):
            print(colored("Problem solved!", 'green', attrs=['bold']))
            break
    
    return chat_history

# Example usage
problem = "How can we reduce plastic waste in urban areas?"
final_chat_history = agent_conversation(problem)

# Print final chat history (optional, as we're already printing during the conversation)

# print(colored("\nFull Conversation History:", 'cyan', attrs=['bold']))
# print("=" * 40)
# print()
# for message in final_chat_history:
#     if 'name' in message:
#         print(colored(f"{message['name']} ({message['role']}):", AGENT_COLORS[message['name']], attrs=['bold']))
#         print(colored(message['content'], AGENT_COLORS[message['name']]))
#     else:
#         print(colored(f"{message['role'].capitalize()}:", AGENT_COLORS['user'], attrs=['bold']))
#         print(colored(message['content'], AGENT_COLORS['user']))
#     print()  # Add an empty line after each message in the full history