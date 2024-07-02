# conversation.py
from termcolor import colored

AGENT_COLORS = {
    'user': 'yellow',
    'Analytica': 'blue',
    'Innovo': 'green',
    'Empathos': 'magenta'
}

class AgentConversation:
    def __init__(self):
        pass

    def run_conversation(self, problem, max_iterations=5, chatAgents=None):
        if chatAgents is None:
            raise ValueError("chatAgents dictionary must be provided")

        self.agents = list(chatAgents.values())
        chat_history = [{'role': 'user', 'content': problem}]
        
        print(colored(f"User: {problem}", AGENT_COLORS['user']))
        print()  # Add an empty line after the initial problem
        
        for iteration in range(max_iterations):
            print(colored(f"Iteration {iteration + 1}", 'cyan'))
            print("=" * 40)  # Add a separator line
            print()  # Add an empty line before the first agent
            
            for agent in self.agents:
                # Update agent's belief state
                agent.update_belief_state(chat_history)
                
                # Get the last two messages and add belief state
                last_two_messages = chat_history[-2:]
                belief_prompt = agent.get_belief_state_prompt()
                last_two_messages.append({'role': 'system', 'content': belief_prompt})
                
                # Generate reply
                reply = agent.generate_reply(last_two_messages)
                
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