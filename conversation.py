# conversation.py
from termcolor import colored
import autogen

AGENT_COLORS = {
    'user': 'yellow',
    'Analytica': 'blue',
    'Innovo': 'green',
    'Empathos': 'magenta'
}

class AgentConversation:
    def __init__(self):
        pass

    def run_conversation(self, problem, max_iterations=5, chatAgents=None, config_list=None):
        if chatAgents is None:
            raise ValueError("chatAgents dictionary must be provided")

        group_chat = autogen.GroupChat(
            agents=[agent for agent in chatAgents.values()],
            messages=[],
            max_round=max_iterations,
            speaker_selection_method="ROUND_ROBIN"
)
        group_chat_manager = autogen.GroupChatManager(
        groupchat=group_chat,
        llm_config=config_list,
        human_input_mode="NEVER",
    )
        chat_history = group_chat_manager.initiate_chat(next(iter(chatAgents.values())), message=problem)
        
        return chat_history