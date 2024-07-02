import autogen
from termcolor import colored
import re

class Agent:
    def __init__(self, name, system_message, config_list, temperature=None, use_belief_state=True):
        llm_config = {"config_list": config_list}
        if temperature is not None:
            llm_config["temperature"] = temperature
        
        self.agent = autogen.AssistantAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        self.name = name
        self.use_belief_state = use_belief_state
        self.belief_state = {}
        self.last_belief = ""

    def generate_reply(self, messages):
        if self.use_belief_state:
            belief_prompt = self.get_belief_state_prompt()
            messages.append({"role": "system", "content": belief_prompt})
        
        reply = self.agent.generate_reply(messages=messages)
        
        if self.use_belief_state:
            self.update_own_belief(reply)
        
        return reply

    def update_belief_state(self, chat_history):
        if not self.use_belief_state:
            return

        self.problem_summary = ""
        self.challenges = []
        self.synergies = []
        self.unexplored_areas = []
        self.action_items = []

        for message in chat_history:
            if message['role'] == 'assistant' and message['name'] != self.name:
                self.belief_state[message['name']] = message['content']
                self.parse_message_content(message['content'])

        self.synthesize_problem_summary()

    def parse_message_content(self, content):
        # Parse challenges
        challenges = re.findall(r"Challenge(?:s)?:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        self.challenges.extend(challenges)

        # Parse synergies
        synergies = re.findall(r"Synerg(?:y|ies):\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        self.synergies.extend(synergies)

        # Parse unexplored areas
        unexplored = re.findall(r"Unexplored area(?:s)?:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        self.unexplored_areas.extend(unexplored)

        # Parse action items
        actions = re.findall(r"Action item(?:s)?:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        self.action_items.extend(actions)

    def synthesize_problem_summary(self):
        all_content = " ".join(self.belief_state.values())
        summary = re.findall(r"Problem summary:\s*(.+?)(?:\n|$)", all_content, re.IGNORECASE)
        if summary:
            self.problem_summary = summary[0]
        else:
            # If no explicit problem summary is found, generate one
            words = all_content.split()
            self.problem_summary = "Problem summary: " + " ".join(words[:30]) + "..."

    def get_belief_state_prompt(self):
        if not self.use_belief_state:
            return ""

        belief_prompt = f"""Current Belief State on Problem Solving Progress:

    1. Problem Summary: 
    {self.problem_summary}

    2. Agent Contributions:
    """
        for agent, contribution in self.belief_state.items():
            belief_prompt += f"   - {agent}: {contribution}\n"

        belief_prompt += f"""
    3. Identified Challenges:
    {'; '.join(self.challenges)}

    4. Potential Synergies:
    {'; '.join(self.synergies)}

    5. Unexplored Areas:
    {'; '.join(self.unexplored_areas)}

    6. Action Items:
    {'; '.join(self.action_items)}

    Your last stated belief: {self.last_belief}

    Based on this information:
    1. Update your understanding of the problem and progress.
    2. Identify how your expertise can address unexplored areas or challenges.
    3. Suggest ways to build on or complement other agents' contributions.
    4. Propose next steps or actions to move the problem-solving process forward.

    Include your updated beliefs and proposals in your response, using the following format:

    My updated belief:
    Problem summary: [Your updated problem summary]
    Challenges: [List new or updated challenges]
    Synergies: [List new or updated synergies]
    Unexplored areas: [List new or updated unexplored areas]
    Action items: [List new or updated action items]

    [Your detailed response and proposals]
    """
        return belief_prompt

    def update_own_belief(self, reply):
        # Extract the belief statement from the reply
        # This is a simple implementation; you might want to use more sophisticated parsing
        belief_start = reply.find("My updated belief:")
        if belief_start != -1:
            self.last_belief = reply[belief_start:].split("\n")[0].strip()

