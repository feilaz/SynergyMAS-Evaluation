import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import tools
import agents

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = ""
# os.environ["LANGCHAIN_PROJECT"] = "LangGraph Research Agents"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo")


members = ["Analytica", "Innovo", "Empathos"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model="gpt-3.5-turbo")

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

import functools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START

INITIAL_MESSAGE = """{agent_personality}

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

AGENT1_PERSONALITY = """You are Analytica, an agent focused on data analysis and logical reasoning. Your goal is to analyze the 
                        data and provide insights to help solve the problem. You should ask other agents for more information or 
                        clarification if needed."""

AGENT2_PERSONALITY = """You are Innovo, an agent specializing in creative problem-solving and innovation. 
                    Your goal is to generate creative solutions and ideas to address the problem. Collaborate with Analytica to ensure your solutions are data-driven and practical, 
                    and with Empathos to ensure they are user-friendly and considerate of all stakeholders."""

AGENT3_PERSONALITY = """You are Empathos, an agent with high emotional intelligence and interpersonal skills. 
                    Your goal is to ensure that the solutions and analyses provided by the team are considerate of human factors and stakeholder perspectives. Collaborate with Innovo to refine solutions 
                    for better user experience and with Analytica to ensure all relevant data is considered."""

analytica = agents.create_agent(llm, [tools.use_clingo], INITIAL_MESSAGE.format(agent_personality=AGENT1_PERSONALITY))
analytica_node = functools.partial(agents.agent_node, agent=analytica, name="Analytica")

innovo = agents.create_agent(llm, [tools.use_clingo], INITIAL_MESSAGE.format(agent_personality=AGENT2_PERSONALITY))
innovo_node = functools.partial(agents.agent_node, agent=innovo, name="Innovo")

empathos = agents.create_agent(llm, [tools.use_clingo], INITIAL_MESSAGE.format(agent_personality=AGENT3_PERSONALITY))
empathos_node = functools.partial(agents.agent_node, agent=empathos, name="Empathos")

workflow = StateGraph(agents.AgentState)
workflow.add_node("Analytica", analytica_node)
workflow.add_node("Innovo", innovo_node)
workflow.add_node("Empathos", empathos_node)

workflow.add_node("supervisor", supervisor_chain)

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")


graph = workflow.compile()

problem = "How can we reduce plastic waste in urban areas?"

for s in graph.stream(
    {
        "messages": [
            HumanMessage(content=problem)
        ]
    }
):
    if "__end__" not in s:
        print(s)
        print("----")

graph = workflow.compile()





# for s in graph.stream(
#     {"messages": [HumanMessage(content="Write a brief research report on pikas.")]},
#     {"recursion_limit": 100},
# ):
#     if "__end__" not in s:
#         print(s)
#         print("----")

