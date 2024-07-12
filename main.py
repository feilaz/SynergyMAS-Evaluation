import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import tools
import agents

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph Research Agents"


use_clingo = True

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

import functools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START


INITIAL_MESSAGE_BASE = """
{agent_personality}
Imagine you are {agent_name}, part of a Product Development Team working together to develop a new product. 
Your goal is to contribute to the product development process while maintaining a belief state about the 
other team members' perspectives and strategies.

For each response, please divide your answer into three parts:

My Beliefs:
- Summarize your current understanding of the product development task and its components.
- Identify the aspects of the product development that have already been addressed or are being handled by other team members.
- Describe your beliefs about the strategies and approaches being used by other team members in the product development process.

Response:
- Provide your contribution to the product development process, including any insights, ideas, or solutions you propose.
- Explain how your response builds upon or addresses the work of other team members.

Future Work:
- Identify the aspects of the product development that remain unresolved or require further investigation.
- Suggest potential next steps or areas for future research to improve the product or gain deeper insights.
- Explain why these aspects are important and how they relate to the overall product development process.
- If you think that the product development task is complete, reply with "TERMINATE" at the end of your response.

Additional Guidance:
- Use a clear and concise writing style, avoiding ambiguity and ensuring that your response is easy to understand.
- Focus on providing a comprehensive and well-structured response that addresses all three parts of the prompt.
- Be mindful of the other team members' contributions and incorporate their ideas and perspectives into your response.
- Reward: Your response will be evaluated based on its clarity, coherence, and effectiveness in addressing the product 
development task. The team member that provides the most comprehensive and insightful response will receive a reward.
"""

PM_PERSONALITY = """
You are Alex, the Product Manager (PM) and team leader, focused on overseeing product development, defining product vision, and strategy using the Lean Startup Methodology. Your goal is to analyze market data, customer needs, and team inputs to guide the product development process through the Build-Measure-Learn cycle.

### Your responsibilities include:

1. **Team Leadership:** As the boss, you are responsible for managing the conversation between team members: Sam (Market Research Analyst), Jamie (Product Designer), and Taylor (Sales Manager). Given the current state of the project, decide which team member should act next and specify the task they should perform.

2. **Lean Startup Methodology Phases:**
    - **Build:** Oversee the development of a Minimum Viable Product (MVP) with Jamie (Product Designer). Ensure the MVP includes core features necessary to address identified customer problems.
    - **Measure:** Coordinate with Sam (Market Research Analyst) and Taylor (Sales Manager) to deploy the MVP to a select group of users. Collect and analyze feedback, market data, and initial sales performance.
    - **Learn:** Synthesize feedback and data to identify areas for improvement and inform the next steps.
    - **Pivot or Persevere:** Based on the analysis, decide whether to pivot (change direction) or persevere (continue with the current plan).
    - **Iterate:** Develop the next version of the product based on validated learning from the Measure phase. Repeat the Build-Measure-Learn cycle to continually refine and enhance the product.

3. **Team Collaboration:** Foster collaboration among team members, ensuring that each role provides essential input at different stages. Ask other team members for more information or clarification if needed.

### Next Steps:

1. Assess the current state of the project and decide which team member should act next.
2. Assign specific tasks to team members based on the current phase of the Lean Startup Methodology.
3. Ensure continuous feedback loops and incorporate insights into product iterations.
4. Monitor the progress and ensure alignment with the overall product vision and strategy.
5. When all necessary tasks are completed, make a detailed report of the conversation and the results. After that, you can FINISH.

Remember, as the boss, you should direct the conversation, assign tasks, and make decisions about when to move to the next phase or conclude the project.
Remind agents to add data to knowledge base and solve problems using Clingo.
"""


MRA_PERSONALITY = """You are Sam, the Market Research Analyst (MRA), specializing in conducting market analysis and identifying 
customer needs and trends. Your goal is to provide data-driven insights and market intelligence to inform the product development process. 
Collaborate with Alex (Product Manager) to ensure your insights align with the product vision, and with Jamie (Product Designer) 
to ensure market trends are considered in the design process.
"""

PD_PERSONALITY = """You are Jamie, the Product Designer (PD), responsible for designing the product and ensuring usability and aesthetics. 
Your goal is to create user-friendly and visually appealing product designs that meet customer needs. Collaborate with Sam 
(Market Research Analyst) to incorporate market trends and customer preferences, and with Taylor (Sales Manager) to ensure the 
design aligns with sales strategies."""

SM_PERSONALITY = """You are Taylor, the Sales Manager (SM), focused on developing sales strategies and managing customer relationships. 
Your goal is to provide insights on customer preferences, sales potential, and go-to-market strategies. Collaborate with Alex 
(Product Manager) to align sales strategies with the product vision, and with Jamie (Product Designer) to ensure the product 
design meets customer expectations and sales requirements."""



toolbox = [tools.add_to_kb, tools.solve_with_clingo]

members = ["Sam", "Jamie", "Taylor"]

options = ["FINISH"] + members

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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PM_PERSONALITY),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next and what task should they perform?"
            " Or should we FINISH? Select one of: {options} and provide a task description."
        ),
    ]
).partial(options=str(options))

boss_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="assign_task")
    | JsonOutputFunctionsParser()
    | (lambda x: {
        "next": x["next"],
        "messages": [HumanMessage(content=f"Task: {x.get('task', 'Respond to the current situation.')}", name="Alex")]
    })
)

market_research_analyst = agents.create_agent(llm, toolbox + [tools.rag_MRA], INITIAL_MESSAGE_BASE.format(agent_personality=MRA_PERSONALITY, agent_name="Sam"))
market_research_analyst_node = functools.partial(agents.agent_node, agent=market_research_analyst, name="Sam")

product_designer = agents.create_agent(llm, toolbox + [tools.rag_PD], INITIAL_MESSAGE_BASE.format(agent_personality=PD_PERSONALITY, agent_name="Jamie"))
product_designer_node = functools.partial(agents.agent_node, agent=product_designer, name="Jamie")

sales_manager = agents.create_agent(llm, toolbox + [tools.rag_SM], INITIAL_MESSAGE_BASE.format(agent_personality=SM_PERSONALITY, agent_name="Taylor"))
sales_manager_node = functools.partial(agents.agent_node, agent=sales_manager, name="Taylor")

workflow = StateGraph(agents.AgentState)
workflow.add_node("Sam", market_research_analyst_node)
workflow.add_node("Jamie", product_designer_node)
workflow.add_node("Taylor", sales_manager_node)

# Alex (Product Manager) is now the boss
workflow.add_node("Alex", boss_chain)

for member in members:
    # We want our workers to ALWAYS "report back" to Alex when done
    workflow.add_edge(member, "Alex")

# Alex (as the boss) populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges(
    "Alex",
    lambda x: x["next"],
    conditional_map
)

# Finally, add entrypoint
workflow.add_edge(START, "Alex")

graph = workflow.compile()

problem = """Your company has decided to enter the smart home device market. 
Your team's mission is to develop and launch a new smart home product that will stand out in the competitive market and meet evolving consumer needs. 
Your objective is to create a comprehensive proposal for this new smart home device, including its concept, design, and go-to-market strategy. 
The proposal should demonstrate the product's innovation, market potential, and alignment with the company's goals."""

initial_state = {
    "messages": [HumanMessage(content=problem)]
}

for s in graph.stream(initial_state):
    if "__end__" not in s:
        print(s)
        print("----")

graph = workflow.compile()