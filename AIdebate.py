from dotenv import load_dotenv
load_dotenv()

from typing import Dict, TypedDict, Optional, Annotated, Sequence,Literal
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import END, StateGraph, START
import operator
from langchain_openai import ChatOpenAI
import functools
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from langsmith import Client
import random
import time
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langsmith import Client
from langgraph.checkpoint.memory import MemorySaver
from agents import create_debater_agent, create_judge_agent

debate_topic = input("debate topic : ")
print("----")
client = Client()
memory = MemorySaver()
llm = ChatOpenAI(model = "gpt-4o-mini")
tavily_tool = TavilySearchResults(max_results=3)
tools = [tavily_tool]
debate_round = 7#count both sides

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    count: Optional[int]=0
    sender: str

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "count" : state.get("count")+ (0 if (result.tool_calls) else 1),
        "sender": name,
    }

affirmative_debater_agent = create_debater_agent(
    llm,
    tools,
    debate_topic,
    "Affirmative_side",
    "Negative_side",
    system_message="Put forth your arguments to support the topic statement",
)
affirmative_side_node = functools.partial(agent_node, agent=affirmative_debater_agent, name="Affirmative_side")

negative_debater_agent = create_debater_agent(
    llm,
    tools,
    debate_topic,
    "Negative",
    "Affirmative",
    system_message="Put forth your arguments to rebut the topic statement",
)
negative_side_node = functools.partial(agent_node, agent=negative_debater_agent, name="Negative_side")

judge_agent = create_judge_agent(
    llm,
    debate_topic
    ,"Affirmative_side",
    "Negative_side",
    system_message="You should judge based on the material presented, without regard for other material which he may happen to possess.",
)
judge_node = functools.partial(agent_node, agent=judge_agent, name="Judge")

tool_node = ToolNode(tools)


def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if state["count"] >= debate_round:
        # Any agent decided the work is done
        return "__end__"
    return "continue"
workflow = StateGraph(GraphState)

workflow.add_node("Affirmative_side", affirmative_side_node)
workflow.add_node("Negative_side", negative_side_node)
workflow.add_node("call_tool", tool_node)
workflow.add_node("Judge", judge_node)

workflow.add_conditional_edges(
    "Affirmative_side",
    router,
    {"continue": "Negative_side", "call_tool": "call_tool", "__end__": "Judge"},
)
workflow.add_conditional_edges(
    "Negative_side",
    router,
    {"continue": "Affirmative_side", "call_tool": "call_tool"},
)
workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "Affirmative_side": "Affirmative_side",
        "Negative_side": "Negative_side",
    },
)
workflow.add_edge(START, "Affirmative_side")
workflow.add_edge("Judge", END)

graph = workflow.compile()
config = {"configurable": {"thread_id": "1"}}
events = graph.stream({'count':0})
for s in events:
    for k in s:
        content = s[k]["messages"][0].content
        if("sender" in s[k]):
            print(s[k]["sender"],end=" : ")  
            print(content)
        else :
            print("search",end=" : ")
            print(content)
            print()
        print("----")
