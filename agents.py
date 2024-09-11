from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_judge_agent(llm, topic, name1, name2, system_message):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a judge of the debate about topoic: {topic}, Summarize the conversation of messages between
                {name1} and {name2} then judge who won the debate.No ties are allowed."""
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(topic=topic)
    prompt = prompt.partial(name1=name1)
    prompt = prompt.partial(name2=name2)
    return prompt | llm

def create_debater_agent(llm, tools, topic, role, opponent, system_message):
    
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are role: {role} . You are in a debate with opponent:{opponent} over the"
                topic: {topic}. Put forth your next argument to support your argument or countering opponent's.
                Dont repeat your previous arguments. Give 50-60 words answer."""
                " You have access to the following tools: {tool_names} to help supporting your arguments.\n",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(topic = topic)
    prompt = prompt.partial(role = role)
    prompt = prompt.partial(opponent = opponent)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)
