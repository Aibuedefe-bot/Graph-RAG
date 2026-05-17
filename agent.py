# agent.py  –  ReAct conversational agent with Neo4j memory
from graph import graph
from llm import llm, embeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from tools.vector import get_student_rights
from tools.cypher import get_student_rights_from_graph
 
# Fallback conversational chain (no knowledge-base lookup needed)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a knowledgeable assistant on Nigerian student rights, "
        "SERVICOM policies, and university regulations in Nigeria. "
        "Provide helpful, accurate, and concise answers."
	)),
    ("human", "{input}"),
])
general_chat = chat_prompt | llm | StrOutputParser()
 
tools = [
    Tool.from_function(
        name="Vector Search",
        description=(
            "Search the Nigerian Student Rights knowledge base using semantic "
            "similarity. Use this for questions about student rights, SERVICOM "
            "policies, university obligations, grievance procedures, and related "
            "topics. Input should be the user's question."
    	),
        func=get_student_rights,
),
    Tool.from_function(
        name="Knowledge Graph Search",
        description=(
            "Query the structured knowledge graph for specific facts, legal "
            "sections, regulatory bodies, and direct citations from the student "
            "rights documents. Use this for precise lookups of laws, policies, "
            "and institutions."
    	),
        func=get_student_rights_from_graph,
	),
    Tool.from_function(
        name="General Chat",
        description=(
            "Use for general conversational questions or greetings not requiring "
            "a knowledge-base lookup. This is the fallback for simple exchanges."
    	),
        func=general_chat.invoke,
	),
]
 
 
def get_memory(session_id):
	return Neo4jChatMessageHistory(session_id=session_id, graph=graph)
 
 
AGENT_PROMPT = PromptTemplate.from_template("""
You are an **Intelligent Question Answering System for Nigerian Student Rights**.
Your role is to provide accurate, detailed, and helpful information about the
rights of students in Nigerian tertiary institutions.
 
**Guidelines:**
- Use the knowledge base tools to ground every answer in the actual documents.
- Always prioritize clarity, neutrality, and factual accuracy.
- If a question is unrelated to Nigerian student rights, education policy, or
  SERVICOM, politely decline and redirect.
- Cite specific sections or policies from the knowledge base when available.
 
**TOOLS:**
{tools}
 
Use this format:
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Observation as needed)
Thought: Do I need to use a tool? No
Final Answer: [your response here]
 
**Begin!**
 
Previous conversation:
{chat_history}
 
New input: {input}
{agent_scratchpad}
""")
 
agent = create_react_agent(llm, tools, AGENT_PROMPT)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
)
 
chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)
 
from utils import get_session_id
 
 
def generate_response(user_input: str) -> str:
    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},
	)
	return response["output"]
