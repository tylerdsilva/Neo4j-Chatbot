from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool

# project modules
from llm import llm, memory
from tools.vector import kg_qa
from tools.cypher import cypher_qa
from tools.llmchain import chat_chain
from tools.youtube import youtube_response

# System message used to define the scope of the chatbot
SYSTEM_MESSAGE ="""
You are a movie expert providing information about movies.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to movies, actors, producers, reviewers or directors.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.
"""


# LLM will used description to decide the tool to use
tools = [
    Tool.from_function(
        name="Vector Search Index",  
        description="Provides information about movie taglines using Vector Search. The question will be a string. Return a string.",
        func = kg_qa
    ),
    Tool.from_function(
        name="Graph Cypher QA Chain",  
        description="Provides information about movies including their actors, directors, producers, and reviews", 
        func = cypher_qa, 
    ),
    Tool.from_function(
        name="YouTubeSearchTool",
        description="For when you need a link to a movie trailer. The question will be a string. Return a link to a YouTube video.",
        func=youtube_response,
        return_direct=True
    ),
    Tool.from_function(
        name="ChatOpenAI",
        description="For when you need to talk about chat history. The question will be a string. Return a string.",
        func=chat_chain.run,
        return_direct=True
    )
]

# Creation of Agent
agent = initialize_agent(
    tools,
    llm,
    memory=memory,
    # verbose=True,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    agent_kwargs={"system_message": SYSTEM_MESSAGE}
)

def generate_response(prompt):
    """
    Handler that calls the Conversational agent
    and returns a response to the Terminal
    """

    response = agent(prompt)

    return response['output']