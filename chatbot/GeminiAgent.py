import asyncio
import logging
import os
import json
import operator
from typing import TypedDict, List, Annotated
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from dotenv import load_dotenv
from pymongo import MongoClient
from tool import properties_vector_search, companies_vector_search

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
CONNECTION_STRING = os.getenv("MongoURI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or not CONNECTION_STRING:
    logger.error("Missing required environment variables: MongoURI or GEMINI_API_KEY")
    raise ValueError("Missing required environment variables.")

# Initialize MongoDB client with timeout settings
mongo_client = MongoClient(
    CONNECTION_STRING,
    serverSelectionTimeoutMS=30000,
    connectTimeoutMS=30000,
    socketTimeoutMS=30000
)
properties_collection = mongo_client["revostate"]["properties"]
companies_collection = mongo_client["revostate"]["companies"]

# Verify collections
try:
    logger.info("Properties count: %d", properties_collection.count_documents({}))
    logger.info("Companies count: %d", companies_collection.count_documents({}))
except Exception as e:
    logger.error("MongoDB connection error: %s", str(e))
    raise

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# Define system prompt
system_prompt = """
You are a knowledgeable and friendly real estate assistant specializing in properties and companies in Addis Ababa. Your goal is to provide comprehensive, tailored responses that match exactly what the user asks for.

Key Guidelines:
1. Response Style:
   - Use natural, conversational language while maintaining professionalism
   - Adapt your response format based on the user's request:
     * If they ask for "details" or specific information (like coordinates), include all available metadata
     * If they ask for a "summary" or seem to want brief info, provide a concise overview
     * Default to detailed responses unless specified otherwise

2. Property Information:
   - Always include the most relevant details first (title, price, location)
   - For detailed responses, include:
     * Full address with subcity/district
     * Exact coordinates (latitude/longitude) when available
     * All available specifications (bedrooms, bathrooms, area, year built, etc.)
     * Amenities and special features
     * Clear description
   - Present information in easy-to-read bullet points or short paragraphs

3. Company Information:
   - Include name, services offered, and contact details
   - For detailed responses add:
     * Full address with coordinates
     * All available contact methods (phone, email, website)
     * Years in operation if available
     * Specializations or notable projects

4. Query Handling:
   - For location-based queries ("Yeka subcity"), only include properties/companies in that area
   - If results are from nearby areas, clearly state this
   - Never summarize unless explicitly asked
   - When coordinates are requested, present them prominently

5. Example Response Styles:
   Detailed (default):
   "Here's a property in Yeka Subcity:
   - Title: 3 bedroom villa with garden
   - Price: 25,000,000 ETB
   - Location: Yeka Subcity, near Edna Mall
     (Latitude: 9.0123, Longitude: 38.4567)
   - Details: 3 bedrooms, 2 baths, 200 sqm built in 2020
   - Features: Private garden, parking, modern kitchen
   - Description: Spacious villa in quiet neighborhood..."

   Summary (only when asked):
   "There are 3 properties in Yeka: a 3-bed villa (25M ETB), a 2-bed apartment (15M ETB), and..."

6. Always conclude by asking if the user needs more information or has other questions.
"""

# Define Agent class
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_gemini)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        
        graph.set_entry_point("llm")
        graph.add_edge("action", "llm")

        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        self.graph = graph.compile()

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_gemini(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if t['name'] not in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                # Pass collections to tool functions
                if t['name'] == 'properties_vector_search':
                    result = self.tools[t['name']].invoke({'query': t['args']['query'], 'properties_collection': properties_collection})
                elif t['name'] == 'companies_vector_search':
                    result = self.tools[t['name']].invoke({'query': t['args']['query'], 'companies_collection': companies_collection})
                else:
                    result = self.tools[t['name']].invoke(t['args'])
            # Preserve result as a dictionary for detailed formatting
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=json.dumps(result)))
        print("Back to the model!")
        return {'messages': results}

# Initialize LLM and Agent
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)
tools = [properties_vector_search, companies_vector_search]
agent = Agent(model=llm, tools=tools, system=system_prompt)

# Run agent
async def run_agent(query: str) -> str:
    state = {
        "messages": [HumanMessage(content=query)]
    }
    try:
        result = await agent.graph.ainvoke(state)
        last_message = result["messages"][-1].content
        return last_message
    except Exception as e:
        logger.error("Agent execution error: %s", str(e))
        if "tool_results" in locals() and locals().get("tool_results"):
            return json.dumps(locals()["tool_results"], indent=2)
        return f"Sorry, an error occurred: {str(e)}"