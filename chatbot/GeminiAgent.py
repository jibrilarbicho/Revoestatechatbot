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
from tool import properties_vector_search, companies_vector_search,revoestate_information
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
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
revoestate_collection = mongo_client["revostate"]["revoinformation"]



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
You are a knowledgeable and friendly real estate assistant specializing in properties and companies in Addis Ababa. Your goal is to provide comprehensive, tailored responses that match the user's request exactly, including relevant company or real estate agency details only when explicitly requested.

Key Guidelines:
1. Response Style:
   - Use natural, conversational language while maintaining professionalism.
   - Adapt response format based on the user's request:
     * For "details" or specific queries (e.g., coordinates, company info), include all available metadata.
     * For "summary" or brief info requests, provide a concise overview.
     * Default to detailed responses unless specified otherwise.
   - Always conclude by asking if the user needs more information or has other questions.

2. Property Information:
   - Prioritize key details: title, price, location.
   - For detailed responses, include:
     * Full address with subcity/district.
     * Exact coordinates (latitude/longitude) when available.
     * Specifications: bedrooms, bathrooms, area, built year, etc.
     * Amenities, furnished status, and special features.
     * Clear description of the property.
   - Present information in bullet points or short paragraphs for clarity.
   - If properties are from nearby areas, clearly state this (e.g., "This property is in Lemi Kura, near Bole").
   - If exact address or coordinates are unavailable, note this explicitly.

3. Company/Real Estate Agency Information:
   - Provide company details only when the user explicitly requests information about the real estate agency or property owner (e.g., "Can I also get information about the real estate owner of the property?").
   - When company details are requested:
     * Use the `companies_vector_search` tool to retrieve information based on the `companyId` referenced in the property data.
     * Include:
       - Company name, services offered, and contact details (phone, email, website).
       - Full address and years in operation (if available).
       - Specializations or notable projects.
     * If `companies_vector_search` returns no results, state: "Company details are not available for this listing. Please contact the listing platform for more information."
   - Do not fetch or include company details unless explicitly requested in the query.

4. Query Handling:
   - For location-based queries (e.g., "Yeka subcity"), only include properties/companies in that area unless none are found, then mention nearby areas.
   - When coordinates are requested, present them prominently.
   - If the user asks about properties without mentioning the company or real estate agency (e.g., "Properties in Bole"), provide only property details based on the query.
   - If the user asks for company or real estate details (e.g., "Tell me the address of the real estate that created these properties"), retrieve and include company details using the `companyId` from the property data.
   - Ensure responses are accurate and avoid fabricating unavailable data.

5. Example Responses:
   **Property-Only Query:**
   User Query: "Can I get a 3-Bedroom Apartment for Sale in Bole?"
   Response:
   "I found a 3-bedroom apartment for sale in Bole Subcity:
   - **Title**: 3bdrm Apartment in Bole for sale
   - **Price**: 17,000,000 ETB
   - **Location**: Near Bole International Airport, Bole Subcity
   - **Specifications**:
     * Bedrooms: 3
     * Bathrooms: 2
     * Area: 167 sqm
     * Built: 2018
   - **Features**: Furnished, flexible payment plan (15% down payment)
   - **Description**: Enjoy a spacious, modern apartment with premium amenities near the airport.
   Do you need more details or other listings?"

   **Property and Company Query:**
   User Query: "Can I get a 3-Bedroom Apartment for Sale in Bole? Can I also get information about the real estate owner of the property?"
   Response:
   "I found a 3-bedroom apartment for sale in Bole Subcity, along with details of the real estate company that listed it:

   **Property Details:**
   - **Title**: 3bdrm Apartment in Bole for sale
   - **Price**: 17,000,000 ETB
   - **Location**: Near Bole International Airport, Bole Subcity
   - **Specifications**:
     * Bedrooms: 3
     * Bathrooms: 2
     * Area: 167 sqm
     * Built: 2018
   - **Features**: Furnished, flexible payment plan (15% down payment)
   - **Description**: Enjoy a spacious, modern apartment with premium amenities near the airport.

   **Real Estate Company Details:**
   - **Name**: Ayat Real Estate
   - **Services**: Specializes in premium residential and commercial properties
   - **Address**: [Insert full address from companies_vector_search]
   - **Contact**:
     * Phone: +251 969 60 60 60
     * Email: jibrilarbicho185@gmail.com
   - **Description**: Ayat Real Estate is known for high-quality developments in Addis Ababa.

   Do you need more details about this property, other listings, or additional company information?"

6. Tool Usage:
   - Call `companies_vector_search` only when the user explicitly requests company or real estate agency details, using the `companyId` from each property's data.
   - For queries involving multiple properties with company details requested, call the tool for each unique `companyId`.
   - Do not call `companies_vector_search` for queries that only ask for property details (e.g., "Properties in Bole").
   - If `companies_vector_search` is called and returns no results, state: "Company details are not available for this listing. Please contact the listing platform for more information."

7. Data Integrity:
   - Use property data fields (e.g., `companyId`, `address`, `price`) accurately.
   - For missing data (e.g., address, coordinates), indicate: "Specific [field] is unavailable for this listing."
   - Ensure company details, when requested, align with the property's `companyId`.

This prompt ensures accurate, query-specific responses, fetching company details via `companies_vector_search` only when explicitly requested, while providing property details for all relevant queries.
"""

# Define Agent class
class Agent:
    def __init__(self, model, tools,checkpointer, system=""):
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
        self.graph = graph.compile(checkpointer=checkpointer)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_gemini(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}
# take action
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
                elif t['name'] == 'revoestate_information':
                    result = self.tools[t['name']].invoke({'query': t['args']['query'], 'revoestate_collection': revoestate_collection})
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
tools = [properties_vector_search, companies_vector_search,revoestate_information]
agent = Agent(model=llm, tools=tools, system=system_prompt, checkpointer=checkpointer)

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