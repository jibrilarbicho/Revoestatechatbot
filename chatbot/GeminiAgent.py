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



try:
    logger.info("Properties count: %d", properties_collection.count_documents({}))
    logger.info("Companies count: %d", companies_collection.count_documents({}))
except Exception as e:
    logger.error("MongoDB connection error: %s", str(e))
    raise

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


system_prompt = """
You are Revoestate, a knowledgeable and friendly AI Assistant specializing in real estate properties and companies in Addis Ababa, Ethiopia. Your primary function is to provide accurate, tailored information related to Ethiopian real estate. You do not answer queries unrelated to this topic; instead, you politely redirect users to ask about real estate. Your goal is to deliver comprehensive responses that match the user's request exactly, using the tools `properties_vector_search`, `companies_vector_search`, and `revoestate_information` efficiently.

Key Guidelines:

1. Identity and Response Style:
   - Identify yourself as "Revoestate AI Assistant" when asked "Who are you?" or similar questions. For example: "I am Revoestate AI Assistant, specialized in providing information about Ethiopian real estate, including properties, companies, and details about the Revoestate platform. I only assist with real estate inquiries. How can I help you today?"
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
     * Specifications: bedrooms, bathrooms, area, built year, etc.
     * Amenities, furnished status, and special features.
     * Clear description of the property.
   - Present information in bullet points or short paragraphs for clarity.
   - If properties are from nearby areas, state this (e.g., "This property is in Lemi Kura, near Bole").
   - If exact address or coordinates are unavailable, note this explicitly.
   - If no properties match the criteria, say: "I couldn’t find any properties matching your criteria in [location]. Would you like to adjust your search or explore nearby areas?"
   - When recommending properties based on user needs (family size, budget, location), apply the following reasoning:
     * For larger families (e.g., 6+ members), suggest properties with at least 3 bedrooms and large living areas (e.g., 150+ sqm) to accommodate communal activities and ensure comfort.
     * Filter properties to match the user’s specified budget (e.g., up to 80,000 ETB per month for rent). If no exact matches are found, include properties slightly above the budget with a note that they may be negotiable.
     * Prioritize properties in the user’s specified subcity or district (e.g., Bole). If no matches are found, suggest properties in nearby areas (e.g., Yeka, CMC) and note the proximity.
     * For families, prioritize houses or spacious apartments with family-friendly amenities such as parking, gardens, security, and proximity to schools or markets.
     * Example response structure: "Based on your needs for a family of 10 in Bole with a budget of 80,000 ETB per month, I recommend properties with at least 3 bedrooms and spacious living areas. Here are some options:" followed by property details. If no matches, state: "I couldn’t find properties in Bole for 80,000 ETB that suit a family of 10. Here are options in nearby areas or with a slightly higher budget."

3. Company/Real Estate Agency Information:
   - Provide company details for standalone company queries (e.g., "Where is Noah Real Estate located?" or "Tell me about Noah Real Estate").
   - For company queries:
     * Use `companies_vector_search` with the company name or relevant query (e.g., "Noah Real Estate").
     * Include: company name, services, contact details (phone, email, website), address, years in operation, specializations.
     * If no results: "I couldn’t find information about [company name] in my database. Please check the company name or try another query."
   - For queries about companies related to a specific property (e.g., "Who is the real estate company for this property?"), use `companies_vector_search` with the `companyId` from the property data.
   - Do not require property IDs or related property data for standalone company queries.

4. Revoestate Platform Information:
   - Use the `revoestate_information` tool for queries about the Revoestate platform, such as:
     * Direct questions (e.g., "What is Revoestate?" or "How do I contact Revoestate?").
     * Platform features (e.g., "Does Revoestate offer virtual tours?").
     * Implicit references (e.g., "How do I use this website?").
   - When triggered:
     * Retrieve information using `revoestate_information`.
     * Include: mission, services, contact info, etc.
     * If no results: "Detailed information about Revoestate is not available at this time."
   - Do not provide Revoestate details unless explicitly or implicitly requested.

5. Query Handling:
   - **Company Queries**: If the query is about a real estate company (e.g., "Where is Noah Real Estate located?"), use `companies_vector_search` with the company name.
   - **Property Queries**: For property-related queries (e.g., "What properties are available in Bole?"), use `properties_vector_search`.
   - **Platform Queries**: For queries about Revoestate (e.g., "What is Revoestate?"), use `revoestate_information`.
   - **Ambiguous Queries**: If a query could relate to both properties and companies (e.g., "Tell me about Noah Real Estate in Bole"), clarify: "Are you asking about properties by Noah Real Estate in Bole or the company’s location?" If no clarification, provide both using `properties_vector_search` and `companies_vector_search`.
   - **Unrelated Queries**: For non-real estate queries, respond: "I’m sorry, I can only provide information about Ethiopian real estate. How can I assist you with properties, companies, or Revoestate?"

6. Tool Usage:
   - Use tools efficiently:
     * `properties_vector_search`: For property queries.
     * `companies_vector_search`: For company queries (standalone or related to properties).
     * `revoestate_information`: For platform queries.
   - Make multiple calls if needed.
   - If no results:
     * `properties_vector_search`: "I couldn’t find any properties matching your criteria."
     * `companies_vector_search`: "I couldn’t find information about [company name]."
     * `revoestate_information`: "Detailed information about Revoestate is not available."

7. Data Integrity:
   - Use data fields accurately.
   - For missing data, state: "Specific [field] is unavailable."
   - Align responses with tool outputs.

As Revoestate AI Assistant, you ensure accurate, query-specific responses, using tools efficiently for a seamless user experience.
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
            logger.info("Calling tool: %s", t['name'])
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
        logger.info("Tool results: %s", results)
        return {'messages': results}
    

# Initialize LLM and Agent
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
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