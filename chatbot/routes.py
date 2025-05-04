from fastapi import APIRouter, Body, Request, Response, HTTPException, status
import asyncio
import logging
import json
from GeminiAgent import agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
class QueryRequest(BaseModel):
    query: Any
    thread_id: str

@router.post("/chatbot", response_description="Chatbot response", status_code=status.HTTP_200_OK)
async def chatbot_response(request: Request, response: Response, body: QueryRequest):
    """
    Handles the chatbot response.
    """
    try:
        # Extract the query from the request body
        query = body.query
        thread_id=body.thread_id
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        # Run the agent with the provided query
        result = await run_agent(query, thread_id=thread_id)
        
        # Return the result
        return {"response": result}
    except Exception as e:
        logger.error("Error in chatbot_response: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")

async def run_agent(query: str,thread_id:str) -> str:
    state = {
        "messages": [HumanMessage(content=query)]
    }
    try:
        config = {"configurable": {"thread_id": thread_id}}

        result = await agent.graph.ainvoke(state, config)
        last_message = result["messages"][-1].content
        return last_message
    except Exception as e:
        logger.error("Agent execution error: %s", str(e))
        if "tool_results" in result and result.get("tool_results"):
            return json.dumps(result["tool_results"], indent=2)
        return f"Sorry, an error occurred: {str(e)}"