import logging
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from serialization import convert_to_serializable
from typing import List
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os
os.environ["HF_HOME"] = "/app/.cache"

# Initialize embeddings
embedmodel = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Raw vector search function
def raw_vector_search(collection, query: str, index_name: str, k: int = 10) -> List[Document]:
    try:
        query_embedding = embedmodel.embed_query(query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "revoemb",
                    "queryVector": query_embedding,
                    "numCandidates": k * 10,
                    "limit": k
                }
            },
            {
                "$project": {
                    "revoemb": 0,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        results = list(collection.aggregate(pipeline))
        return [Document(
            page_content=r.get("description", ""),
            metadata={k: v for k, v in r.items() if k != "description"},
            score=r.get("score", 0)
        ) for r in results]
    except Exception as e:
        logger.error("Vector search error: %s", str(e))
        return []

# Define tools
@tool
def properties_vector_search(query: str, properties_collection=None) -> List[dict]:
    """Search for real estate properties based on a query."""
    try:
        if properties_collection is None:
            raise ValueError("Properties collection not provided")
        results = raw_vector_search(properties_collection, query, "properties_vector_index")
        logger.info("Properties query: %s, results: %d", query, len(results))
        return [
            {
                "content": r.page_content,
                "metadata": convert_to_serializable(r.metadata),
                "score": r.metadata.get("score", 0)
            }
            for r in results
        ]
    except Exception as e:
        logger.error("Properties search error: %s", str(e))
        return []

@tool
def companies_vector_search(query: str, companies_collection=None) -> List[dict]:
    """Search for real estate companies based on a query."""
    try:
        if companies_collection is None:
            raise ValueError("Companies collection not provided")
        results = raw_vector_search(companies_collection, query, "companies_vector_index")
        logger.info("Companies query: %s, results: %d", query, len(results))
        return [
            {
                "content": r.page_content,
                "metadata": convert_to_serializable(r.metadata),
                "score": r.metadata.get("score", 0)
            }
            for r in results
        ]
    except Exception as e:
        logger.error("Companies search error: %s", str(e))
        return []
async def get_properties_by_context(query: str, properties_collection=None) -> List[dict]:
    """Get properties by context."""
    try:
        if properties_collection is None:
            raise ValueError("Properties collection not provided")
        query_embedding = embedmodel.embed_query(query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "properties_vector_index",
                    "path": "revoemb",
                    "queryVector": query_embedding,
                    "numCandidates":  100,
                    "limit": 10
                }
            },
            {
                "$project": {
                    "revoemb": 0,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        results = list(properties_collection.aggregate(pipeline))
        
        # Convert ObjectId fields to strings
        for result in results:
            if '_id' in result:
                result['_id'] = str(result['_id'])
            if 'companyId' in result:
                result['companyId'] = str(result['companyId'])
            if 'userId' in result:
                result['userId'] = str(result['userId'])
            # Add other ObjectId fields as needed (e.g., purchaseId)
            if 'purchaseId' in result:
                result['purchaseId'] = str(result['purchaseId'])
        
        logger.info("Properties by context query: %s, results: %d", query, len(results))
        return results
    except Exception as e:
        logger.error("Properties by context error: %s", str(e))
        return []