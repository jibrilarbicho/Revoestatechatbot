# import logging
# from langchain_core.tools import tool
# from langchain_core.documents import Document
# from langchain_huggingface import HuggingFaceEmbeddings
# from serialization import convert_to_serializable
# from typing import List
# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# import os
# os.environ["HF_HOME"] = "/app/.cache"

# # Initialize embeddings
# embedmodel = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Raw vector search function
# def raw_vector_search(collection, query: str, index_name: str, k: int = 10) -> List[Document]:
#     try:
#         query_embedding = embedmodel.embed_query(query)
#         pipeline = [
#             {
#                 "$vectorSearch": {
#                     "index": index_name,
#                     "path": "revoemb",
#                     "queryVector": query_embedding,
#                     "numCandidates": k * 10,
#                     "limit": k
#                 }
#             },
#             {
#                 "$project": {
#                     "revoemb": 0,
#                     "score": {"$meta": "vectorSearchScore"}
#                 }
#             }
#         ]
#         results = list(collection.aggregate(pipeline))
#         return [Document(
#             page_content=r.get("description", ""),
#             metadata={k: v for k, v in r.items() if k != "description"},
#             score=r.get("score", 0)
#         ) for r in results]
#     except Exception as e:
#         logger.error("Vector search error: %s", str(e))
#         return []

# # Define tools
# @tool
# def properties_vector_search(query: str, properties_collection=None) -> List[dict]:
#     """Search for real estate properties based on a query."""
#     try:
#         if properties_collection is None:
#             raise ValueError("Properties collection not provided")
#         results = raw_vector_search(properties_collection, query, "properties_vector_index")
#         logger.info("Properties query: %s, results: %d", query, len(results))
#         return [
#             {
#                 "content": r.page_content,
#                 "metadata": convert_to_serializable(r.metadata),
#                 "score": r.metadata.get("score", 0)
#             }
#             for r in results
#         ]
#     except Exception as e:
#         logger.error("Properties search error: %s", str(e))
#         return []

# @tool
# def companies_vector_search(query: str, companies_collection=None) -> List[dict]:
#     """Search for real estate companies based on a query."""
#     try:
#         if companies_collection is None:
#             raise ValueError("Companies collection not provided")
#         results = raw_vector_search(companies_collection, query, "companies_vector_index")
#         logger.info("Companies query: %s, results: %d", query, len(results))
#         return [
#             {
#                 "content": r.page_content,
#                 "metadata": convert_to_serializable(r.metadata),
#                 "score": r.metadata.get("score", 0)
#             }
#             for r in results
#         ]
#     except Exception as e:
#         logger.error("Companies search error: %s", str(e))
#         return []
# @tool
# def revoestate_information(query: str, revoestate_collection=None) -> List[dict]:
#     """Search for Revoestate information based on a query and to get information about systsem."""

#     try:
#         if revoestate_collection is None:
#             raise ValueError("Revoestate collection not provided")
#         query_embedding = embedmodel.embed_query(query)
#         index_name = "revoinformation_vector_index"

#         pipeline = [
#             {
#                 "$vectorSearch": {
#                     "index": index_name,
#                     "queryVector": query_embedding,
#                     "path": "revoemb",
#                     "limit": 5,
#                     "numCandidates": 100  # Added required parameter for approximate search
#                 }
#             },
#             {
#                 "$project": {
#                     "_id": 0,
#                     "text": 1,
#                     "revoemb": 0,
#                     "score": {"$meta": "vectorSearchScore"}  # Include relevance score
#                 }
#             }
#         ]

#         results = revoestate_collection.aggregate(pipeline)
#         return list(results)
#     except Exception as e:
#         logger.error("Revoestate search error: %s", str(e))
#         return []
# async def get_properties_by_context(query: str, properties_collection=None) -> List[dict]:
#     """Get properties by context."""
#     try:
#         if properties_collection is None:
#             raise ValueError("Properties collection not provided")
#         query_embedding = embedmodel.embed_query(query)
#         pipeline = [
#             {
#                 "$vectorSearch": {
#                     "index": "properties_vector_index",
#                     "path": "revoemb",
#                     "queryVector": query_embedding,
#                     "numCandidates":  100,
#                     "limit": 10
#                 }
#             },
#             {
#                 "$project": {
#                     "revoemb": 0,
#                     "score": {"$meta": "vectorSearchScore"}
#                 }
#             }
#         ]
#         results = list(properties_collection.aggregate(pipeline))
        
#         # Convert ObjectId fields to strings
#         for result in results:
#             if '_id' in result:
#                 result['_id'] = str(result['_id'])
#             if 'companyId' in result:
#                 result['companyId'] = str(result['companyId'])
#             if 'userId' in result:
#                 result['userId'] = str(result['userId'])
#             # Add other ObjectId fields as needed (e.g., purchaseId)
#             if 'purchaseId' in result:
#                 result['purchaseId'] = str(result['purchaseId'])
        
#         logger.info("Properties by context query: %s, results: %d", query, len(results))
#         return results
#     except Exception as e:
#         logger.error("Properties by context error: %s", str(e))
#         return []
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
def raw_vector_search(collection, query: str, index_name: str, exclude_fields: List[str] = [], k: int = 10) -> List[Document]:
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
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"}
                }
            },
            {
                "$project": {
                    **{field: 0 for field in exclude_fields},
                }
            }
        ]
        results = list(collection.aggregate(pipeline))
        return [
            Document(
                page_content=r.get("description", ""),
                metadata={k: v for k, v in r.items() if k != "description"},
                score=r.get("score", 0)
            )
            for r in results
        ]
    except Exception as e:
        logger.error("Vector search error: %s", str(e))
        return []
# Define tools
@tool
def properties_vector_search(query: str, properties_collection=None) -> List[dict]:
    """Search for real estate properties in Addis Ababa, Ethiopia, based on a user query.
    
    This tool searches a collection of properties including homes, apartments, villas, condos, and more.
    It returns detailed information such as title, price, location (with subcity/district and coordinates if available),
    specifications (bedrooms, bathrooms, area, built year), amenities, and descriptions.
    
    Args:
        query (str): The user's search query (e.g., "apartments in Bole" or "villas with 3 bedrooms").
        properties_collection: The database collection containing property data (defaults to None).
    
    Returns:
        List[dict]: A list of dictionaries, each containing:
            - content (str): The property description.
            - metadata (dict): Property details (e.g., price, location, bedrooms).
            - score (float): Relevance score of the match.
    
    Raises:
        ValueError: If properties_collection is not provided.
        Exception: If the search fails due to database or processing errors.
    """ 
    try:
        if properties_collection is None:
            raise ValueError("Properties collection not provided")
        results = raw_vector_search(properties_collection, query, "properties_vector_index",exclude_fields=["images", "panoramicImages","revoemb"])
        logger.info("Properties query: %s, results: %d", query, len(results),results)
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
    """Search for real estate companies in Addis Ababa, Ethiopia, based on a user query.
    
    This tool retrieves information about real estate agencies or companies, including their name,
    services offered, contact details (phone, email, website), physical address, years in operation,
    and specializations. Use this when the user explicitly asks about a company (e.g., "Tell me about ABC Realty").
    
    Args:
        query (str): The user's search query (e.g., "real estate companies in Addis" or "ABC Realty details").
        companies_collection: The database collection containing company data (defaults to None).
    
    Returns:
        List[dict]: A list of dictionaries, each containing:
            - content (str): The company description.
            - metadata (dict): Company details (e.g., contact info, services).
            - score (float): Relevance score of the match.
    
    Raises:
        ValueError: If companies_collection is not provided.
        Exception: If the search fails due to database or processing errors.
    """
    try:
        if companies_collection is None:
            raise ValueError("Companies collection not provided")
        results = raw_vector_search(companies_collection, query, "companies_vector_index",exclude_fields=["revoemb"])
        logger.info("Companies query: %s, results: %d", query, len(results),results)
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

@tool
def revoestate_information(query: str, revoestate_collection=None) -> List[dict]:
    """Search for information about the Revoestate platform based on a user query.
    
    This tool provides details about Revoestate, including its mission, services (e.g., property listings,
    company profiles), role in Ethiopian real estate, how to use the platform (e.g., listing properties,
    searching for homes), and contact information. Use this for queries like "What is Revoestate?",
    "How do I use this website?", or "What services does Revoestate offer?".
    
    Args:
        query (str): The user's query about Revoestate (e.g., "What is Revoestate?" or "How to list a property").
        revoestate_collection: The database collection containing Revoestate data (defaults to None).
    
    Returns:
        List[dict]: A list of dictionaries, each containing:
            - text (str): Information about Revoestate.
            - score (float): Relevance score of the match.
    
    Raises:
        ValueError: If revoestate_collection is not provided.
        Exception: If the search fails due to database or processing errors.
    """
    try:
        if revoestate_collection is None:
            raise ValueError("Revoestate collection not provided")
        query_embedding = embedmodel.embed_query(query)
        index_name = "revoinformation_vector_index"

        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "queryVector": query_embedding,
                    "path": "revoemb",
                    "limit": 5,
                    "numCandidates": 100
                }
            },
            {
                "$project": {
                    "_id": 0,  # Explicitly exclude _id
                    "text": 1,  # Include text
                    "score": {"$meta": "vectorSearchScore"}  # Include score
                }
            }
        ]

        results = revoestate_collection.aggregate(pipeline)
        return list(results)
    except Exception as e:
        logger.error("Revoestate search error: %s", str(e))
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
                    "numCandidates": 100,
                    "limit": 10
                }
            },
            {
                "$project": {
                    "_id": 0,  # Explicitly exclude _id
                    "description": 1,  # Include description
                    "score": {"$meta": "vectorSearchScore"},  # Include score
                    # Include other fields as needed
                    "companyId": 1,
                    "userId": 1,
                    "purchaseId": 1
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
            if 'purchaseId' in result:
                result['purchaseId'] = str(result['purchaseId'])
        
        logger.info("Properties by context query: %s, results: %d", query, len(results))
        return results
    except Exception as e:
        logger.error("Properties by context error: %s", str(e))
        return []