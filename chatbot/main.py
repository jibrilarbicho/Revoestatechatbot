from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routes import router as chatbot_router
import os
from pymongo import MongoClient

# Load variables from .env file
load_dotenv()

# Get MongoDB connection string from environment
CONNECTION_STRING = os.getenv("MongoURI")
app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_db_client():
    try:
        app.mongodb_client = MongoClient(CONNECTION_STRING)
        app.database = app.mongodb_client['revoestate']
        app.mongodb_client.server_info()
        print("Connected successfully to the database!")
    except Exception as e:
        print(f"Failed to connect to the database: {str(e)}")
        app.mongodb_client = None
        app.database = None

@app.on_event("shutdown")
def shutdown_db_client():
    app.mongodb_client.close()
    print("Database connection closed.")
app.include_router(chatbot_router, tags=["chatbot"], prefix="/chatbot")



