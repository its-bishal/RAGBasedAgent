import uvicorn
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status

# LangChain and Redis imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings

from qdrant_client import QdrantClient, models

import sqlite3

from models import AgentQueryRequest, AgentQueryResponse, BookingRequest, BookingResponse, SimilarityComparisonRequest, SimilarityComparisonResponse, SimilarityComparisonResult
from tools import retrieve_context, book_interview_tool, send_email_confirmation
from dbinit import init_db

load_dotenv()

DATABASE_FILE = "bookings.db"

# --- Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "QDRANT_PORT"))

REDIS_HOST = os.getenv("REDIS_HOST", "REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "REDIS_PORT"))

EMAIL_SENDER = os.getenv("EMAIL_SENDER", "EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "EMAIL_RECEIVER")

init_db()

app = FastAPI(
    title="RAG Agent & Interview Booking System",
    description="A backend system with a RAG-based agent, interview booking, and Qdrant similarity comparison.",
    version="1.0.0",
)

llm = ChatOllama(model="llama2")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

tools = [retrieve_context, book_interview_tool]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can answer questions and book interviews. "
                   "You have access to the following tools: {tools}\n\n"
                   "The way you use the tools is by specifying a json blob.\n"
                   "Specifically, you must call a tool by following this format:\n\n"
                   "```json\n{{\"action\": \"tool_name\", \"action_input\": \"tool input\"}}\n```\n\n"
                   "Valid tool names: {tool_names}\n\n"
                   "Always use the tools to answer questions. If you need to book an interview, use the 'book_interview_tool' and ask for all necessary details (full name, email, date, time) if not provided."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Add memory to the agent
def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """Returns a RedisChatMessageHistory instance for a given session ID."""
    return RedisChatMessageHistory(session_id=session_id, url=f'redis://{REDIS_HOST}:{REDIS_PORT}/0')

conversational_agent_chain = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- Document Loading and Qdrant Population (for RAG) ---
def load_and_embed_documents(collection_name: str, distance_metric: models.Distance):
    """
    Loads sample documents and embeds them into Qdrant for a given collection and distance metric.
    """
    print(f"Initializing Qdrant collection '{collection_name}' with {distance_metric} metric...")
    # Sample documents for RAG
    documents = [
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals.",
        "Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed.",
        "Deep learning is a specialized subset of machine learning that uses neural networks with many layers.",
        "The best way to learn programming is by doing hands-on projects.",
        "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.",
        "Qdrant is a vector similarity search engine and vector database. It provides a production-ready service with a convenient API to store, search, and manage points (vectors with attached payload).",
        "Redis is an open-source, in-memory data structure store, used as a database, cache, and message broker. It supports data structures such as strings, hashes, lists, sets, sorted sets with range queries, bitmaps, hyperloglogs, and geospatial indexes with radius queries.",
        "LangChain is a framework designed to simplify the creation of applications using large language models. It provides tools for chaining together different components to build more complex use cases.",
        "LangGraph is a library for building stateful, multi-actor applications with LLMs, inspired by Apache Airflow and other workflow orchestration tools."
    ]

    # Check if collection exists, delete if it does to ensure fresh start for comparison
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection '{collection_name}'.")
    except Exception as e:
        print(f"Collection '{collection_name}' did not exist or could not be deleted: {e}")

    # Create collection with specified distance metric
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings.client.get_sentence_embedding_dimension(), distance=distance_metric),
    )

    # Add documents to Qdrant
    points = []
    for i, doc_content in enumerate(documents):
        embedding = embeddings.embed_query(doc_content)
        points.append(
            models.PointStruct(
                id=i,
                vector=embedding,
                payload={"page_content": doc_content}
            )
        )
    qdrant_client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points,
    )
    print(f"Loaded {len(documents)} documents into Qdrant collection '{collection_name}'.")

# Load documents for the main RAG agent (using Cosine similarity by default)
# This will be the 'rag_documents' collection used by the `retrieve_context` tool.
load_and_embed_documents("rag_documents", models.Distance.COSINE)

# Load documents for comparison collections
load_and_embed_documents("rag_documents_cosine", models.Distance.COSINE)
load_and_embed_documents("rag_documents_euclid", models.Distance.EUCLID)


# --- FastAPI Endpoints ---
@app.post("/agent_query", response_model=AgentQueryResponse, status_code=status.HTTP_200_OK)
async def agent_query(request: AgentQueryRequest):
    """
    Endpoint to interact with the RAG-based agent.
    Provide a query and a session_id for conversational memory.
    """
    try:
        response = await conversational_agent_chain.ainvoke(
            {"input": request.query},
            config={"configurable": {"session_id": request.session_id}}
        )
        return AgentQueryResponse(response=response["output"])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing agent query: {e}"
        )

@app.post("/book_interview", response_model=BookingResponse, status_code=status.HTTP_201_CREATED)
async def book_interview(request: BookingRequest):
    """
    Sends a confirmation email and stores booking information.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO bookings (full_name, email, booking_date, booking_time) VALUES (?, ?, ?, ?)",
            (request.full_name, request.email, request.date, request.time)
        )
        booking_id = cursor.lastrowid
        conn.commit()
        conn.close()

        send_email_confirmation(request.full_name, request.email, request.date, request.time)

        return BookingResponse(
            message=f"Interview successfully booked for {request.full_name} on {request.date} at {request.time}. A confirmation email has been sent.",
            booking_id=booking_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to book interview: {e}"
        )

@app.post("/compare_similarity_algorithms", response_model=SimilarityComparisonResponse, status_code=status.HTTP_200_OK)
async def compare_similarity_algorithms(request: SimilarityComparisonRequest):
    """
    Compares two different similarity search algorithms (Cosine and Euclidean)
    supported by Qdrant for a given query.
    """
    query_vector = embeddings.embed_query(request.query_text)
    comparison_results = []

    # --- Cosine Similarity Comparison ---
    try:
        cosine_search_results = qdrant_client.search(
            collection_name="rag_documents_cosine",
            query_vector=query_vector,
            limit=request.top_k,
            with_payload=True
        )
        cosine_formatted_results = []
        for hit in cosine_search_results:
            cosine_formatted_results.append({
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload.get("page_content", "N/A")
            })
        comparison_results.append(SimilarityComparisonResult(
            metric="Cosine",
            results=cosine_formatted_results
        ))
    except Exception as e:
        print(f"Error during Cosine similarity search: {e}")
        comparison_results.append(SimilarityComparisonResult(
            metric="Cosine",
            results=[],
            message=f"Error: {e}"
        ))

    # --- Euclidean Distance Comparison ---
    try:
        euclid_search_results = qdrant_client.search(
            collection_name="rag_documents_euclid",
            query_vector=query_vector,
            limit=request.top_k,
            with_payload=True
        )
        euclid_formatted_results = []
        for hit in euclid_search_results:
            euclid_formatted_results.append({
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload.get("page_content", "N/A")
            })
        comparison_results.append(SimilarityComparisonResult(
            metric="Euclidean",
            results=euclid_formatted_results
        ))
    except Exception as e:
        print(f"Error during Euclidean distance search: {e}")
        comparison_results.append(SimilarityComparisonResult(
            metric="Euclidean",
            results=[],
            message=f"Error: {e}"
        ))

    return SimilarityComparisonResponse(
        query=request.query_text,
        comparison_results=comparison_results
    )

@app.get("/")
async def read_root():
    """Root endpoint for basic health check."""
    return {"message": "RAG Agent and Interview Booking System is running!"}

