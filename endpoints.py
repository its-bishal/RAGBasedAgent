
from fastapi import FastAPI, HTTPException, status
from qdrant_client import QdrantClient
import sqlite3

from main import conversational_agent_chain, app, DATABASE_FILE, embeddings, qdrant_client
from models import AgentQueryRequest, AgentQueryResponse, BookingRequest, BookingResponse, SimilarityComparisonRequest, SimilarityComparisonResponse, SimilarityComparisonResult
from tools import retrieve_context, book_interview_tool, send_email_confirmation


# --- FastAPI Endpoints ---
@app.post("/agent_query", response_model=AgentQueryResponse, status_code=status.HTTP_200_OK)
async def agent_query(request: AgentQueryRequest):
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
    """Root endpoint for basic check."""
    return {"message": "RAG Agent and Interview Booking System is running!"}

