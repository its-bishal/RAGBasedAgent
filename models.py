from pydantic import BaseModel,EmailStr, Field
from typing import List, Dict, Any, Optional
from fastapi import FastAPI


# --- Pydantic Models ---
class AgentQueryRequest(BaseModel):
    """Request model for agent queries."""
    query: str
    session_id: str = Field(..., description="Unique session ID for conversational memory.")

class AgentQueryResponse(BaseModel):
    """Response model for agent queries."""
    response: str

class BookingRequest(BaseModel):
    """Request model for booking an interview."""
    full_name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date in YYYY-MM-DD format.")
    time: str = Field(..., pattern=r"^\d{2}:\d{2}$", description="Time in HH:MM format.")

class BookingResponse(BaseModel):
    """Response model for interview booking confirmation."""
    message: str
    booking_id: Optional[int] = None

class SimilarityComparisonRequest(BaseModel):
    """Request model for similarity search comparison."""
    collection_name: str = "rag_documents"
    query_text: str
    top_k: int = 5

class SimilarityComparisonResult(BaseModel):
    """Result model for a single similarity search."""
    metric: str
    results: List[Dict[str, Any]]

class SimilarityComparisonResponse(BaseModel):
    """Response model for similarity search comparison."""
    query: str
    comparison_results: List[SimilarityComparisonResult]

# --- FastAPI App Initialization ---
app = FastAPI(
    title="RAG Agent & Interview Booking System",
    description="A backend system with a RAG-based agent, interview booking, and Qdrant similarity comparison.",
    version="1.0.0",
)

