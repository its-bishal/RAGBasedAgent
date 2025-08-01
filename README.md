RAG Agent & Interview Booking SystemThis project implements a robust backend system using FastAPI, featuring a RAG (Retrieval-Augmented Generation) based agent powered by LangChain, a memory layer with Redis, and a complete interview booking process. Additionally, it includes a comparative analysis of Qdrant's similarity search algorithms.

# Features
RAG-based Agent: An intelligent agent capable of answering user queries by retrieving relevant context from a vector database.
Conversational Memory: Utilizes Redis to maintain conversational context across multiple turns, enabling more natural interactions.Interview Booking System:Captures full name, email, date, and time for interview appointments.Stores booking information in a local SQLite database.Sends automated confirmation emails to a specified receiver.

# Setup Instructions
Clone the RepositoryFirst, clone this project to your local machine:git clone repository_url 
cd <project_directory>     
Install Dependencies from requirements.txt file:
pip install -r requirements.txt

# create a .env file and store the values for following fields
QDRANT_HOST="localhost"
QDRANT_PORT=6333
QDRANT_API_KEY="" # Optional

# Redis Configuration
REDIS_HOST="localhost"
REDIS_PORT=6379

EMAIL_SENDER="sending_email@gmail.com"
EMAIL_PASSWORD="email_password"
EMAIL_RECEIVER="receiving_email@example.com"
Run QdrantQdrant is used as the vector database. The easiest way to run it is via Docker:docker run -p 6333:6333 qdrant/qdrant or you could download and run the standalone executable from the Qdrant GitHub.

Start the FastAPI application:uvicorn main:app --reload
You should see output indicating that Uvicorn is running on http://127.0.0.1:8000.API Endpoint.

# Agent Query
Interacts with the RAG-based agent, leveraging conversational memory and tools.Endpoint: /agent_queryMethod: POSTRequest Body (JSON):{
  "query": "What is FastAPI?",
  "session_id": "user_session_abc123"
}
curl command (Linux/macOS):curl -X POST -H "Content-Type: application/json" -d '{
  "query": "What is FastAPI?",
  "session_id": "user_session_abc123"
}' http://127.0.0.1:8000/agent_query
curl command (Windows cmd/PowerShell):curl -X POST -H "Content-Type: application/json" -d "{ \"query\": \"What is FastAPI?\", \"session_id\": \"user_session_abc123\" }" http://127.0.0.1:8000/agent_query

# Book Interview
Books an interview appointment and sends a confirmation email.Endpoint: /book_interviewMethod: POSTRequest Body (JSON):{
  "full_name": "Jane Doe",
  "email": "jane.doe@example.com",
  "date": "2025-08-30",
  "time": "15:30"
}
curl command (Linux/macOS):curl -X POST -H "Content-Type: application/json" -d '{
  "full_name": "Jane Doe",
  "email": "jane.doe@example.com",
  "date": "2025-08-30",
  "time": "15:30"
}' http://127.0.0.1:8000/book_interview
curl command (Windows cmd/PowerShell):curl -X POST -H "Content-Type: application/json" -d "{ \"full_name\": \"Jane Doe\", \"email\": \"jane.doe@example.com\", \"date\": \"2025-08-30\", \"time\": \"15:30\" }" http://127.0.0.1:8000/book_interview

# Compare Similarity Algorithms
Compares Cosine and Euclidean similarity search results in Qdrant for a given query.Endpoint: /compare_similarity_algorithmsMethod: POSTRequest Body (JSON):{
  "query_text": "What is AI?",
  "top_k": 3
}
curl command (Linux/macOS):curl -X POST -H "Content-Type: application/json" -d '{
  "query_text": "What is AI?",
  "top_k": 3
}' http://127.0.0.1:8000/compare_similarity_algorithms
Example curl command (Windows cmd/PowerShell):curl -X POST -H "Content-Type: application/json" -d "{ \"query_text\": \"What is AI?\", \"top_k\": 3 }" http://127.0.0.1:8000/compare_similarity_algorithms
