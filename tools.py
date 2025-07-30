
# --- Tools for the Agent ---
from langchain_core.tools import tool
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
from datetime import datetime
import sqlite3

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=os.getenv("QDRANT_PORT"), api_key=os.getenv("QDRANT_API_KEY"))

DATABASE_FILE = "bookings.db"

@tool
def retrieve_context(query: str) -> str:
    """Performs a similarity search on the 'rag_documents' collection."""
    try:
        vectorstore = Qdrant(
            client=qdrant_client,
            collection_name="rag_documents",
            embeddings=embeddings,
        )
        docs = vectorstore.similarity_search(query, k=3) # Retrieve top 3 relevant documents
        if not docs:
            return "No relevant context found."
        context = "\n\n".join([doc.page_content for doc in docs])
        return f"Retrieved context:\n{context}"
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return f"Failed to retrieve context due to an error: {e}"

@tool
def book_interview_tool(full_name: str, email: str, date: str, time: str) -> str:
    """Books an interview for the user."""
    try:
        datetime.strptime(date, "%Y-%m-%d")
        datetime.strptime(time, "%H:%M")

        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO bookings (full_name, email, booking_date, booking_time) VALUES (?, ?, ?, ?)",
            (full_name, email, date, time)
        )
        booking_id = cursor.lastrowid
        conn.commit()
        conn.close()

        send_email_confirmation(full_name, email, date, time)

        return f"Interview successfully booked for {full_name} on {date} at {time}. Booking ID: {booking_id}. A confirmation email has been sent to {email}."
    except ValueError:
        return "Invalid date or time format. Please use YYYY-MM-DD for date and HH:MM for time."
    except Exception as e:
        print(f"Error booking interview: {e}")
        return f"Failed to book interview due to an error: {e}"
    


def send_email_confirmation(full_name: str, recipient_email: str, date: str, time: str):
    """Sends an interview confirmation email."""
    if not os.getenv("EMAIL_SENDER") or not os.getenv("EMAIL_PASSWORD"):
        print("Credentials not configured. Skipping email sending.")
        return

    msg = MIMEMultipart()
    msg['From'] = os.getenv("EMAIL_SENDER")
    msg['To'] = os.getenv("EMAIL_RECEIVER")
    msg['Subject'] = "Interview Booking Confirmation"

    body = f"""
    Dear {full_name},

    This is a confirmation that your interview has been successfully booked.

    Details:
    Full Name: {full_name}
    Email: {recipient_email}
    Date: {date}
    Time: {time}

    We look forward to speaking with you!

    Best regards,
    The Interview Team
    """
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465) # Use SMTP_SSL for Gmail
        server.login(os.getenv("EMAIL_SENDER"), os.getenv("EMAIL_PASSWORD"))
        server.send_message(msg)
        server.quit()
        print(f'Confirmation email sent to {os.getenv("EMAIL_RECEIVER")} for {full_name}.')
    except Exception as e:
        print(f"Failed to send email: {e}")
        print("Please ensure EMAIL_SENDER and EMAIL_PASSWORD are correct..")

