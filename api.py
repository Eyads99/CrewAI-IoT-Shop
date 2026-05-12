from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from flow import ResearchFlow
from rag import rag_query, rag_chat, rag_chat_planner, rag_chat_full_flow
from chat_store import chat_store
from data import get_user_by_email
import re

app = FastAPI()

class RequestModel(BaseModel):
    topic: str
class Query(BaseModel):
    question: str

@app.get("/")
@app.get("/health")
def root():
    return {"message": "CrewAI API is running"}

@app.post("/run")
def run_flow(request: RequestModel):
    flow = ResearchFlow(topic=request.topic)
    result = flow.kickoff()

    return {
        "topic": request.topic,
        "result": result
    }

@app.post("/rag") # basic rag no chat history
def rag_endpoint(query: Query):
    result = rag_query(query.question)

    return {
        "query": query.question,
        "result": result
    }


class ChatRequest(BaseModel):
    chat_id: str
    message: str
    email: str = None

@app.post("/chat/create")
def create_chat():
    chat_id = chat_store.create_chat()
    return {"chat_id": chat_id}

@app.post("/chat/send")
def send_message(req: ChatRequest):
    if req.chat_id not in chat_store.chats:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Save user message
    chat_store.add_message(req.chat_id, "user", req.message)

    # Get history
    history = chat_store.get_messages(req.chat_id)

    # Run RAG & Crew
    response, rag_context = rag_chat(req.message, history)

    # Save AI response
    chat_store.add_message(req.chat_id, "AI", str(response), rag_context=rag_context)

    return {
        "chat_id": req.chat_id,
        "response": str(response)
    }

@app.post("/chat/send/planner")
def send_message_planner(req: ChatRequest):
    """Uses the hierarchical crew planner for combined recommendation + troubleshooting."""
    if req.chat_id not in chat_store.chats:
        raise HTTPException(status_code=404, detail="Chat not found")

    chat_store.add_message(req.chat_id, "user", req.message)
    history = chat_store.get_messages(req.chat_id)

    response, rag_context = rag_chat_planner(req.message, history)

    chat_store.add_message(req.chat_id, "AI", str(response), rag_context=rag_context)

    return {
        "chat_id": req.chat_id,
        "response": str(response)
    }

@app.post("/chat/send/full_flow")
def send_message_full_flow(req: ChatRequest):
    """Uses the hierarchical full flow crew for end-to-end interactions."""
    if req.chat_id not in chat_store.chats:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Email validation
    if not req.email:
        raise HTTPException(status_code=400, detail="You must provide an email.")
    
    # Basic email regex
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if not re.match(email_regex, req.email):
        raise HTTPException(status_code=400, detail="Invalid email format.")

    user_info = get_user_by_email(req.email)
    if not user_info:
        raise HTTPException(status_code=400, detail="Email not found in database.")

    chat_store.add_message(req.chat_id, "user", req.message)
    history = chat_store.get_messages(req.chat_id)

    response, rag_context = rag_chat_full_flow(req.message, history, user_data=user_info)

    chat_store.add_message(req.chat_id, "AI", str(response), rag_context=rag_context)

    return {
        "chat_id": req.chat_id,
        "response": str(response)
    }

@app.get("/chat/{chat_id}/history")
def get_chat_history(chat_id: str):
    if chat_id not in chat_store.chats:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    history = chat_store.get_messages(chat_id)
    return {"chat_id": chat_id, "history": history}