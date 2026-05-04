from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from flow import ResearchFlow
from rag import rag_query, rag_chat
from chat_store import chat_store

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

    # Run RAG + Crew
    response = rag_chat(req.message, history)

    # Save AI response
    chat_store.add_message(req.chat_id, "AI", str(response))

    return {
        "chat_id": req.chat_id,
        "response": response
    }