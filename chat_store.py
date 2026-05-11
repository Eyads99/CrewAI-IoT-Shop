import uuid
from datetime import datetime

class ChatStore:
    def __init__(self):
        self.chats = {}

    def create_chat(self):
        chat_id = str(uuid.uuid4())
        self.chats[chat_id] = {
            "messages": [],
            "created_at": datetime.utcnow()
        }
        return chat_id

    def add_message(self, chat_id, role, content, rag_context=None):
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
        }
        if rag_context is not None:
            entry["rag_context"] = rag_context
        self.chats[chat_id]["messages"].append(entry)

    def get_messages(self, chat_id):
        return self.chats.get(chat_id, {}).get("messages", [])

chat_store = ChatStore()
