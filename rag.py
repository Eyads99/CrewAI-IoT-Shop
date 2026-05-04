from vector_db import SmartHomeVectorDB
from crew import create_crew_with_context

def format_chat_history(messages, max_turns=5):
    # Keep last N turns to avoid token explosion
    trimmed = messages[-max_turns*2:]

    history = ""
    for msg in trimmed:
        role = msg["role"]
        content = msg["content"]
        history += f"{role.upper()}: {content}\n"

    return history


db = SmartHomeVectorDB()


def rag_chat(user_query: str, chat_history: list):
    #  retrieve context
    results = db.search(user_query)

    context = "\n".join([
        f"{r['name']}: {r['description']} (Features: {r['features']})"
        for r in results
    ])

    # format memory
    history_text = format_chat_history(chat_history)

    # combine everything
    full_context = f"""
    CHAT HISTORY:
    {history_text}

    RETRIEVED KNOWLEDGE:
    {context}
    """

    crew = create_crew_with_context(user_query, full_context) # manually run crew from here

    return crew.kickoff() # needed to actually run crew


def rag_query(user_query: str):
    # Retrieve relevant docs
    results = db.search(user_query)

    context = "\n".join([
        f"{r['name']}: {r['description']} (Features: {r['features']})"
        for r in results
    ])

    # Inject into Crew task
    crew = create_crew_with_context(user_query, context)

    return crew.kickoff()
