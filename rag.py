from vector_db import SmartHomeVectorDB
from crew import create_iot_crew, create_iot_crew_planner, create_iot_full_flow_crew

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
    # format memory
    history_text = format_chat_history(chat_history)

    # Run crew with memory
    rag_context_list = []
    crew = create_iot_crew(user_query, history_text, rag_context_list=rag_context_list)

    result = crew.kickoff() # needed to actually run crew
    return result, rag_context_list


def rag_chat_planner(user_query: str, chat_history: list):
    """Use the hierarchical crew planner for combined recommendation + troubleshooting."""
    history_text = format_chat_history(chat_history)

    rag_context_list = []
    crew = create_iot_crew_planner(user_query, history_text, rag_context_list=rag_context_list)

    result = crew.kickoff()
    return result, rag_context_list


def rag_query(user_query: str):
    # Retrieve relevant docs
    results = db.search(user_query)

    context = "\n".join([
        f"{r['name']}: {r['description']} (Features: {r['features']}, Price: {r['price']})"
        for r in results
    ])

    # Inject into Crew task
    crew = create_crew_with_context(user_query, context)

    return crew.kickoff()


def rag_chat_full_flow(user_query: str, chat_history: list, user_data: dict = None):
    """Use the hierarchical full flow crew for end-to-end interactions."""
    history_text = format_chat_history(chat_history)

    rag_context_list = []
    crew, email_payload = create_iot_full_flow_crew(user_query, history_text, rag_context_list=rag_context_list, user_data=user_data)

    try:
        result = crew.kickoff()
    except Exception as e:
        print(f"Error during crew execution: {e}")
        # Human in the loop fallback message if the crew crashes
        result = "I apologize, but I've encountered an unexpected system error."
        
    return result, rag_context_list, email_payload
