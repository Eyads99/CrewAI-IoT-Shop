import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from vector_db import SmartHomeVectorDB
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

MODEL = os.getenv("LLM_MODEL", "ollama/llama3.2:1b")
#MODEL = "ollama/llama3.2:1b"

db = SmartHomeVectorDB()

def create_crew_basic(topic: str):
    researcher = Agent(
        role="Researcher",
        goal=f"Find useful facts about {topic}",
        backstory="You are a helpful researcher",
        llm=MODEL,
        verbose=True
    )


    writer = Agent(
        role="Writer",
        goal="Write a concise summary",
        backstory="You summarize clearly",
        llm=MODEL,
        verbose=True
    )

    task1 = Task(
        description=f"Find 3 interesting facts about {topic}",
        expected_output=f"A bullet point list with 3 facts about {topic}",
        agent=researcher
    )

    task2 = Task(
        description="Write a short paragraph based on the research",
        expected_output="A short paragraph summarizing the facts",
        agent=writer
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        verbose=True
    )


def create_iot_crew(topic: str, chat_history: str = "", rag_context_list: list = None):
    if rag_context_list is None:
        rag_context_list = []

    @tool("Search Smart Home Devices")
    def search_smart_home_devices(query: str) -> str:
        """Useful to search for smart home devices and IoT products based on a user query.
        Returns a list of relevant products with descriptions, features, and prices."""
        results = db.search(query)
        rag_context_list.extend(results)
        if not results:
            return "No relevant products found."
        context = "\n".join([
            f"{r['name']}: {r['description']} (Features: {r['features']}, Price: {r['price']})"
            for r in results
        ])
        return context

    researcher = Agent(
        role="Smart Home Expert",
        goal="Answer the user's questions about smart home devices. Use the Search Smart Home Devices tool if you need "
             "to find specific products.",
        backstory="Expert in IoT devices and smart homes for Emirati telecommunications company e&, you help users with"
                  " their queries related to IoT devices. You decide whether a query requires searching for "
                  "specific products or can be answered directly.",
        llm=MODEL,
        tools=[search_smart_home_devices],
        verbose=True
    )

    writer = Agent(
        role="Technical Writer",
        goal="Summarize smart home information clearly and succinctly apologizing when information is not available, "
             "or if the topic is not related to IoT devices ",
        backstory="Writes concise product summaries",
        llm='ollama/llama3.2:1b',
        verbose=True
    )

    guardrail = Agent(
        role="IoT Conversation Moderator",
        goal=(
            "Ensure all responses are strictly about IoT and smart home devices. "
            "Reject or correct anything outside this scope"
        ),
        backstory=(
            "You are a moderator specializing in IoT systems and smart home devices. "
            "Your responsibilities:\n"
            "- Allow ONLY topics related to IoT, connected devices, automation, or smart home technology, "
            "  or related requests like users asking what kind of product may be best for them\n"
            "- Reject any non-IoT topics (e.g., politics, sports, general knowledge, unrelated tech)\n"
            "- Remove hallucinations or unsupported claims\n"
            "If the user asks about a NON-IoT topic:\n"
            "- DO NOT attempt to answer it\n"
            "- Respond with a brief apology and redirect\n\n"
            "Response format for off-topic queries:\n"
            "'Sorry, I can only help with IoT and smart home device-related questions. "
            "Please ask something within that scope.'\n\n"
            "If the response is partially off-topic:\n"
            "- Remove irrelevant parts\n"
            "- Keep only IoT-relevant content\n"
        ),
        llm=MODEL,
        verbose=True
    )

    task1 = Task(
        description=f"""
        Analyze the user's query and respond appropriately. If they are asking for recommendations or specific products,
         use your tool to search for them. If it's a general query, answer directly.

        CHAT HISTORY:
        {chat_history}

        USER QUESTION:
        {topic}
        """,
        agent=researcher,
        expected_output='A helpful response to the user query, utilizing product information if relevant.'
    )

    task2 = Task(
        description=f"""
        Review the previous response.

        USER QUESTION:
        {topic}

        Your responsibilities:

        1. Scope enforcement:
           - The response should be about IoT (Internet of Things) or smart devices.
           - IoT includes: smart home devices, connected appliances, sensors, automation systems, smart plugs, smart light bulb, etc.


        3. If the response contains:
           - Off-topic content → REMOVE it
           - Hallucinated or unsupported claims → REMOVE or CORRECT them

        4. If the response is valid but noisy:
           - Rewrite it to be concise, accurate, and focused on IoT devices only

        Output rules:
        - Return ONLY the final cleaned or rejection response
        - Do NOT explain your reasoning
        """,
        agent=guardrail,
        expected_output=(
            "Either a corrected IoT-focused response OR a short apology stating that only IoT topics are supported."
        )
    )

    task3 = Task(
        description="Write a clear final response to the user based on the preceding information.",
        agent=writer,
        expected_output='A clear, friendly, and concise response to the user.'
    )

    return Crew(
        agents=[researcher, guardrail, writer],
        tasks=[task1, task2, task3],
        verbose=True
    )



class IoTResponse(BaseModel):
    recommended_products: List[str] = Field(
        description="Names of the recommended products, empty list if none apply"
    )
    reasoning: str = Field(
        description="Why these products or steps were chosen for the user's needs"
    )
    price_range: str = Field(
        description="Approximate price range of any recommendations, or 'N/A' if not applicable"
    )
    troubleshooting_steps: Optional[List[str]] = Field(
        default=None,
        description="Numbered step-by-step troubleshooting instructions if the user had a device issue"
    )


def create_iot_crew_planner(topic: str, chat_history: str = "", rag_context_list: list = None):
    if rag_context_list is None:
        rag_context_list = []

    @tool("Search Smart Home Devices")
    def search_smart_home_devices(query: str) -> str:
        """Useful to search for smart home devices and IoT products based on a user query.
        Returns a list of relevant products with descriptions, features, and prices."""
        results = db.search(query)
        rag_context_list.extend(results)
        if not results:
            return "No relevant products found."
        context = "\n".join([
            f"{r['name']}: {r['description']} (Features: {r['features']}, Price: {r['price']})"
            for r in results
        ])
        return context

    @tool("Troubleshooting Knowledge Base")
    def troubleshooting_guide_search(query: str) -> str:
        """Search the support knowledge base for step-by-step troubleshooting
        instructions for common IoT device problems: going offline, resetting,
        being unresponsive, or installation failures.
        Use this when the user reports a problem with an existing device."""
        knowledge_base = {
            "offline": "1. Check the power source. 2. Restart the router. 3. Re-pair the device.",
            "reset": "1. Hold the reset button for 10 seconds. 2. Wait for the LED to blink rapidly.",
            "unresponsive": "1. Ensure the device is connected to 2.4GHz Wi-Fi. 2. Check for firmware updates.",
            "install": "1. Download the app. 2. Follow in-app instructions to add device. 3. Enter Wi-Fi credentials.",
        }
        query_lower = query.lower()
        for key, steps in knowledge_base.items():
            if key in query_lower:
                return f"Troubleshooting for '{key}': {steps}"
        return "General troubleshooting: 1. Restart device. 2. Check internet connection. 3. Contact support."

    recommender = Agent(
        role="Product Recommender",
        goal="Recommend smart home and IoT products based on user needs.",
        backstory="You are an expert in IoT product specifications, features, and pricing. You help customers find the right devices by searching the product database.",
        llm=MODEL,
        tools=[search_smart_home_devices],
        verbose=True
    )

    troubleshooter = Agent(
        role="IoT Troubleshooter",
        goal="Diagnose and resolve issues with smart home and IoT devices.",
        backstory=(
            "You are a technical support specialist. You use the troubleshooting "
            "knowledge base to give users clear, step-by-step resolution paths."
        ),
        llm=MODEL,
        tools=[troubleshooting_guide_search],  # scoped to its own tool
        verbose=True
    )

    manager = Agent(
        role="IoT Crew Manager",
        goal="Determine if the user needs product recommendations or troubleshooting, plan the execution, and delegate to the appropriate agents to provide a unified response.",
        backstory="You are the manager of an IoT support and sales team. You analyze user queries, decide whether they need troubleshooting help or product recommendations, delegate tasks to your team, and assemble the final response.",
        llm=MODEL,
        allow_delegation=True,
        verbose=True
    )

    writer = Agent(
        role="Technical Writer",
        goal="Summarize  information clearly and succinctly apologizing when information is not available",
        backstory="You are an experienced copywriter",
        llm='ollama/llama3.2:1b',
        verbose=True
    )

    main_task = Task(
        description=f"""
        Analyze the user's query and provide a comprehensive response.
        If they need a product, delegate to the Product Recommender.
        If they have an issue with a device, delegate to the IoT Troubleshooter.
        Ensure the final answer is clear, helpful, and directly addresses the user's needs.

        CHAT HISTORY:
        {chat_history}

        USER QUESTION:
        {topic}
        """,
        expected_output="A helpful, accurate, and clear response to the user's IoT query, either recommending a product or providing troubleshooting steps.",
    )

    summary_task = Task(
        description="Write a clear final response to the user based on the preceding information.",
        agent=writer,
        # context=[main_task], # Showcasing CrewAI Context Feature
        # output_file="iot_recommendation_report.md", # Showcasing CrewAI File Output/App Integration
        expected_output='A formatted, clear, friendly, and concise response to the user, in no more than 100 words.'
    )

    return Crew(
        agents=[recommender, troubleshooter, writer],
        tasks=[main_task,summary_task],
        manager_agent=manager,
        process=Process.hierarchical,
        #planning=True,
        #planning_llm='ollama/gemma4:26b',
        #step_callback=take action after every agent action
        verbose=True,
        tracing=True
    )

