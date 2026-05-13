import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from vector_db import SmartHomeVectorDB
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from data import smart_home_data, smart_home_data_elife

MODEL = os.getenv("LLM_MODEL", "ollama/llama3.2:1b")
#MODEL = "ollama/llama3.2:1b"

# Initialize separate databases for standard and e-life members
db_standard = SmartHomeVectorDB(smart_home_data)
db_elife = SmartHomeVectorDB(smart_home_data_elife)

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
        tools=[troubleshooting_guide_search],
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
        # context=[main_task],
        # output_file="iot_recommendation_report.md", # how CrewAI does File Output/App Integration
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


def create_iot_full_flow_crew(topic: str, chat_history: str = "", rag_context_list: list = None, user_data: dict = None):
    if rag_context_list is None:
        rag_context_list = []

    # Shared dict for the email — populated by the tool and sent when filled
    email_payload: dict = {}

    user_info_str = ""
    is_elife = False
    if user_data:
        user_info_str = f"USER INFORMATION: {user_data}\n"
        is_elife = user_data.get("elife_member", False)

    # Choose the correct database based on membership
    current_db = db_elife if is_elife else db_standard
    tool_name = "Search Premium Smart Home Devices" if is_elife else "Search Smart Home Devices"

    @tool("Search Smart Home Devices") # @tool(tool_name)
    def search_smart_home_devices(query: str) -> str:
        """Useful to search for smart home devices and IoT products based on a user query.
        Returns a list of relevant products with descriptions, features, and prices."""
        results = current_db.search(query)
        rag_context_list.extend(results)
        if not results:
            return "No relevant products found."
        context = "\n".join([
            f"{r['name']}: {r['description']} (Features: {r['features']}, Price: {r['price']})"
            for r in results
        ])
        return context

    @tool("Generate Recommendation Email")
    def generate_recommendation_email(recommendations_summary: str) -> str:
        """Call this tool ONLY when the user has explicitly confirmed they are done and
        would like to receive an email with their device recommendations.
        Pass a clear summary of the recommended devices as the argument.
        The tool will prepare the email and confirm it is ready to be sent."""
        receiver = user_data.get("email", "") if user_data else ""
        name = user_data.get("name", "Valued Customer") if user_data else "Valued Customer"

        subject = "Your Smart Home Device Recommendations from e&"
        body = (
            f"Dear {name},\n\n"
            f"Thank you for using our Smart Home assistant. "
            f"Here is a summary of the devices we recommended for you:\n\n"
            f"{recommendations_summary}\n\n"
            f"Feel free to reach out if you have any questions or need further assistance.\n\n"
            f"Best regards,\n"
            f"e& Smart Home Team"
        )

        email_payload["receiver"] = receiver
        email_payload["subject"] = subject
        email_payload["email_body"] = body

        return (
            f"Email prepared for {receiver}. "
            "Please let the user know their recommendations have been compiled and the email is ready to be sent."
        )

    manager = Agent(
        role="IoT End-to-End Manager",
        goal="Route user queries to the appropriate agent based on their intent, ensuring a helpful and accurate response.",
        backstory=(
            "You are the central coordinator for an IoT device support and sales team. "
            "You analyze user input to determine if they are just making small talk, asking for a specific device, "
            "or looking for tailored recommendations based on their home. "
            "if they are looking for a device call the search_smart_home_devices agent to search for them in the database"
            "only theses devices are sold and acceptable to be given to the user"
            "Delegate the task to the right agent. "
            "After providing recommendations, always ask the user if they are done and if they would like an email summary. "
            "If the user confirms both, delegate to the Email Specialist to prepare the email."
        ),
        llm=MODEL,
        allow_delegation=True,
        verbose=True
    )

    email_agent = Agent(
        role="Email Specialist",
        goal="Prepare summary emails for users with their recommended IoT devices.",
        backstory=(
            "You are responsible for compiling all device recommendations into a professional email. "
            "You use the Generate Recommendation Email tool to prepare the final payload for the user."
        ),
        llm=MODEL,
        tools=[generate_recommendation_email],
        verbose=True
    )

    chitchat = Agent(
        role="IoT Chit-Chat & Guardrail Specialist",
        goal="Handle general greetings, small talk, and gracefully deflect non-IoT topics.",
        backstory=(
            "You are a friendly representative for an IoT smart home company. "
            "You greet users and respond to casual small talk. "
            "Crucially, if a user asks about anything NOT related to IoT or smart home devices "
            "(like politics, sports, general knowledge), "
            "you politely explain that you can only assist with smart home and IoT topics."
            "If asked say that you can not comment about other companies or other products"
        ),
        llm=MODEL,
        verbose=True
    )

    home_profiler = Agent(
        role="Home Profiler & Recommender",
        goal="Gather home details if missing and recommend tailored smart home devices.",
        backstory=(
            "You specialize in full-home IoT setups. "
            "If a user asks for broad recommendations without specifying their home details (number of bedrooms, bathrooms, living rooms,if there is a garden, etc.), "
            "you ask clarifying questions to build a profile. "
            "If the user has already provided a clear explanation of their home or if they ask for a specific group of products, "
            "do not ask for more details; instead, use the Search Smart Home Devices tool to provide recommendations tailored to their setup."
        ),
        llm=MODEL,
        tools=[search_smart_home_devices],
        verbose=True
    )

    direct_recommender = Agent(
        role="Direct Recommender",
        goal="Directly recommend specific IoT products based on direct user requests.",
        backstory=(
            "You are an expert in finding the exact smart home device a user wants. "
            "When a user asks for a specific product or category (e.g., 'a smart bulbs', 'a smart plugs', 'entertainment'), "
            "you use the Search Smart Home Devices tool to find it and provide a clear, direct recommendation without asking unnecessary questions."
        ),
        llm=MODEL,
        tools=[search_smart_home_devices],
        verbose=True
    )

    writer = Agent(
        role="Technical Writer",
        goal="Summarize the final response clearly, ensuring it is formatted well and user-friendly.",
        backstory="You are an experienced technical copywriter who formats outputs perfectly.",
        llm="ollama/llama3.2:1b",
        verbose=True
    )

    main_task = Task(
        description=f"""
        Analyze the user's query and provide the best response by delegating to the appropriate agent.
        
        {user_info_str}
        
        - If the user is making small talk or asking non-IoT questions, delegate to the Chit-Chat Specialist.
        - If the user is asking for broad recommendations and hasn't described their home, delegate to the Home Profiler to ask for details (bedrooms, bathrooms, etc.).
        - If the user describes their home or asks for a specific group of products, delegate to the Home Profiler to recommend devices.
        - If the user asks for a single specific product (e.g., 'smart bulb'), delegate to the Direct Recommender.

        POST-RECOMMENDATION EMAIL FLOW:
        - After any product recommendations have been provided in the chat history or in this turn,
          you MUST ask the user TWO questions at the end of your response:
            1. Are you done with your queries?
            2. Would you like us to send you an email summary of the recommended devices?
        - If the user's current message indicates they are done AND they want an email
          (e.g. 'yes', 'yes please', 'send it', 'go ahead'), delegate to the Email Specialist
          to compile a clear summary of all the products that were recommended during this conversation.
        - Only involve the Email Specialist once the user has explicitly confirmed both conditions.
        
        CHAT HISTORY:
        {chat_history}
        
        USER QUESTION:
        {topic}
        """,
        expected_output="A complete, accurate, and contextually appropriate response to the user's input.",
    )

    summary_task = Task(
        description="Format and write the final response to the user based on the manager's delegation result.",
        agent=writer,
        expected_output="A well-formatted, friendly, and concise response addressing the user's query."
    )

    return Crew(
        agents=[chitchat, home_profiler, direct_recommender, email_agent, writer],
        tasks=[main_task, summary_task],
        manager_agent=manager,
        process=Process.hierarchical,
        verbose=True,
        tracing=True
    ), email_payload
