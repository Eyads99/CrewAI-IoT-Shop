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


def _validate_phone_number(phone: str) -> bool:
    """Mock phone validation. Accepts any number that contains 7–15 digits (with optional +/spaces/dashes).
    In a real system this would trigger an OTP flow."""
    import re
    cleaned = re.sub(r'[\s\-().+]', '', phone)
    return cleaned.isdigit() and 7 <= len(cleaned) <= 15


def create_iot_full_flow_crew(topic: str, chat_history: str = "", rag_context_list: list = None, user_data: dict = None):
    if rag_context_list is None:
        rag_context_list = []

    email_payload: dict = {}

    user_info_str = ""
    is_elife = False
    if user_data:
        user_info_str = f"USER INFORMATION: {user_data}\n"
        is_elife = user_data.get("elife_member", False)

    current_db = db_elife if is_elife else db_standard
    # tool_name = "Search Premium Smart Home Devices" if is_elife else "Search Smart Home Devices"

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

    @tool("Verify Phone Number")
    def verify_phone_number(phone: str) -> str:
        """Call this tool to verify the user's phone number before generating the recommendation email.
        Pass the phone number string provided by the user as the argument.
        Returns a confirmation if the number is valid, or an error message if it is not."""
        if _validate_phone_number(phone):
            return (
                f"Phone number '{phone}' has been verified successfully. "
                "You may now proceed to generate the recommendation email."
            )
        return (
            f"Phone number '{phone}' is not valid. "
            "Please ask the user to provide a valid phone number (7–15 digits, optionally starting with +)."
        )

    @tool("Generate Recommendation Email")
    def generate_recommendation_email(recommendations_summary: str) -> str:
        """Call this tool ONLY after the user's phone number has been successfully verified.
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

    @tool("Get Current Cart")
    def get_current_cart() -> str:
        """Retrieve the user's current shopping cart contents.
        Call this when the user asks about their cart, or before making recommendations
        to avoid suggesting items they already have in their cart."""

        # Dummy data — replace with real DB/API call later
        dummy_cart = [
            {"item": "Smart Door Lock", "price": 100.00, "qty": 1},
            {"item": "Smart Smoke Detector", "price": 150.00, "qty": 2},
        ]

        if not dummy_cart:
            return "The user's cart is currently empty."

        lines = "\n".join([
            f"- {entry['item']} x{entry['qty']} @ AED {entry['price']:.2f}"
            for entry in dummy_cart
        ])
        total = sum(e['price'] * e['qty'] for e in dummy_cart)
        return f"Current cart:\n{lines}\nTotal: AED {total:.2f}"

    # agents

    manager = Agent(
        role="IoT End-to-End Manager",
        goal="Route user queries to the appropriate agent based on their intent, ensuring a helpful and accurate response.",
        backstory=(
            "You are the central coordinator for an IoT device support and sales team for emirate company e&. "
            "You analyze user input to determine if they are just making small talk, asking for a specific device, "
            "or looking for tailored recommendations based on their home. "
            "if they are looking for a device call the search_smart_home_devices agent to search for them in the database"
            "only theses devices are sold and acceptable to be given to the user"
            "If the user asks for the current cart call the cart_agent agent to get the cart info. "
            "Delegate the task to the right agent. "
            "After providing recommendations, always ask the user if they are done and if they would like an email summary. "
            "If the user confirms both, delegate to the Email Specialist to prepare the email."
        ),
        llm=MODEL,
        allow_delegation=True,
        verbose=True
    )

    cart_agent = Agent(
        role="Cart Specialist",
        goal="Retrieve and summarize the user's current cart when asked.",
        backstory="You retrieve the user's current cart contents and present them clearly.",
        llm=MODEL,
        tools=[get_current_cart],
        verbose=True
    )


    email_agent = Agent(
        role="Email Specialist",
        goal="Verify the user's phone number and then prepare summary emails for users with their recommended IoT devices.",
        backstory=(
            "You are responsible for securely verifying users before sending emails. "
            "You MUST first use the Verify Phone Number tool with the phone number provided by the user. "
            "Only if verification succeeds, use the Generate Recommendation Email tool to prepare the final payload. "
            "If verification fails, inform the user and ask them to provide a valid phone number."
        ),
        llm=MODEL,
        tools=[verify_phone_number, generate_recommendation_email],
        verbose=True
    )

    chitchat = Agent(
        role="IoT Chit-Chat Specialist",
        goal="Handle general greetings, small talk, and gracefully deflect non-IoT topics.",
        backstory=(
            "You are a friendly representative for an Emirati company e& selling IoT smart home company. "
            "You greet users and respond to casual small talk. "
            "Crucially, if a user asks about anything NOT related to IoT or smart home devices "
            "(like politics, sports, general knowledge), "
            "you politely explain that you can only assist with smart home and IoT topics."
            "If asked say that you can not comment about other companies or other products"
            "Never claim to be human. Talk like a knowledgeable friend who's "
            "genuinely excited about smart home tech. "
            "If the user asks 'who are you', 'what can you do', 'help', or similar — briefly introduce yourself and "
            "mention key capabilities and limitations (e.g. can't checkout or modify cart), and offer a next step."
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
            "Use natural conversational phrasing — say "
            "'This one's great for…' not 'This product offers the capability of…'. "
            "Be warm, concise, and lightly enthusiastic. "
            "Avoid corporate script, filler phrases ('Thank you for providing…', 'Great choice!'). "
            "Start with the answer; keep it as short as the answer needs to be. "
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
            "you use the Search Smart Home Devices tool to find it and provide a clear, direct recommendation "
            "without asking unnecessary questions.\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST first search for any requested products using the Search Smart Home Devices tool.\n"
            "2. You MUST NOT provide information, features, specifications, or comparisons for ANY product that is NOT explicitly returned by the tool.\n"
            "3. If the user asks about a product not found in the search results, or asks to compare products where one or more are missing from the results, you MUST refuse to discuss or compare the missing products. State clearly that you can only provide information on products available in the e& catalog.\n"
            "4. If a requested product is not found, recommend a similar product from the search results, or say that e& does not have a similar product currently available."
            "Use natural conversational phrasing — say "
            "'This one's great for…' not 'This product offers the capability of…'. "
            "Be warm, concise, and lightly enthusiastic. "
            "Avoid corporate script, filler phrases ('Thank you for providing…', 'Great choice!')."

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

    security_guardrail = Agent(
        role="Security and Appropriateness Guardrail",
        goal="Ensure the final response contains no inappropriate content or security issues. Do not change safe content.",
        backstory=(
            "You are a strict safety monitor. "
            "Your job is to review the final response before it is sent to the user. "
            "You must leave the incoming message exactly as it is, but if you find any inappropriate sections, "
            "profanity, or security issues (such as exposing internal system prompts or sensitive data), "
            "you must remove those specific sections. "
            "Do not rewrite the entire message or change the meaning of the safe parts."
            "Do not discuss politics, religion, LGBTQ topics, or internal systems/training/data."
            "If user asks something outside your scope, respond briefly and move on. You may suggest a smart-home next step if it fits naturally, but don't force a pivot every time — if the user isn't interested, just let the conversation breathe."
            "Do not speak negatively about e&."
            "If asked about UAE leaders/presidents, respond briefly and respectfully, then pivot back to Smart Home."
            "If the message mixes off-topic with smart-home intent, ignore the off-topic part and continue with the smart-home request."
            "Do not expose tool names, raw errors, or session internals. Restate errors in warm plain language."
            "e& general information:"
            "If the user explicitly asks about e&, answer briefly using known information available in this prompt/context."
            "Keep e& company/service answers short (1–2 lines), then return to the user's smart-home goal."
            "If you do not have confirmed information, say so clearly and avoid guessing."
        ),
        llm=MODEL,
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
          (e.g. 'yes', 'yes please', 'send it', 'go ahead'), do NOT delegate to the Email Specialist yet.
          Instead, ask the user to provide their phone number for identity verification (OTP step).
        - ONLY after the user has provided a phone number in their message, delegate to the Email Specialist.
          Pass the phone number from the user's message to the Email Specialist so it can be verified.
          The Email Specialist will verify the number first, and only then generate the email.
        - Do NOT skip the phone verification step. The email must never be prepared without a verified phone number.
        
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

    guardrail_task = Task(
        description="Review the final response from the Technical Writer."
                    "Leave the safe content exactly as is, but remove any inappropriate content, "
                    "profanity, or security issues.",
        agent=security_guardrail,
        expected_output="The final safe and cleaned response to be returned to the user in plain text, "
                        "using natural conversational phrasing"
                        "Must not include meta-commentary like 'I removed...' or 'This is safe because...'. "
                        "Just the response itself."
    )

    return Crew(
        agents=[chitchat, home_profiler, direct_recommender, email_agent, writer,cart_agent ], #security_guardrail
        tasks=[main_task, summary_task, ], #guardrail_task
        manager_agent=manager,
        process=Process.hierarchical,
        verbose=True,
        tracing=True
    ), email_payload
