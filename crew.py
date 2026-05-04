from crewai import Agent, Task, Crew


MODEL = "ollama/llama3.2:1b"


def create_crew_with_context(topic: str, context: str):

    researcher = Agent(
        role="Smart Home Expert",
        goal="Use provided context to answer accurately",
        backstory="Expert in IoT devices and smart homes",
        llm=MODEL,
        verbose=True
    )

    writer = Agent(
        role="Technical Writer",
        goal="Summarize smart home information clearly",
        backstory="Writes concise product summaries",
        llm=MODEL,
        verbose=True
    )

    task1 = Task(
        description=f"""
        Use the following context to analyze smart home devices:

        CONTEXT:
        {context}

        USER QUESTION:
        {topic}

        Extract relevant insights and respond naturally while maintaining continuity with the conversation.
        """,
        agent=researcher,
        expected_output='Write a short paragraph on the options returned and how relevant it is to the user question.'
    )

    task2 = Task(
        description="Write a clear summary of the best relevant smart home devices.",
        agent=writer,
        expected_output=f'Write a sentence or two explaining the most relevant product for the user depending '
                        f'on the {topic}'

    )

    return Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        verbose=True
    )

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
        verbose=False
    )
