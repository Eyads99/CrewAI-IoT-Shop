from crewai.flow.flow import Flow, start
from crew import create_crew_basic

class ResearchFlow(Flow):

    def __init__(self, topic: str):
        super().__init__()
        self.topic = topic

    @start()
    def run(self):
        print("Starting crew")
        crew = create_crew_basic(self.topic)
        result = crew.kickoff()
        return result
