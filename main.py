from crew import create_crew_basic, create_iot_crew

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    crew = create_crew_basic("Egypt")
    print(crew.kickoff())
    crew = create_iot_crew("I want to reduce my electricity bill")
    print(crew.kickoff())