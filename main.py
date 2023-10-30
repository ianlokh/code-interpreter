import langchain

from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent

from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType

load_dotenv()


def _handle_error(error) -> str:
    return str(error)[:50]


def main():
    print(langchain.__version__)
    print("Start...")
    py_agent_exe = create_python_agent(llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
                                       tool=PythonREPLTool(),
                                       agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                       verbose=True,
                                       handle_parsing_errors=_handle_error)

    # py_agent_exe.run("Generate 15 QR codes that point to https://nlbsg.udemy.com/course/langchain/learn/lecture"
    #                  "/39035378#overview and save into the folder 'QR Codes' in the current directory. You have all "
    #                  "the Python libraries needed already installed")

    csv_agent = create_csv_agent(llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
                                 path="./episode_info.csv",
                                 agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                 verbose=True,
                                 handle_parsing_errors=_handle_error)

    # csv_agent.run("How many columns are there in the file episode_info.csv")
    # csv_agent.run("In the episode_info.csv file, which writer wrote the most episodes? How many episode did he write? "
    #               "You have all the Python libraries already installed")

    routing_agent = initialize_agent(tools=[
        Tool(
            name="PythonAgent",
            func=py_agent_exe.run,
            description="""Useful when you need to take instructions in natural language and generate from it python code,
            execute the python code and returning the results of the code execution. MUST NOT SEND PYTHON CODE TO THIS 
            TOOL!"""
        ),
        Tool(
            name="CSVAgent",
            func=csv_agent.run,
            description="""Useful when you need to answer questions over episode_info.csv file, takes an input the 
            entire question and returns the answer after running pandas calculations"""
        ),
    ],
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=_handle_error
    )

    # routing_agent.run("""Generate 15 QR codes that point to
    # https://nlbsg.udemy.com/course/langchain/learn/lecture/39035378#overview and save into the folder 'QR Codes' in
    # the current directory. All necessary Python Libraries have been installed.""")

    # routing_agent.run("""Generate 15 QR codes that point to
    # https://nlbsg.udemy.com/course/langchain/learn/lecture/39035378#overview and save into the folder 'QR Codes' in
    # the current directory and all the Python libraries that are needed have already been installed.""")

    # py_agent_exe.run("""Generate 15 QR codes that point to
    # https://nlbsg.udemy.com/course/langchain/learn/lecture/39035378#overview and save into the folder 'QR Codes' in
    # the current directory. All necessary Python Libraries have been installed.""")

    routing_agent.run("""In the episode_info.csv file, which writer wrote the most episodes? How many episode did he
    write? All necessary Python Libraries have been installed""")


if __name__ == "__main__":
    main()
