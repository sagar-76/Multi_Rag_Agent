# import os
# from langchain_groq import ChatGroq
#
# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain import hub
# from langgraph.prebuilt import create_react_agent
#
# class SQLAgentGroq:
#     def __init__(self, groq_api_key, db_uri, dialect="mysql", top_k=5, model="llama3-70b-8192"):
#         # Set Groq API Key
#         os.environ["GROQ_API_KEY"] = groq_api_key
#
#         # Load LLM
#         self.llm = ChatGroq(model_name=model, api_key=groq_api_key)
#
#         # Connect to SQL Database
#         self.db = SQLDatabase.from_uri(db_uri, view_support=True)
#
#         # Create Tools and Prompt
#         toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
#         self.tools = toolkit.get_tools()
#         prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
#         self.system_message = prompt_template.format(dialect=dialect, top_k=top_k)
#
#         # Create agent
#         self.agent = create_react_agent(self.llm, self.tools, prompt=self.system_message)
#
#     def ask(self, question):
#         try:
#             result = self.agent.invoke({"messages": [{"role": "user", "content": question}]})
#             return result["messages"][-1].content
#         except Exception as e:
#             return f"Error: {e}"
#
#
# if __name__ == "__main__":
#     # Define credentials
#     groq_api_key = "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"
#     db_uri = "mysql+pymysql://root:root@127.0.0.1:3306/campusx"
#
#     # Initialize Agent
#     agent = SQLAgentGroq(groq_api_key, db_uri)
#
#     # User input
#     while True:
#         question = input("\nAsk a question about your database (or 'exit'): ")
#         if question.lower() == "exit":
#             break
#         answer = agent.ask(question)
#         print("\nAnswer:\n", answer)

import os
from langchain_groq import ChatGroq

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain import hub
from langgraph.prebuilt import create_react_agent


class SQLAgentGroq:
    def __init__(self, groq_api_key, db_uri, dialect="mysql", top_k=5, model="llama3-70b-8192"):
        # Set Groq API Key
        os.environ["GROQ_API_KEY"] = groq_api_key

        # Load LLM
        self.llm = ChatGroq(model_name=model, api_key=groq_api_key)

        # Connect to SQL Database
        self.db = SQLDatabase.from_uri(db_uri, view_support=True)

        # Create Tools and Prompt
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()
        prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
        self.system_message = prompt_template.format(dialect=dialect, top_k=top_k)

        # Create agent
        self.agent = create_react_agent(self.llm, self.tools, prompt=self.system_message)

    def get_tables(self):
        """Returns real table names using toolkit tool."""
        try:
            list_tool = next(t for t in self.tools if t.name == "sql_db_list_tables")
            result = list_tool.run("")
            tables = [t.strip() for t in result.split(",")]
            return tables
        except Exception as e:
            return f"Error fetching tables: {e}"

    def ask(self, table: str, question: str):
        """Asks a question in the context of a given table."""
        try:
            prompt = f"In the table `{table}`, {question}"
            result = self.agent.invoke({"messages": [{"role": "user", "content": prompt}]})
            return result["messages"][-1].content
        except Exception as e:
            return f"Error: {e}"


# Example Usage
if __name__ == "__main__":
    groq_api_key = "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"
    db_uri = "mysql+pymysql://root:root@127.0.0.1:3306/campusx"

    # Initialize Agent
    agent = SQLAgentGroq(groq_api_key, db_uri)

    # Step 1: Show tables
    tables = agent.get_tables()
    if isinstance(tables, str):
        print(tables)
        exit()

    print("\nAvailable Tables:")
    for t in tables:
        print(f" - {t}")

    # Step 2: Get user input
    while True:
        selected_table = input("\nEnter the table name you want to query (or 'exit'): ")
        if selected_table.lower() == "exit":
            break
        if selected_table not in tables:
            print("Table not found. Please select from the available tables.")
            continue

        question = input("Now enter your question about this table: ")
        answer = agent.ask(selected_table, question)
        print("\nAnswer:\n", answer)
