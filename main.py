from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine, save_note
from pdf import croatia_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI


load_dotenv()


#### querying the data in pandas using llama index query engine

population_path = os.path.join("data", "world_population.csv")
population_df = pd.read_csv(population_path)
#print(population_df.head())

population_query_engine = PandasQueryEngine(df=population_df, verbose = True, instruction_str=instruction_str)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})
#population_query_engine.query("what is the capital of canada?")



#### collection of tools for agent

tools = [
    FunctionTool.from_defaults(
        fn=save_note,
        name="note_saver",
        description="this tool can save a text or answer as a note to a file for the user"
        ),

    QueryEngineTool(
        query_engine=population_query_engine, 
        metadata=ToolMetadata(
        name="population_data",
        description="this tool gives information about the world demographic information"
        )
    ),
    QueryEngineTool(
        query_engine=croatia_engine, 
        metadata=ToolMetadata(
        name="croatian_data",
        description="this tool gives all public available information about the croatia from pdf"
        )
    )
]

#### LLM and Agent
llm = OpenAI(model="gpt-3.5-turbo")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)


while (prompt := input("Enter a prompt or write q to quit: ")) != "q":
    result = agent.query(prompt)
    print(result)