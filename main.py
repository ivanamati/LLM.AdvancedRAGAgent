from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
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



#### collection of tools

