from dotenv import load_dotenv
load_dotenv() 

import os
deployment_id = os.environ.get("OPENAI_DEPLOYMENT_NAME")
model_name = os.environ.get("OPENAI_MODEL_NAME")
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent

# First Question

from langchain.llms import AzureOpenAI
llm = AzureOpenAI(
    temperature=0,
    deployment_name="mayemsftgpt001",
    model_name="text-davinci-003",
)

print(llm)

QUESTION = "Tell me five most popular fruits."
print(QUESTION)
fruits_names = llm(QUESTION)
print(fruits_names)
print(type(fruits_names))

# OutputParser

from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List five most popular {subject}.\n{format_instructions}",
    input_variables=['subject'],
    partial_variables={"format_instructions": format_instructions}
)


from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
output = chain.run(subject='fruits')
fruits_names = output_parser.parse(output)
print(fruits_names)
print(type(fruits_names))

# Next two questions

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Prompt Template
agriculture_template = """
You are a very smart agriculture professor. \
You are great at answering questions about agriculture. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""


nutrition_template = """You are a very good nutritionist. You are great at answering nutrition questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{input}"""
prompt_infos = [

    {
        "name": "agriculture",
        "description": "Good for answering questions about agriculture",
        "prompt_template": agriculture_template
    },
    {
        "name": "nutrition",
        "description": "Good for answering questions about nutrition",
        "prompt_template": nutrition_template
    },
]

# Prompt Router
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

# Router Chain
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt, output_key="text")
    destination_chains[name] = chain
default_chain = ConversationChain(llm=llm, output_key="text")
router_chain = LLMRouterChain.from_llm(llm, router_prompt)
multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=False
)

result = multi_prompt_chain.run(input="what is the top fruit apple producing country?")
print(result)
result = multi_prompt_chain.run(input="what is the calories value of apple per 100 grams?")
print(result)

# Extract value from result

extract_template = """
You are a very smart assistant. \
You need to extract information from a text base on instructions. \
Here is the instructions: \
{extract_instructions} \
\
Here is  the text: \
{text} \
\
{format_instructions}"""

extract_prompt = PromptTemplate(
    template=extract_template,
    input_variables=['extract_instructions', 'text'],
    partial_variables={"format_instructions": format_instructions}
)
extract_chain = LLMChain(llm=llm, prompt=extract_prompt, output_key="extracted_value")
output = extract_chain.run({"extract_instructions":"Get the calories number only.", 
                   "text":"The calorie value of an apple per 100 grams is 52 calories."})

print(output)

# SequentialChain

from langchain.chains import SequentialChain
overall_chain = SequentialChain(
    chains=[multi_prompt_chain, extract_chain],
    input_variables=["input", "extract_instructions"],
    output_variables=["extracted_value"])
output = overall_chain.run(input="What is the top fruit apple producing country?", 
                           extract_instructions="Find the country name in the text below, get the country name only.")
print(output)

output = overall_chain.run(input="what is the calories value of apple per 100 grams?", 
                           extract_instructions="Find the calories number in the text below, get the number only.")
print(output)


# Agent

from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain.agents import AgentType

from langchain.agents import initialize_agent
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="query with search engine",
    ),
]

agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION)
result = agent_chain.run(input="give me a short description of apple as fruit")
print(result)

# Put All Together

import pandas as pd 
fruits_countries = []
fruits_kalories = []
fruits_desc = []
for name in fruits_names:
    output = overall_chain.run(input=f"What is the top fruit {name} producing country?", 
                               extract_instructions="get the country name only.")
    fruits_countries.append(output_parser.parse(output)[0])
    output = overall_chain.run(input=f"what is the calories value of {name} per 100 grams", 
                               extract_instructions="get the calories number only.")
    fruits_kalories.append(output_parser.parse(output)[0])
    output = agent_chain.run(input=f"give me a short description of {name} as fruit")
    fruits_desc.append(output)
df = pd.DataFrame({"country": fruits_countries, "calories": fruits_kalories, "desc": fruits_desc}, 
                  index = fruits_names)

print(df)

# Query teh data with Agent

from langchain.agents import create_pandas_dataframe_agent
csv_agent = create_pandas_dataframe_agent(llm, df, verbose=False)

result = csv_agent.run("how many rows are there?")

print(result)

result = csv_agent.run("which fruit with the lowest calories and what country produce it most?")

print(result)
