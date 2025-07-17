# model.py - UPDATED VERSION
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
import os
import pandas as pd
import pandasql as ps
import numpy as np
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# Remove this deprecated import: from langchain import LLMChain

# Load environment variables from the .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

data_dir = "/Users/jaishankarbhagat/Documents/AI Hub/Agentic-chatbot/Chatbot/Data"

customers = pd.read_csv(os.path.join(data_dir, "customers.csv"), encoding='ISO-8859-1')
products = pd.read_csv(os.path.join(data_dir, "products.csv"))
productsubcategory = pd.read_csv(os.path.join(data_dir, "productsubcategories.csv"))
productcategory = pd.read_csv(os.path.join(data_dir, "productcategories.csv"))
vendor = pd.read_csv(os.path.join(data_dir, "vendors.csv"))
productvendor = pd.read_csv(os.path.join(data_dir, "vendorproduct.csv"))
employee = pd.read_csv(os.path.join(data_dir, "employees.csv"), encoding='ISO-8859-1')
sales = pd.read_csv(os.path.join(data_dir, "sales.csv"))

# PromptTemplate
def generate_prompt_inference(question, prompt_file="prompt_adv.md", query_example="query_example.txt", metadata_file="metadata_adv.txt"):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    
    with open(metadata_file, "r") as f:
        table_metadata_string = f.read()
            
    with open(query_example, "r") as f:
        query_example_string = f.read()         

    prompt = prompt.format(
        user_question=question, 
        table_metadata_string=table_metadata_string, 
        query_example_string=query_example_string
    )
    return prompt

# Model initialization
sql_llm_model = ChatOpenAI(
    temperature=0, 
    model="gpt-4o-mini", 
    openai_api_key=openai_api_key, 
    streaming=True
)

# UPDATED: Modern LangChain syntax (no more LLMChain)
def run_inference(question, prompt_file="prompt_adv.md", query_example="query_example.txt", metadata_file="metadata_adv.txt"):
    try:
        # Create the prompt template
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template=generate_prompt_inference(question)
        )
        
        # Modern LangChain syntax: use | operator
        chain = prompt_template | sql_llm_model
        
        # Invoke the chain with proper input format
        response = chain.invoke({"question": question})
        
        # Extract content from response
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
            
    except Exception as e:
        print(f"Error in run_inference: {e}")
        return "SELECT 'Error generating SQL' as message"

summary_prompt_template = """
You have been provided: 
a. The metadata of database {metadata}, 
b. User question: {question}
c. An sql code to generate a table {sql_code} and 
d. the table output
{table}

You have been asked to convert this table output to simple human language?
"""    

# UPDATED: Modern LangChain syntax for summary
def run_summary_inference(output, sql_code, user_question, metadata_file="metadata_adv.txt"):
    try:
        prompt_template = PromptTemplate(
            input_variables=["metadata", "question", "sql_code", "table"],
            template=summary_prompt_template
        )
        
        # Modern LangChain syntax
        chain = prompt_template | sql_llm_model
        
        response = chain.invoke({
            "metadata": metadata_file,
            "question": user_question,
            "sql_code": sql_code,
            "table": output.to_string() if not output.empty else "No data found"
        })
        
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
            
    except Exception as e:
        print(f"Error in run_summary_inference: {e}")
        return f"Analysis complete. Found {len(output)} records in the results."

def sql_query_execution(sql_query):
    try:
        output = ps.sqldf(sql_query)
    except Exception as e:
        print(f"SQL execution error: {e}")
        output = pd.DataFrame()    
    return output    

def func_final_result(query):
    llm_output = run_inference(query)
    if llm_output.find('```sql\n') >= 0:
        query_start_position = llm_output.find('```sql\n') + 7
        if llm_output.find('\n```') > 0:
            query_end_position = llm_output.find('\n```')
            sql_query = llm_output[query_start_position:query_end_position]
        else:
            sql_query = llm_output[query_start_position:]
        
        output = sql_query_execution(sql_query)
    else:
        sql_query = llm_output
        output = pd.DataFrame()
        
    summarized_output = ''
    if output.shape[0] > 0:
        summarized_output = run_summary_inference(output, sql_query, query)
    
    final_output = [summarized_output, sql_query]
    return final_output