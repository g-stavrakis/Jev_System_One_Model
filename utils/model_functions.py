from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import time
from typesafe_sdk import TypeSafeClient

# Defining helper function for chat models with structured output
class QuestionSchema(BaseModel):
    """Sample Schema to be overwritten."""

MODEL_COSTS = {
    "gpt-6.1-sol": {"input": 2 , "output": 10},
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50}
}

def call_chat_model(prompt: str, categories: list[str], model_name: str, schema_class: type[BaseModel]) -> dict:
    """Function to invoke and parse GPT chat models"""

    model = ChatOpenAI(model=model_name)
    model_with_structure = model.with_structured_output(schema_class, include_raw=True)

    start_time = time.perf_counter()
    response = model_with_structure.invoke(prompt)
    latency = time.perf_counter() - start_time

    res = response['parsed'].model_dump()
    results= {}
    results['latency'] = latency

    for category in categories:
        results[category] = res[category]

    input_cost = (response['raw'].usage_metadata['input_tokens'] / 1000000) * MODEL_COSTS[model_name]['input']
    output_cost = (response['raw'].usage_metadata['output_tokens'] / 1000000) * MODEL_COSTS[model_name]['output']

    results['cost'] = input_cost + output_cost

    return results


# Defining helper function for jev model

# Initialize the client
client = TypeSafeClient(model="jev-latest")

def call_jev(state: dict, questions: dict, categories: list) -> dict:
    """Function to invoke and parse Jev model"""

    start_time = time.perf_counter()
    response = client.system_one(state=state, questions=questions)
    latency = time.perf_counter() - start_time

    results= {}
    results['latency'] = latency

    for category in categories:
        if hasattr(response.answers[category], 'choice'):
            results[category] =response.answers[category].choice
        elif hasattr(response.answers[category], 'score'):
           results[category] =response.answers[category].score 
        else:
            results[category] =response.answers[category].noul
        
    results['cost'] = (response.usage.input_tokens / 1000000) * 0.042
    return results
