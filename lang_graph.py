from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline



class GraphState(TypedDict):
    prompt: str
    context: str
    retrieved_cases: list
    generated_case: str

# Loading data
test_cases_data = pd.read_pickle('test_cases_with_embeddings.pkl')
prompts_data = pd.read_csv('prompts.csv')
prompts = prompts_data['prompt'].tolist()

# Loading models
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0')  # استفاده از GPU
generator = pipeline('text-generation', model='gpt2', max_new_tokens=50)


# Defining nodes
def retrieve_node(state: Annotated[GraphState, add_messages]):
    logger.info(f"Retrieving for prompt: {state['prompt']}")
    prompt_embedding = retrieval_model.encode([state['prompt']])
    embeddings_1 = np.array(test_cases_data['Embedding_MiniLM_L6'].tolist(), dtype=np.float32)
    similarities = cosine_similarity(prompt_embedding, embeddings_1)
    top_k = 3
    top_indices = similarities[0].argsort()[-top_k:][::-1]
    related_context = "\n".join(test_cases_data.iloc[idx]['Full_Text'] for idx in top_indices)
    return {"context": related_context,
            "retrieved_cases": [test_cases_data.iloc[idx]['Full_Text'] for idx in top_indices]}


def generate_node(state: Annotated[GraphState, add_messages]):
    logger.info("Generating new test case")
    input_text = f"Based on this context: {state['context'][:200]}\nNew test case:\nTitle:"  # محدود کردن context
    generated = generator(input_text, max_new_tokens=50, truncation=True)[0]['generated_text']
    return {"generated_case": generated}


# Building graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.set_entry_point("retrieve")

# Compiling the workflow for every prompt
app = workflow.compile()

for prompt in prompts:
    initial_state = {"prompt": prompt}
    final_state = app.invoke(initial_state)

    print(f"\nPrompt: {prompt}")
    print("Related context:")
    print(final_state['context'])
    print("New test case generated:")
    print(final_state['generated_case'])