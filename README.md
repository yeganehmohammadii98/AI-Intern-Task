1. Data Preparation and Embedding
To start, we keep our data clean and organized. We store prompts in prompts.csv and test cases in test_cases.csv.

Next, we convert our test cases into embeddings format. We use pre-trained Sentence Transformer models for this: all-MiniLM-L6-v2 and paraphrase-MiniLM-L3-v2. We chose two different models, even for a small dataset of 10 test cases, to compare their performance. Sometimes lighter models can work just as well

2. Similarity Search and Retrieval
After embedding, we find test cases that are most similar to a user's query with cosine similarity.

a. Embed the Prompt: We turn the user's query into an embedding, just like we did with the test cases.

b. Calculate Cosine Similarity: We use cosine similarity to measure how "close" the prompt's embedding is to each test case's embedding. A higher score means they're more similar.

c. Retrieve Relevant Test Cases: Based on these similarity scores, we pull the most relevant test cases.

3. Generating New Test Cases
Finally, we generate brand-new test cases using a large language model.

GPT-2 for Generation: We use the gpt2 model to create text.

Contextual Generation: To ensure the new test cases are relevant and well-formatted, we provide GPT-2 with:

