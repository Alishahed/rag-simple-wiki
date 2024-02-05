# Download NLTK stopwords
from transformers import AutoTokenizer, AutoModel
import utils as ut
import numpy as np
import pandas as pd
import nltk
import faiss
import json
import torch
import sys
from dotenv import load_dotenv
import os

def main():
    load_dotenv()

    api_key = os.getenv('OPEN_AI_API_KEY')

    if api_key is None:
        print("API key not found. Please set the MY_API_KEY in the .env file.")
        exit(1)
    
    if len(sys.argv) != 2:
        sys.exit(1)

    query = sys.argv[1]

    # Download NLTK stopwords
    nltk.download('stopwords')

    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Sentence-BERT model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    # Load the model
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device)

    # Load the FAISS index
    faiss_index = faiss.read_index('faiss_index_mpnet.idx')
    # Load the articles
    wiki_simple = pd.read_csv('wiki_simple_text.csv')
    # Convert the articles to a list
    articles = wiki_simple['text'].tolist()
    # Expanding the user query using an LLM (Here we use ChatGPT-3.5 but an open source LLM can used here)
    processed_query = ut.process_query_with_chatgpt(query, api_key)
    # Here we parse the output of the LLM to get the entities, relation and summary
    faiss_query, entities, relation, summary = ut.gpt_parsing(processed_query)
    # Here we use the faiss_query, i.e. summery + relation, to find the most similar articles using FAISS.
    similar_articles = ut.search_with_faiss(ut.clean_text(faiss_query), faiss_index, articles, model,tokenizer,device,top_k=3)
    # To make our solution more robust, we use the entities we had in query to find more relevant articles using keyword search, i.e. we find the articles that contain the entities in the query.
    articles_key_word = ut.keyword_articles(entities,wiki_simple,2)

    # Here we combine the results of the two methods to get the final knowledge base.
    knowledge_base = similar_articles + articles_key_word
    # Here we use the final knowledge base to find the answer to the user query using ChatGPT-3.5.
    processed_Faiss_output = ut.process_Faiss_output_with_chatgpt(query,knowledge_base, api_key)
    # Here we parse the output of the LLM to get the answer and the index of the article in the knowledge base.
    json_answer = processed_Faiss_output.choices[0].message.content
    answer = json.loads(json_answer)['answer']
    index = json.loads(json_answer)['index']
    print(f'Answer: {answer}')
    print(f'Article: {knowledge_base[index]}')

if __name__ == "__main__":
    # Run the app
    main()