
import torch
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import re
from nltk.corpus import stopwords
import faiss
import json

# Typical cleaning of text. This is not a comprehensive list of cleaning steps.
def clean_text(text):
    """Clean the input text."""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Strip extra whitespaces
    text = text.strip()
    return text

def encode_articles(articles, model,tokenizer,device,batch_size=32):
    """Encode list of articles into vectors."""
    model.eval()
    all_vectors = []

    # Integrate tqdm progress bar
    for i in tqdm(range(0, len(articles), batch_size), desc="Encoding articles"):
        batch = [clean_text(article) for article in articles[i:i+batch_size]]
        with torch.no_grad():
            # Process batch and ensure it is on the same device as the model
            encoded_batch = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            outputs = model(**encoded_batch)
            # Get the mean of the last hidden states as the sentence embeddings
            vectors = outputs.last_hidden_state.mean(dim=1)
            all_vectors.append(vectors.cpu().numpy())
    
    return np.concatenate(all_vectors, axis=0)


# def encode_articles(articles,model,tokenizer,device,batch_size=32):
    
#     """Encode list of articles into vectors."""
#     model.eval()
#     all_vectors = []
    
#     # Integrate tqdm progress bar
#     for i in tqdm(range(0, len(articles), batch_size), desc="Encoding"):
#         batch = [clean_text(article) for article in articles[i:i+batch_size]]
#         with torch.no_grad():
#             encoded_batch = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
#             outputs = model(**encoded_batch)
#             vectors = outputs.pooler_output
#             all_vectors.append(vectors.cpu().numpy())
    
#     return np.concatenate(all_vectors, axis=0)

def build_faiss_index(vectors):
    """Build and train a FAISS index."""
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index

def process_query_with_chatgpt(query, openai_api_key):
    """Use ChatGPT to expand the user query."""
    client = OpenAI(api_key=openai_api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",  # Verify this is the correct model
            messages=[{"role": "system", "content": '''You are a helpful assistant. You are given a query. 
                       Find the important entities in the query and write a 5 sentece summary definition of those entities. 
                       Afterward explain the query relation to the entities.
                       The output should be in a json format with the following keys: "summary", "entities", "relation'''},
                      {"role": "user", "content": query}],
            max_tokens=300,
            temperature=0.7,
        )
        return response
    except Exception as e:
        print(f"An error occurred: {type(e).__name__}, {str(e)}")
        return None
    
def search_with_faiss(query, index, articles, model,tokenizer,device,top_k=5):
    """Search for similar articles using FAISS and return the original articles."""
    # Encode the query to get its vector representation
    query_vector = encode_articles([query], model,tokenizer,device)[0]
    # Search in FAISS index
    distances, indices = index.search(np.array([query_vector]), top_k)
    # Retrieve the original articles based on the indices
    return [articles[i] for i in indices[0]]

def process_Faiss_output_with_chatgpt(query,list_pages, openai_api_key):
    """Use ChatGPT to find the answer to the user query. In the list of articles
      and return the answer and the index of the article in the list."""
    client = OpenAI(api_key=openai_api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",  # Verify this is the correct model
            messages=[{"role": "system", "content": f'''You are a helpful assistant. You are given a query. 
                       Please try your best to find the answer to the query in the following list:{list_pages}.
                        Also return the index in the list that you found the answer in. The output should be in json format with the following keys: "answer", "index"'''},
                      {"role": "user", "content": query}],
            max_tokens=300,
            temperature=0.7,
        )
        return response
    except Exception as e:
        # Enhanced error logging
        print(f"An error occurred: {type(e).__name__}, {str(e)}")
        return None

def gpt_parsing(processed_query):
    """Parse the output of the LLM to get the entities, relation and summary."""
    json_string = processed_query.choices[0].message.content

    # Parse the JSON string into a Python dictionary
    llm_output = json.loads(json_string)

    # Accessing the different parts of the JSON
    summary = llm_output['summary']
    entities = llm_output['entities']
    relation = llm_output['relation']
    faiss_query = relation + " " + summary
    
    return faiss_query, entities, relation, summary

def keyword_articles(entities,wiki_df,n):
    """Find the articles that contain the entities in the query."""
    regex_pattern = '|'.join(entities)
    articles_key_word = wiki_df[wiki_df['text'].str.contains(regex_pattern,case=False,na=False)][:n]['text'].tolist()
    return articles_key_word