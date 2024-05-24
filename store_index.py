from src.helper import load_file, text_split, download_hugging_face_embedding
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file

from pinecone import Pinecone

import itertools
from pinecone import Pinecone
import os
load_dotenv() 
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
index_name = "gamerecommendationsystem"



extracted_data = load_file('data_set/finalData.csv')

text_chunks = text_split(extracted_data)


embeddings = download_hugging_face_embedding()

print("I'm here right now")
## Vector Embedding and vector store


vector=[embeddings.embed_query(t.page_content) for t in text_chunks]

vectors=[]
for i,vec in enumerate(vector):
    vectors.append({'id':str(i),'values':vec,"metadata": {'text':text_chunks[i].page_content}})


pc = Pinecone(api_key=PINECONE_API_KEY, pool_threads=30)
index = pc.Index("gamerecommendationsystem")


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))



with pc.Index('gamerecommendationsystem', pool_threads=30) as index:

    async_results = [
        index.upsert(vectors=ids_vectors_chunk, async_req=True)
        for ids_vectors_chunk in chunks(vectors, batch_size=100)
    ]
    [async_result.get() for async_result in async_results]


print("I'm here in last right now")



