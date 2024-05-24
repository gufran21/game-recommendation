from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from  pinecone import Pinecone
from pinecone.core.client.model.query_response import QueryResponse
import time
from dotenv import load_dotenv
import os
from langchain_community.llms import Ollama

load_dotenv() 
PINECONE_API_KEY =os.environ.get('PINECONE_API_KEY')
index = "gamerecommendationsystem"
llm = Ollama(model="llama2")
## Extract data from CSV file
def load_file(path):
    loader = CSVLoader(file_path=path, encoding="utf-8")
    data = loader.load()
    return data



# Transform data(create chunks)
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20) 
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


## download model from Hugging face
def download_hugging_face_embedding():
    model_name = "BAAI/bge-large-en"
    encode_kwargs = {'normalize_embeddings': True} 

    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )

    return embedding

embeddings = download_hugging_face_embedding()

pc=Pinecone(PINECONE_API_KEY)
index = pc.Index(index)
limit=3070
def retrieve(query,conversation_history):
    vector=embeddings.embed_query(query)
    # get relevant contexts
    contexts = []
    for message in conversation_history:
        contexts.append(f"{message['role'].capitalize()}: {message['content']}\n")
    l=len(contexts)
    res=index.query(vector=vector,top_k=1,include_values=True,include_metadata=True).to_dict()
    time.sleep(2)
    print("retrieve the answer from database having length = ",l)
    for x in res['matches']:
        contexts.append(x['metadata']['text'])

    print(f"Retrieved {len(contexts)} contexts, sleeping for 5 seconds...")
    if len(contexts)<=l:
        print("Timed out waiting for contexts to be retrieved.")
        contexts = ["No contexts retrieved. Try to answer the question yourself!"]


    # build our prompt with the retrieved contexts included
    prompt_start = (
        """Use the following pieces of information to answer the user's question.

        When a user submits a query, my model accesses the knowledge base containing information about game name, game title, game genre, game average rating, and more.\
        From this knowledge base, my model selects the top five responses that best match the user's query. These responses are presented to you as part of the prompt \
        along with the user's query. Your role then is to evaluate these responses with your expertise and enthusiasm for gaming, ensuring they adequately address\
        user's interests and preferences. If a justifiable answer is not present in the knowledge base responses, you can draw upon your own knowledge and intuition to provide additional recommendations or guidance. 

        STRICTLY FOLLOW 1: Response in bullet points with headings.
        STRICTLY FOLLOW 2: Response is not too long.
        STRICTLY FOLLOW 3: Each bullet point presented in a new line and short, concise form.
        STRICTLY FOLLOW 4: Give response in markdown format so we can add this to our existing HTML code.
        STRICTLY FOLLOW 5: If the user gives a greeting message, your reply should be in greeting format like: 'Hello there, how can I help you with game recommendations?'.

        If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"""+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\n Only return the helpful answer below and nothing else.\nAnswer:"
    )
    # append contexts until hitting limit
    prompt=""
    print(len(contexts))
    for i in range(0, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )

    return prompt


def complete(prompt):
    ## load ollama LAMA2 llm mODEL
    print("Now I'm inside of complete")
    return llm(prompt)


def chatbot(query,chat_history):
    query_with_contexts = retrieve(query,chat_history)
    print(query_with_contexts)
    response=complete(query_with_contexts)
    print("get the result from complter")
    return response,chat_history

