from operator import index
import streamlit as st
import torch
import os
import pinecone
from langchain_community.retrievers import PineconeHybridSearchRetriever
## from Pinecone import ServerlessSpec
import time
from pinecone import Pinecone,ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
import psycopg2
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from psycopg2.extras import RealDictCursor
#from langchain.llms import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceEndpoint
from operator import index
from dotenv import load_dotenv
load_dotenv()

## user_query = "When is James joined?"
# Step 1: Initialize Pinecone
PC_API_KEY=PC_API_KEY
#Pinecone.init(api_key=PC_API_KEY, environment="us-east-1")
pc=Pinecone(api_key=PC_API_KEY)
# Create a Pinecone Index
index_name = "hybrid-search-langchain-pinecone"
#create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Step 2: Load Hugging Face Model for Text Embedding
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Helper Function to Convert Text to Embeddings
def get_embeddings(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().tolist()

# Step 3: Data Ingestion from PostgreSQL
def ingest_data_from_postgres():
    connection = psycopg2.connect(
        dbname="postgres", user="postgres", password=PG_password, host="mypgdb-4.c38oaogsexra.us-east-1.rds.amazonaws.com", port="5432"
    )
    cursor = connection.cursor()

    # Query your data
    cursor.execute('SELECT id, "Description" FROM employee')
    rows = cursor.fetchall()

    for row in rows:
        record_id, text = row
        ## print(text)
        embeddings = get_embeddings(text)
        # Ensure embeddings are in the correct format
        embeddings = [float(val) for val in embeddings]  # Convert to float if necessary

        # Store in Pinecone
        index = pc.Index(index_name)  # Properly initialize the index
        ## index = pc.Index(host="https://hybrid-search-langchain-pinecone-bei4j4u.svc.aped-4627-b74a.pinecone.io")
       ## index.upsert([(str(record_id),embeddings)])
        index.upsert([
    {
        "id": str(record_id),
        "values": embeddings,
        "metadata": {"Description": text}  # Add metadata here
    }
])

    cursor.close()
    connection.close()

# Ingest Data
ingest_data_from_postgres()

# Step 4: Query Processing
def query_pinecone(user_query):
    query_embeddings = get_embeddings(user_query)
    index_name = "hybrid-search-langchain-pinecone"
    index = pc.Index(index_name)  # Properly initialize the index
    # Search Pinecone Index
    search_results = index.query(
        vector=query_embeddings,
        top_k=3,
        include_metadata=True
    )
    ## print(f"Search Results: {search_results}")
    # Filter and extract context from metadata
    context = "\n".join(
        match["metadata"].get("Description", "No Description Found")
        for match in search_results.get("matches", [])
        if "metadata" in match and "Description" in match["metadata"]
    )
    ## print(f"Context: {context}")
    return context
## Title of the Application
st.title("Clinsys RAG Based Gen AI Implementation POC")
st.write("Response retrived from PC and complete from LLM")
##user_query = "When is Bond joined?"
user_query=st.text_input("Enter the User Query:")
context=query_pinecone(user_query)

## from langchain.llms import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceEndpoint
## !pip install --upgrade --quiet langchain_huggingface     
## !pip install -U langchain-community
from langchain_huggingface import HuggingFaceEndpoint
## from langchain_huggingface import HuggingFaceEndpoint

repo_id="mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(
    endpoint_url=f"https://api-inference.huggingface.co/models/{repo_id}",
    huggingfacehub_api_token=HF_TOKEN,
    ##max_length=128,
    temperature=0.1,
)

# Example usage of the model
prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
response = llm.invoke(prompt)
first_answer = response.split("Question:")[0].strip()
st.write(first_answer)
#print(f"Final_LLM_Response: {first_answer}")
##print(f"Final_LLM_Response: {response}")
