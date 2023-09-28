import json
import os

import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

emb_fn = embedding_functions.DefaultEmbeddingFunction()

path = "qa_5.tronaze"
if path in os.listdir("."):
    os.rmdir(path)

client = chromadb.PersistentClient(path=path)
collection = client.create_collection(name="qa_5", embedding_function=emb_fn)
dataset_path = "../datasets/qa_5.json"
dataset = json.load(open(dataset_path, "r"))

chunk_size = 512
custom_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=20,
    length_function=len,
)

documents = []
metadatas = []
ids = []
id_ = 0

for d in dataset:
    chunks = custom_text_splitter.create_documents([d["context"]])
    for i, chunk in enumerate(chunks):
        documents.append(chunk.page_content)
        metadatas.append(
            {
                "doc_id": d["doc_id"],
                "patient_id": d["patient_id"],
                "chunk": i,
            }
        )
        ids.append(str(id_))
        id_ += 1

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
)
