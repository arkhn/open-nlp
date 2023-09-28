from dotenv import load_dotenv

load_dotenv()

import json  # noqa: E402

import chromadb  # noqa: E402
import datasets  # noqa: E402
from langchain.chains import RetrievalQA  # noqa: E402
from langchain.chat_models import AzureChatOpenAI  # noqa: E402
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings  # noqa: E402
from langchain.prompts.prompt import PromptTemplate  # noqa: E402
from langchain.vectorstores import Chroma  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    AnswerRelevancy,
    ContextRecall,
    ContextRelevancy,
    Faithfulness,
)

QA_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}
Question: {question}
Helpful Answer in French:
"""

qa_prompt = PromptTemplate(input_variables=["context", "question"], template=QA_TEMPLATE)


embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./qa_5.tronaze")
chroma_documents = Chroma(
    persist_directory="./qa_5.tronaze",
    embedding_function=embedding_function,
)

llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo", model_name="gpt-35-turbo", temperature=0, streaming=True
)

dataset_path = "../datasets/qa_5.json"
dataset = json.load(open(dataset_path, "r"))
ragas_dataset: dict = {"question": [], "answer": [], "ground_truths": [], "contexts": []}
for document in dataset:
    for qa in document["qa"]:
        # Run the chain
        retrieved_docs = chroma_documents.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "filter": {"doc_id": document["doc_id"]},
                "k": 3,
                "score_threshold": 0.7,
            },
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retrieved_docs,
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
        )
        qa_result = qa_chain({"query": qa["question"]})
        qa_human_result = qa_result["result"]

        ragas_dataset["question"].append(qa["question"])
        ragas_dataset["answer"].append(qa_human_result)
        ragas_dataset["ground_truths"].append([qa["answer"]])
        ragas_dataset["contexts"].append([document["context"]])


hf_ragas_dataset = datasets.Dataset.from_dict(ragas_dataset)
result = evaluate(
    hf_ragas_dataset,
    metrics=[
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm),
        ContextRelevancy(llm=llm),
        ContextRecall(llm=llm),
    ],
)

print(result)
