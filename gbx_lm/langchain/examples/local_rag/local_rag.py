# Import necessary libraries
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from gbx_lm.langchain.chat_gbx import ChatGBX
from gbx_lm.langchain import GBXPipeline

import re

from .emb_model import get_bert_mlx_embeddings

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Helper function to print task separators
def print_task_separator(task_name):
    print("\n" + "="*50)
    print(f"Task: {task_name}")
    print("="*50 + "\n")

def clean_output(text):
    # Remove all non-alphanumeric characters except periods and spaces
    cleaned = re.sub(r'[^a-zA-Z0-9\.\s]', ' ', text)
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Remove any mentions of "assistant" or other unwanted words
    cleaned = re.sub(r'\b(assistant|correct|I apologize|mistake)\b', '', cleaned, flags=re.IGNORECASE)
    # Remove any remaining leading/trailing whitespace
    cleaned = cleaned.strip()
    # Ensure the first letter is capitalized
    cleaned = cleaned.capitalize()
    # Ensure the answer ends with a period
    if cleaned and not cleaned.endswith('.'):
        cleaned += '.'
    return cleaned

def extract_answer(text):
    # Try to extract a single sentence answer
    match = re.search(r'([A-Z][^\.!?]*[\.!?])', text)
    if match:
        return match.group(1)
    # If no clear sentence is found, return the first 100 characters
    return text[:100] + '...' if len(text) > 100 else text

# Load and prepare data
def prepare_data():
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    return all_splits

# Create vector store
def create_vectorstore(documents):
    bert_mlx_embeddings = get_bert_mlx_embeddings()
    return Chroma.from_documents(documents=documents, embedding=bert_mlx_embeddings)

# Initialize GBX model
def init_gbx_model():
    llm = GBXPipeline.from_model_id(
        model_id="GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0-mlx",
        pipeline_kwargs={"max_tokens": 100, "temp": 0.6}
    )
    return ChatGBX(llm=llm)

# Task 1: Rap Battle Simulation
def simulate_rap_battle(model):
    print_task_separator("Rap Battle Simulation")
    response = model.invoke("Simulate a rap battle between RAG and GraphRAG")
    print(response.content)

# Task 2: Summarization
def summarize_docs(model, vectorstore, question):
    print_task_separator("Summarization")
    prompt = ChatPromptTemplate.from_template(
        "Summarize the main themes in these retrieved docs in a single, complete sentence of no more than 50 words: {docs}"
    )
    chain = (
        {"docs": format_docs}
        | prompt
        | model
        | StrOutputParser()
        | clean_output
        | extract_answer
    )
    docs = vectorstore.similarity_search(question)
    response = chain.invoke(docs)
    print(response)

# Task 3: Q&A
def question_answering(model, vectorstore, question):
    print_task_separator("Q&A")
    RAG_TEMPLATE = """
    Answer the following question based on the context provided. Give a direct and concise answer in a single, complete sentence of no more than 30 words. Do not include any additional dialogue or explanation.

    Context:
    {context}

    Question: {question}

    Answer:"""

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    chain = (
        RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
        | rag_prompt
        | model
        | StrOutputParser()
        | clean_output
        | extract_answer
    )
    docs = vectorstore.similarity_search(question)
    response = chain.invoke({"context": docs, "question": question})
    print(response)

# Task 4: Q&A with Retrieval
def qa_with_retrieval(model, vectorstore, question):
    print_task_separator("Q&A with Retrieval")
    RAG_TEMPLATE = """
    Answer the following question based on the retrieved information. Provide a direct and concise answer in a single, complete sentence of no more than 30 words. Do not include any additional dialogue or explanation.

    Question: {question}

    Answer:"""

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    retriever = vectorstore.as_retriever()
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | model
        | StrOutputParser()
        | clean_output
        | extract_answer
    )
    response = qa_chain.invoke(question)
    print(response)

# Main execution
def main():
    print_task_separator("Initialization")
    print("Preparing data and initializing model...")
    # Prepare data and initialize model
    all_splits = prepare_data()
    vectorstore = create_vectorstore(all_splits)
    model = init_gbx_model()
    print("Initialization complete.")

    # Execute tasks
    simulate_rap_battle(model)

    question = "What are the approaches to Task Decomposition?"
    summarize_docs(model, vectorstore, question)
    question_answering(model, vectorstore, question)
    qa_with_retrieval(model, vectorstore, question)

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()