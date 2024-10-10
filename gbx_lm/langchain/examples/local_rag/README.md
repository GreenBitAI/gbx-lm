# Local RAG Demo

## Overview

This project demonstrates a local implementation of Retrieval-Augmented Generation (RAG) using the GBX language model and BERT embeddings. It includes features such as document loading, text splitting, vector store creation, and various natural language processing tasks.

## Features

- Document loading from web sources
- Text splitting for efficient processing
- Vector store creation using BERT embeddings
- Rap battle simulation
- Document summarization
- Question answering
- Question answering with retrieval

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Download the pre-converted MLX embedding model:
   - The `bge-small-en.npz` and `config.json` file will be automatically downloaded from the Hugging Face repository: [Jaward/mlx-bge-small-en](https://huggingface.co/Jaward/mlx-bge-small-en).
   - `bge-small-en.npz` is a pre-converted MLX format embedding file, which is necessary for running the BERT embedding model in this project.

3. Set `BGE_SMALL_EN_PATH` either via environ variable or in `config.yaml` file:
   ```
   export BGE_SMALL_EN_PATH=/path/to/bge-small-en
   ```
   or add the path into the `project_root/gbx_lm/langchain/examples/common/config.yaml` file
   ```
   BGE_SMALL_EN_PATH: /path/to/bge-small-en
   ```

## File Structure

- `run.py`: Main script containing the RAG implementation and task execution
- `common/emb_model.py`: Custom BERT embedding model implementation
- `requirements.txt`: List of required Python packages
- `common/mlx-bge-small-en/`: Directory containing the MLX embedding model file
- `common/bge-small-en.npz`: Pre-converted MLX format embedding file (to be downloaded)

## Usage

Run the main script to execute all tasks:

```
python -m gbx_lm.langchain.examples.local_rag.run --model "GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0-mlx"  \
                                                   --query "What are the core method components of GraphRAG?" \
                                                   --max_tokens 300 \
                                                   --web_source "https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/"
```

This will perform the following tasks:
1. Initialize the model and prepare data
2. Simulate a rap battle
3. Summarize documents based on a question
4. Perform question answering
5. Perform question answering with retrieval