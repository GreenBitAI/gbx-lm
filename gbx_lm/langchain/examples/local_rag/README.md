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
   - Go to the Hugging Face repository: [Jaward/mlx-bge-small-en](https://huggingface.co/Jaward/mlx-bge-small-en)
   - Download the "bge-small-en.npz" file
   - Place the downloaded file in the `common/mlx-bge-small-en` directory of your project

   This file is a pre-converted MLX format embedding file, which is necessary for running the BERT embedding model in this project.

## File Structure

- `run.py`: Main script containing the RAG implementation and task execution
- `common/emb_model.py`: Custom BERT embedding model implementation
- `requirements.txt`: List of required Python packages
- `common/mlx-bge-small-en/`: Directory containing the MLX embedding model file
- `common/bge-small-en.npz`: Pre-converted MLX format embedding file (to be downloaded)

## Usage

Run the main script to execute all tasks:

```
python -m gbx_lm.langchain.examples.local_rag.run
```

This will perform the following tasks:
1. Initialize the model and prepare data
2. Simulate a rap battle
3. Summarize documents based on a question
4. Perform question answering
5. Perform question answering with retrieval