# GraphRAG Demo
This project implements a graph-based Retrieval-Augmented Generation (RAG) system using Neo4j, Graph Data Science, and Large Language Models.
It processes news articles, extracts entities and relationships, performs entity resolution, generates summaries for graph communities, and supports RAG queries.

## Structure

```
graph_rag/
│
├── run.py
├── config.py
├── data_processing/
│   ├── __init__.py
│   ├── news_processor.py
│   └── text_processor.py
├── graph_operations/
│   ├── __init__.py
│   ├── neo4j_graph.py
│   └── graph_data_science.py
├── llm/
│   ├── __init__.py
│   ├── gbx_model.py
│   └── entity_resolution.py
├── utils/
│   ├── __init__.py
│   ├── visualization.py
│   └── debug.py
└── requirements.txt
```

## Prerequisites

- Python 3.8+
- Neo4j Enterprise Edition (with APOC and Graph Data Science plugins installed)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/GreenBitAI/gbx-lm.git
   cd gbx-lm
   ```
   
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   
3. Download the pre-converted MLX embedding model:
   - The `bge-small-en.npz` and `config.json` file will be automatically downloaded from the Hugging Face repository: [Jaward/mlx-bge-small-en](https://huggingface.co/Jaward/mlx-bge-small-en).
   - `bge-small-en.npz` is a pre-converted MLX format embedding file, which is necessary for running the BERT embedding model in this project.

4. Set `BGE_SMALL_EN_PATH` either via environ variable or in `config.yaml` file:
   ```
   export BGE_SMALL_EN_PATH=/path/to/bge-small-en
   ```
   or add the path into the `project_root/gbx_lm/langchain/examples/common/config.yaml` file
   ```
   BGE_SMALL_EN_PATH: /path/to/bge-small-en
   ```

## Environment Setup

1. Download and install **Neo4j Desktop** from https://neo4j.com/deployment-center/

2. Create a new project in Neo4j Desktop and add a new database (or use an existing one).

3. Install the APOC and Graph Data Science plugins for your Neo4j database.

4. Update the `config.py` file with your Neo4j credentials and other settings:
   ```python
   NEO4J_URI = "bolt://localhost:7687"
   NEO4J_USERNAME = "neo4j"
   NEO4J_PASSWORD = "your-password"
   ```

5. Make sure you have the necessary NLTK data:
   ```
   python -c "import nltk; nltk.download('punkt_tab')"
   ```

## Running the System

The system now supports multiple modes of operation, controlled by command-line arguments:

1. Ensure your Neo4j database is running.

2. Run the main script with the desired mode:

   - To create a new graph database (note that the creation process may take several hours):
     ```
     python -m gbx_lm.langchain.examples.graph_rag.run --mode create --debug
     ```
   - To perform a RAG query:
     ```
     python -m gbx_lm.langchain.examples.graph_rag.run --mode rag --rag_mode community --rag_query "Your query here"
     ```
   - To query community summaries:
     ```
     python -m gbx_lm.langchain.examples.graph_rag.run --mode query --community_id <id>  # For a specific community
     python -m gbx_lm.langchain.examples.graph_rag.run --mode query --keyword <keyword>  # To search by keyword
     python -m gbx_lm.langchain.examples.graph_rag.run --mode query  # To get all summaries
     ```

   Add the `--debug` flag to any command for detailed logging and visualization. `--rag_mode`: ["community", "entity"]

## Main Components

- `run.py`: The entry point of the application, now supporting multiple modes of operation.
- `data_processing/`: Modules for loading and processing news data.
- `graph_operations/`: Modules for interacting with Neo4j and performing graph operations.
- `llm/`: Modules for working with Large Language Models, entity resolution, and RAG queries.
- `utils/`: Utility modules for visualization and debugging.

## Features

- **Multiple Operation Modes**: The system now supports creating the graph database, querying community summaries, and performing RAG queries.
- **Community Summaries**: Generate and retrieve summaries for graph communities.
- **GraphRAG Queries**: Perform Retrieval-Augmented Generation queries using the graph structure.

## Customization

You can customize various aspects of the system by modifying the `config.py` file, including:
- Number of articles to process
- Model parameters
- Graph projection settings

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are correctly installed.
2. Check that Neo4j is running and accessible with the provided credentials.
3. Verify that the APOC and Graph Data Science plugins are installed in Neo4j.
4. Check the console output for any error messages.
5. Use the `--debug` flag for more detailed information during execution.

## Acknowledgement

The code in this example project is inspired by the work from [tomasonjo](https://github.com/tomasonjo/blogs/blob/master/llm/ms_graphrag.ipynb). Special thanks to the author for sharing valuable insights and contributions, which greatly influenced the development of this project.