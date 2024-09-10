# Chat Demo

## Overview

This project demonstrates a quick setup for a chat interface using [GreenBitAI's MLX model](https://huggingface.co/collections/GreenBitAI/greenbitai-mlx-llm-6614eb6ceb8da657c2b4ed58), [FastChat](https://github.com/lm-sys/FastChat) tool, and Gradio web UI. It enables conversations using a local model creating an efficient and accessible environment for deploying chat applications by leveraging the capabilities of FastChat and Gradio.


<img src="../../assets/web_chat_demo_mlx.gif" width="960">


## Installation

Installation involves two main steps: setting up the gbx_lm package and installing FastChat along with its dependencies.

### Step 1: Install the gbx_lm Package

```bash
pip install gbx-lm
```

### Step 2: Install FastChat

For the FastChat installation, refer to the [FastChat GitHub page](https://github.com/lm-sys/FastChat) for the complete installation guide. The simplest method to install FastChat and its necessary dependencies is by executing the following command in your terminal:

```bash
pip install "fschat[model_worker,webui]"
```

Ensure your system has Python3 and pip installed before proceeding.

## Starting the Local Service

To activate the chat interface with your local MLX model, follow these steps:

### Start the FastChat Controller

Initiate the FastChat controller by running the command below. This step is crucial for FastChat's functionality.

```bash
python -m fastchat.serve.controller
```

### Run the MLX Worker

Activate our mlx worker with the following command:

```bash
python -m gbx_lm.serve.mlx_fastchat_worker --model-path <input file path or a Hugging Face repo>
```

For instance:

```bash
python -m gbx_lm.serve.mlx_fastchat_worker --model-path GreenBitAI/Mistral-7B-Instruct-v0.2-layer-mix-bpw-3.0-mlx
```

### Launch the Gradio Web UI

Ensure Gradio is installed on your system. If not, you can install it using pip:

```bash
pip install gradio
```

To start the Gradio web UI, execute:

```bash
python -m fastchat.serve.gradio_web_server --share
```

Upon completing these steps, the worker should be operational and accessible via a local URL: http://0.0.0.0:7860. Open this URL in your preferred web browser to begin interacting with your local MLX LLM. Enjoy your conversations!

## License
- `mlx_worker.py` released under the [Apache 2.0 License](https://github.com/lm-sys/FastChat/tree/main/LICENSE) in [FastChat-serve](https://github.com/lm-sys/FastChat/tree/main/fastchat/serve).
- We release our changes and additions to these files under the [Apache 2.0 License](../../LICENSE).