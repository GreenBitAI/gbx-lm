# GBA Model Toolkit for MLX

## Introduction
Welcome to the GreenBitAI (GBA) Model Toolkit for [MLX](https://github.com/ml-explore/mlx)! This comprehensive Python package not only facilitates the conversion of [GreenBitAI's Low-bit Language Models (LLMs)](https://huggingface.co/collections/GreenBitAI/greenbitai-mlx-llm-6614eb6ceb8da657c2b4ed58) to MLX framework compatible format but also supports generation, model loading, and other essential scripts tailored for GBA quantized models. Designed to enhance the integration and deployment of GBA models within the MLX ecosystem, this toolkit enables the efficient execution of GBA models on a variety of platforms, with special optimizations for Apple devices to enable local inference and natural language content generation. 

## Installation
To get started with this package, simply run:
```bash
pip install gbx-lm
```
or clone the repository and install the required dependencies (for Python >= 3.9):
```bash
git clone https://github.com/GreenBitAI/gbx-lm.git
pip install -r requirements.txt
```
Alternatively you can also use the prepared conda environment configuration:
```bash
conda env create -f environment.yml
conda activate gbai_mlx_lm
```

## Usage

### Generating Content
To generate natural language content using a converted model:
```bash
python -m gbx_lm.generate --model <path to a converted model or a Hugging Face repo name>

# Example
python -m gbx_lm.generate --model GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0-mlx  --max-tokens 100 --prompt "calculate 4*8+1024="
```

### Interactive Chat
```bash
python -m gbx_lm.chat --model GreenBitAI/Llama-3.2-3B-Instruct-layer-mix-bpw-4.0-mlx  --max-tokens 100
```

### Managing Local Model
You can use the following scripts to explore and delete local models stored in the Hugging Face cache.
```shell
# List local models
python -m gbx_lm.manage --scan

# Specify a `--pattern`:
python -m gbx_lm.manage --scan --pattern GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-2.2-mlx

# To delete a model
python -m gbx_lm.manage --delete --pattern GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-2.2-mlx
```

### FastAPI Model Server
A high-performance HTTP API for text generation with GreenBitAI's mlx models. Improvements over the original `mlx-lm/server.py`:

- **Concurrent Processing**: Handles multiple requests simultaneously
- **Enhanced Performance**: Faster response times and better resource utilization
- **Robust Validation**: Automatic request validation and error handling
- **Interactive Docs**: Built-in Swagger UI for easy testing

#### Quick Start
1. Run:
   ```shell
   python -m gbx_lm.fastapi_server --model GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0-mlx
   ```
2. Use:
   ```shell
   # Chat
   curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
     -d '{"model": "default_model", "messages": [{"role": "user", "content": "Hello!"}]}'
   
   # Chat stream
   curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json"  \
     -d '{"model": "default_model", "messages": [{"role": "user", "content": "Hello!"}], "stream": "True"}'
   ```

#### Features
- Chat and text completion endpoints
- Streaming responses
- Customizable generation parameters
- Support for custom models and adapters

For API details, visit `http://localhost:8000/docs` after starting the server.

> Note: Not recommended for production without additional security measures.

### Converting Models
To convert a GreenBitAI's Low-bit LLM to the MLX format, run:
```bash
python -m gbx_lm.gba2mlx --hf-path <input file path or a Hugging Face repo> --mlx-path <output file path> --hf-token <your huggingface token> --upload-repo <a Hugging Face repo name>

# Example
python -m gbx_lm.gba2mlx --hf-path GreenBitAI/yi-6b-chat-w4a16g128 --mlx-path yi-6b-chat-w4a16g128-mlx/ --hf-token <your huggingface token> --upload-repo GreenBitAI/yi-6b-chat-w4a16g128-mlx
```

## Requirements
- Python 3.9/3.10
- See `requirements.txt` or `environment.yml` for a complete list of dependencies

## Web Demo
<img src="assets/web_chat_demo_mlx.gif" width="960">

We also prepared a demo for deploying chat applications by leveraging the capabilities of FastChat and Gradio.
By following this [instruction](https://github.com/GreenBitAI/gbx-lm/tree/main/gbx_lm/serve), you can quickly build a local chat demo page.

## License
The original code was released under its respective license and copyrights, i.e.:

- `generate.py`, `lora.py`, `*utils.py`, `tuner/*.py` and `models/*.py` etc. released under the [MIT License](https://github.com/ml-explore/mlx-examples/blob/main/LICENSE) in [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm).
- We release our changes and additions to these files under the [Apache 2.0 License](LICENSE).
