# GBA Model Toolkit for MLX

## Introduction
Welcome to the GreenBitAI (GBA) Model Toolkit for [MLX](https://github.com/ml-explore/mlx)! This comprehensive Python package not only facilitates the conversion of [GreenBitAI's Low-bit Language Models (LLMs)](https://huggingface.co/collections/GreenBitAI/greenbitai-mlx-llm-6614eb6ceb8da657c2b4ed58) to MLX framework compatible format but also supports generation, model loading, and other essential scripts tailored for GBA quantized models. Designed to enhance the integration and deployment of GBA models within the MLX ecosystem, this toolkit enables the efficient execution of GBA models on a variety of platforms, with special optimizations for Apple devices to enable local inference and natural language content generation. 

## Installation

To get started with this package, simply run:
```bash
pip install gbx-lm
```
Optional dependencies: gbx-lm supports various optional features that can be installed as needed:

```bash
# Install with LangChain integration
pip install gbx-lm[langchain]

# Install with support for MLX-LM models in FastAPI server
pip install gbx-lm[mlx-lm]

# Install with development tools (testing)
pip install gbx-lm[dev]

# Install all optional dependencies
pip install gbx-lm[all]
```
Each extension provides specific functionality:
- langchain: Integration with LangChain for building AI applications
- mlx-lm: Support for loading and serving MLX-LM community models
- dev: Development and testing utilities

Or clone the repository and install the required dependencies (for Python >= 3.9):
```bash
git clone https://github.com/GreenBitAI/gbx-lm.git
```

via `requirements.txt` file:
```bash
pip install -r requirements.txt
```

or via `setup.py`:
```bash
# Basic editable installation
pip install -e . -v

# Install editable mode plus specific optional dependencies
pip install -e ".[langchain]" -v
pip install -e ".[mlx-lm]" -v
pip install -e ".[dev]" -v

# Install all optional dependencies
pip install -e ".[all]" -v
```
Alternatively you can also use the prepared conda environment configuration:
```bash
conda env create -f environment.yml
conda activate gbai_mlx_lm
```

## Usage

### Generating Content
To generate natural language content using a converted model:

- Example using terminal:
```bash
python -m gbx_lm.generate --model GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0-mlx  --max-tokens 100 --prompt "calculate 4*8+1024="
```

- Example code integration:
```bash
from gbx_lm import load, generate

model, tokenizer = load("GreenBitAI/Llama-3.2-3B-Instruct-layer-mix-bpw-4.0-mlx")

prompt = "What is the capital of France?"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
print(response)
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
3. To enable support for MLX-LM community models:
   ```bash
   pip install gbx-lm[mlx-lm]
   ```
   Then you can use models from the mlx-community organization:
   ```bash
   python -m gbx_lm.fastapi_server --model mlx-community/Qwen3-4B-4bit
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
python -m gbx_lm.gba2mlx --hf-path GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0 --mlx-path Llama-3-8B-instruct-layer-mix-bpw-4.0-mlx/ --hf-token <your huggingface token> --upload-repo GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0-mlx
```

### Evaluating Models
To evaluate a model, run:
```bash
gbx_lm.evaluate \
    --model gbx_model \
    --tasks winogrande boolq arc_challenge arc_easy hellaswag openbookqa piqa social_iqa   
```

## Requirements
- Python >= 3.9
- See `setup.py` for a complete list of dependencies

## License
The original code was released under its respective license and copyrights, i.e.:

- `generate.py`, `lora.py`, `*utils.py`, `tuner/*.py` and `models/*.py` etc. released under the [MIT License](https://github.com/ml-explore/mlx-examples/blob/main/LICENSE) in [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm).
- We release our changes and additions to these files under the [Apache 2.0 License](LICENSE).
