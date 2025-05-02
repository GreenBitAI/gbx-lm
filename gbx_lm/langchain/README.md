# GBX Langchain Demos

## Overview

GBX Langchain Demos showcase the integration of GBX language models with the Langchain framework, enabling powerful and flexible natural language processing capabilities.

## Installation

### Step 1: Install the gbx_lm Package

```bash
pip install gbx-lm
```

### Step 2: Install Langchain Package

```bash
pip install langchain-core
pip install json-repair
```

Ensure your system has Python3 and pip installed before proceeding.

## Usage

### Basic Example

Here's a basic example of how to use the GBX Langchain integration:

```python
from gbx_lm.langchain import ChatGBX
from gbx_lm.langchain import GBXPipeline

llm = GBXPipeline.from_model_id(
    model_id="GreenBitAI/Llama-3-8B-layer-mix-bpw-4.0-mlx",
)
chat = ChatGBX(llm=llm)
```

### Advanced Usage

#### Using `from_model_id` with custom parameters:

```python
from gbx_lm.langchain import GBXPipeline

pipe = GBXPipeline.from_model_id(
    model_id="GreenBitAI/Llama-3-8B-layer-mix-bpw-4.0-mlx",
    pipeline_kwargs={"max_tokens": 100, "temp": 0.7},
)
```

#### Passing model and tokenizer directly:

```python
from gbx_lm.langchain import GBXPipeline
from gbx_lm import load

model_id = "GreenBitAI/Llama-3-8B-layer-mix-bpw-4.0-mlx"
model, tokenizer = load(model_id)
pipe = GBXPipeline(model=model, tokenizer=tokenizer)
```
