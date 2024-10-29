# Copyright © 2023-2024 Apple Inc.
# Additional code from GreenBitAI is licensed under the Apache 2.0 License.

# This refactored version uses FastAPI to improve concurrency performance of the original MLX "server.py".
#
# The main improvements and changes are:
# - Use of FastAPI framework: Replaced the original BaseHTTPRequestHandler with FastAPI to handle HTTP requests.
# - Asynchronous processing: Leveraged the asynchronous features of FastAPI, making all route handler functions asynchronous, which can significantly enhance concurrency performance.
# - Request validation: Utilized Pydantic models (CompletionRequest and ChatCompletionRequest) to validate and parse input data.
# - Streaming response: Used FastAPI's StreamingResponse to handle streaming outputs.
# - Route separation: Divided text completion and chat completion into two separate routes, making the code structure clearer.
# - Error handling: Employed FastAPI's exception handling mechanism to better manage and return errors.
# - Dependency injection: Used FastAPI's dependency injection system to manage the ModelProvider.
# - ASGI server: Adopted uvicorn as the ASGI server, which is faster and more reliable than Python's built-in HTTP server.
# - Type hints: Fully utilized Python's type hints to enhance code readability and maintainability.
# - Configuration management: Retained the original command-line argument configuration method, but it can be easily extended to use environment variables or configuration files.


import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, Set

import mlx.core as mx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .utils import generate_step, load


from .server_utils import (
    stopping_criteria,
    sequence_overlap,
    convert_chat,
    get_model_endpoint_path
)

# Global configurations
server_config = None
model_provider = None


def parse_args():
    parser = argparse.ArgumentParser(description="MLX FastAPI Server.")
    # Server configuration
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind the server")
    parser.add_argument("--port", type=int, default=8000, help="Port number to run the server")
    # Model configuration
    parser.add_argument("--model", type=str, help="The path to the MLX model weights, tokenizer, and config")
    parser.add_argument("--model_list", type=str, nargs="+", help="List of model paths to serve")
    parser.add_argument("--adapter_path", type=str, help="Optional path for the trained adapter weights and config.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Enable trusting remote code for tokenizer")
    parser.add_argument("--chat_template", type=str, default="", help="Specify a chat template for the tokenizer")
    parser.add_argument("--use_default_chat_template", action="store_true", help="Use the default chat template")
    parser.add_argument("--eos_token", type=str, default="<|eot_id|>", help="End of sequence token for tokenizer")
    return parser.parse_args()


class ServerConfig:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, **kwargs):
        self.host = host
        self.port = port
        self.model_config = argparse.Namespace(**kwargs)

        # Store the list of models to serve
        self.models_to_serve: Set[str] = set()
        if kwargs.get("model_list"):
            self.models_to_serve.update(kwargs["model_list"])
        elif kwargs.get("model"):
            self.models_to_serve.add(kwargs["model"])


class ModelProvider:
    def __init__(self, cli_args: argparse.Namespace):
        self.cli_args = cli_args
        self.model_cache: Dict[str, tuple] = {}  # Cache for loaded models

        # Initialize with default model if specified
        if self.cli_args.model_list:
            for model_path in self.cli_args.model_list:
                self.load(model_path)
        elif self.cli_args.model is not None:
            self.load(self.cli_args.model)

    def _validate_model_path(self, model_path: str):
        model_path = Path(model_path)
        if model_path.exists() and not model_path.is_relative_to(Path.cwd()):
            raise RuntimeError("Local models must be relative to the current working dir.")

    def load(self, model_path, adapter_path=None):
        cache_key = (model_path, adapter_path)
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        tokenizer_config = {
            "trust_remote_code": True if self.cli_args.trust_remote_code else None,
            "eos_token": self.cli_args.eos_token
        }
        if self.cli_args.chat_template:
            tokenizer_config["chat_template"] = self.cli_args.chat_template

        self._validate_model_path(model_path)
        model, tokenizer = load(
            model_path,
            adapter_path=(adapter_path if adapter_path else self.cli_args.adapter_path),
            tokenizer_config=tokenizer_config,
        )

        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template

        self.model_cache[cache_key] = (model, tokenizer)
        return model, tokenizer


def create_app(args):
    """Create and configure the FastAPI application with routes."""
    app = FastAPI(
        title="GBX-Model API",
        description="API using gbx-lm models",
    )

    # Create server config with all arguments
    server_config = ServerConfig(
        host=args.host,
        port=args.port,
        model=args.model,
        model_list=args.model_list,
        adapter_path=args.adapter_path,
        trust_remote_code=args.trust_remote_code,
        chat_template=args.chat_template,
        use_default_chat_template=args.use_default_chat_template,
        eos_token=args.eos_token
    )

    # Initialize model provider
    model_provider = ModelProvider(server_config.model_config)

    # Helper function to create endpoints for a specific model
    def create_model_endpoints(model_path: str):
        completion_path = get_model_endpoint_path(model_path, "completions")
        chat_completion_path = get_model_endpoint_path(model_path, "chat/completions")

        @app.post(completion_path, response_model=Dict)
        async def create_completion(request: CompletionRequest):
            model, tokenizer = model_provider.load(model_path)

            prompt = mx.array(tokenizer.encode(request.prompt))

            if request.stream:
                return StreamingResponse(
                    stream_completion(prompt, request, model, tokenizer),
                    media_type="text/event-stream"
                )
            else:
                return JSONResponse(generate_completion(prompt, request, model, tokenizer))

        @app.post(chat_completion_path, response_model=Dict)
        async def create_chat_completion(request: ChatCompletionRequest):
            model, tokenizer = model_provider.load(model_path)

            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(
                    request.messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            else:
                prompt = convert_chat(request.messages)
                prompt = tokenizer.encode(prompt)

            prompt = mx.array(prompt)

            if request.stream:
                return StreamingResponse(
                    stream_chat_completion(prompt, request, model, tokenizer),
                    media_type="text/event-stream"
                )
            else:
                return JSONResponse(generate_chat_completion(prompt, request, model, tokenizer))

    # Create endpoints for each model
    for model_path in server_config.models_to_serve:
        create_model_endpoints(model_path)

    # Add root endpoint for API information
    @app.get("/")
    async def root():
        return {
            "api": "GBX-Model API",
            "version": "1.0",
            "models": list(server_config.models_to_serve),
            "endpoints": [
                             get_model_endpoint_path(model, "completions")
                             for model in server_config.models_to_serve
                         ] + [
                             get_model_endpoint_path(model, "chat/completions")
                             for model in server_config.models_to_serve
                         ]
        }

    return app, server_config


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    logit_bias: Optional[Dict[str, float]] = None
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    logit_bias: Optional[Dict[str, float]] = None
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20


async def stream_completion(prompt, request, model, tokenizer):
    created = int(time.time())
    request_id = f"cmpl-{uuid.uuid4()}"

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    tokens = []

    stop_id_sequences = []
    if request.stop:
        stop_words = request.stop if isinstance(request.stop, list) else [request.stop]
        stop_id_sequences = [tokenizer.encode(stop) for stop in stop_words]

    for (token, _, _), _ in zip(
            generate_step(
                prompt=prompt,
                model=model,
                temp=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size,
                logit_bias=request.logit_bias,
            ),
            range(request.max_tokens),
    ):
        detokenizer.add_token(token)
        tokens.append(token)

        stop_condition = stopping_criteria(tokens, stop_id_sequences, tokenizer.eos_token_id)
        if stop_condition.stop_met:
            break

        if any(sequence_overlap(tokens, sequence) for sequence in stop_id_sequences):
            continue

        new_text = detokenizer.last_segment
        response = {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "text": new_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None,
                }
            ]
        }
        yield f"data: {json.dumps(response)}\n\n"

    yield "data: [DONE]\n\n"


async def stream_chat_completion(prompt, request, model, tokenizer):
    created = int(time.time())
    request_id = f"chatcmpl-{uuid.uuid4()}"

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    tokens = []

    stop_id_sequences = []
    if request.stop:
        stop_words = request.stop if isinstance(request.stop, list) else [request.stop]
        stop_id_sequences = [tokenizer.encode(stop) for stop in stop_words]

    for (token, _, _), _ in zip(
            generate_step(
                prompt=prompt,
                model=model,
                temp=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size,
                logit_bias=request.logit_bias,
            ),
            range(request.max_tokens),
    ):
        detokenizer.add_token(token)
        tokens.append(token)

        stop_condition = stopping_criteria(tokens, stop_id_sequences, tokenizer.eos_token_id)
        if stop_condition.stop_met:
            break

        if any(sequence_overlap(tokens, sequence) for sequence in stop_id_sequences):
            continue

        new_text = detokenizer.last_segment
        response = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "delta": {"role": "assistant", "content": new_text},
                    "index": 0,
                    "finish_reason": None,
                }
            ]
        }
        yield f"data: {json.dumps(response)}\n\n"

    yield "data: [DONE]\n\n"


def generate_completion(prompt, request, model, tokenizer):
    created = int(time.time())
    request_id = f"cmpl-{uuid.uuid4()}"

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    tokens = []

    stop_id_sequences = []
    if request.stop:
        stop_words = request.stop if isinstance(request.stop, list) else [request.stop]
        stop_id_sequences = [tokenizer.encode(stop) for stop in stop_words]

    for (token, _, _), _ in zip(
            generate_step(
                prompt=prompt,
                model=model,
                temp=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size,
                logit_bias=request.logit_bias,
            ),
            range(request.max_tokens),
    ):
        detokenizer.add_token(token)
        tokens.append(token)

        stop_condition = stopping_criteria(tokens, stop_id_sequences, tokenizer.eos_token_id)
        if stop_condition.stop_met:
            break

    detokenizer.finalize()
    text = detokenizer.text

    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length" if len(tokens) == request.max_tokens else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(tokens),
            "total_tokens": len(prompt) + len(tokens),
        },
    }


def generate_chat_completion(prompt, request, model, tokenizer):
    created = int(time.time())
    request_id = f"chatcmpl-{uuid.uuid4()}"

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    tokens = []

    stop_id_sequences = []
    if request.stop:
        stop_words = request.stop if isinstance(request.stop, list) else [request.stop]
        stop_id_sequences = [tokenizer.encode(stop) for stop in stop_words]

    for (token, _, _), _ in zip(
            generate_step(
                prompt=prompt,
                model=model,
                temp=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size,
                logit_bias=request.logit_bias,
            ),
            range(request.max_tokens),
    ):
        detokenizer.add_token(token)
        tokens.append(token)

        stop_condition = stopping_criteria(tokens, stop_id_sequences, tokenizer.eos_token_id)
        if stop_condition.stop_met:
            break

    detokenizer.finalize()
    text = detokenizer.text

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "index": 0,
                "finish_reason": "length" if len(tokens) == request.max_tokens else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(tokens),
            "total_tokens": len(prompt) + len(tokens),
        },
    }


def main():
    import uvicorn
    args = parse_args()
    app, server_config = create_app(args)
    uvicorn.run(app, host=server_config.host, port=server_config.port)

if __name__ == "__main__":
    main()