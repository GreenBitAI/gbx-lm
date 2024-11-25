import os
import logging
from datetime import datetime
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
import asyncio

from .utils import generate_step, load

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"


from .server_utils import (
    stopping_criteria,
    sequence_overlap,
    convert_chat,
    get_model_endpoint_path
)

# Global configurations
server_config = None
model_provider = None
logger = None

# Global UE confidence scorers
UE_MODELS = {
    "qwen": "qwen2.5",
    "llama": "llama-3"
}
_confidence_scorers = {}

def setup_logging():
    """Configure logging for the FastAPI server."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Create timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f"server_{timestamp}.log"

    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )

    # Create logger
    logger = logging.getLogger("gbx_server")

    # Log startup message
    logger.info(f"Starting GBX-Model API server. Log file: {log_file}")
    return logger


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
    parser.add_argument("--ue_parameter_path", type=str, default="db/router.db", help="Path to the method parameters database")
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
        try:
            if self.cli_args.model_list:
                for model_path in self.cli_args.model_list:
                    self.load(model_path)
            elif self.cli_args.model is not None:
                self.load(self.cli_args.model)
        except Exception as e:
            logger.error(f"Failed to initialize ModelProvider: {str(e)}")
            raise

    def _validate_model_path(self, model_path: str):
        try:
            model_path = Path(model_path)
            if model_path.exists() and not model_path.is_relative_to(Path.cwd()):
                logger.error(f"Invalid model path: {model_path}")
                raise RuntimeError("Local models must be relative to the current working dir.")
        except Exception as e:
            logger.error(f"Model path validation failed: {str(e)}")
            raise

    def load(self, model_path, adapter_path=None):
        try:
            cache_key = (model_path, adapter_path)
            if cache_key in self.model_cache:
                return self.model_cache[cache_key]

            tokenizer_config = {
                "trust_remote_code": True if self.cli_args.trust_remote_code else None,
                "eos_token": "<|im_end|>" if model_path.lower().__contains__("qwen") else "<|eot_id|>",
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
            logger.info(f"Successfully loaded model from {model_path}")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise

def convert_hidden_states_to_list(hidden_states):
    """Convert MLX array hidden states to nested Python lists."""
    if hidden_states is None or hidden_states == []:
        return None
    return [h.tolist() if isinstance(h, mx.array) else h for h in hidden_states]


def create_app(args):
    """Create and configure the FastAPI application with routes."""

    # Initialize logging
    global logger
    logger = setup_logging()

    try:
        from .routing import ConfidenceScorer
        for model_family, model_id in UE_MODELS.items():
            scorer = ConfidenceScorer(
                parameters_path=args.ue_parameter_path,
                model_id=model_id
            )
            _confidence_scorers[model_family] = scorer
    except Exception as e:
        logger.error(f"Error loading confidence scorers: {str(e)}")

    app = FastAPI(
        title="GBX-Model API",
        description="API using gbx-lm models",
    )

    # Add logging middleware
    @app.middleware("http")
    async def log_requests(request, call_next):
        start_time = time.time()
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            logger.info(
                f"Request: {request.method} {request.url.path} "
                f"Status: {response.status_code} "
                f"Duration: {duration:.2f}s"
            )
            return response
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise

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
            try:
                model, tokenizer = model_provider.load(model_path)
                prompt = mx.array(tokenizer.encode(request.prompt))

                if request.stream:
                    return StreamingResponse(
                        stream_completion(prompt, request, model, tokenizer),
                        media_type="text/event-stream"
                    )
                else:
                    result = await generate_completion(prompt, request, model, tokenizer)
                    return JSONResponse(result)
            except Exception as e:
                logger.error(f"Completion request failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post(chat_completion_path, response_model=Dict)
        async def create_chat_completion(request: ChatCompletionRequest):
            try:
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
                    result = await generate_chat_completion(prompt, request, model, tokenizer)
                    return JSONResponse(result)
            except Exception as e:
                logger.error(f"Chat completion request failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    # Create endpoints for each model
    for model_path in server_config.models_to_serve:
        try:
            create_model_endpoints(model_path)
        except Exception as e:
            logger.error(f"Failed to create endpoints for model {model_path}: {str(e)}")
            raise

    # Add root endpoint for API information
    @app.get("/")
    async def root():
        try:
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
        except Exception as e:
            logger.error(f"Root endpoint request failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"}
        )

    return app, server_config, logger


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
    with_hidden_states: bool = False
    remote_score: bool = True

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
    with_hidden_states: bool = False
    remote_score: bool = True


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

    for gen_result, _ in zip(
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
        token = gen_result[0]  # Safely extract token from the tuple
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

    for gen_result, _ in zip(
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
        token = gen_result[0]
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


async def async_generate_step(prompt, model, temp, top_p, repetition_penalty,
                            repetition_context_size, logit_bias, with_hidden_states, max_tokens):
    """Wrap the synchronous generate_step as an async generator."""
    for gen_output, _ in zip(
        generate_step(
            prompt=prompt,
            model=model,
            temp=temp,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            logit_bias=logit_bias,
            with_hidden_states=with_hidden_states,
        ),
        range(max_tokens),
    ):
        yield gen_output
        await asyncio.sleep(0)

async def generate_completion(prompt, request, model, tokenizer):
    created = int(time.time())
    request_id = f"cmpl-{uuid.uuid4()}"

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    tokens = []
    all_hidden_states = [] if request.with_hidden_states else None

    stop_id_sequences = []
    if request.stop:
        stop_words = request.stop if isinstance(request.stop, list) else [request.stop]
        stop_id_sequences = [tokenizer.encode(stop) for stop in stop_words]

    try:
        async for gen_output in async_generate_step(
                prompt=prompt,
                model=model,
                temp=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size,
                logit_bias=request.logit_bias,
                with_hidden_states=request.with_hidden_states,
                max_tokens=request.max_tokens
        ):
            (token, _, hidden_states) = gen_output

            detokenizer.add_token(token)
            tokens.append(token)

            stop_condition = stopping_criteria(tokens, stop_id_sequences, tokenizer.eos_token_id)
            if stop_condition.stop_met:
                break

            await asyncio.sleep(0)

        detokenizer.finalize()
        text = detokenizer.text

        score = None
        if request.with_hidden_states and hidden_states is not None:
            # averaging the hidden states of the prompt tokens for reducing the data transfer overhead.
            avg_hs = hidden_states.mean(axis=1)
            if request.remote_score:
                model_key = "qwen" if request.model.lower().__contains__("qwen") else "llama"
                scorer = _confidence_scorers.get(model_key)
                if scorer:
                    score = await asyncio.to_thread(scorer.calculate_confidence, avg_hs.tolist())
            else:
                all_hidden_states.append(avg_hs)

        # Convert hidden states to regular Python lists before JSON serialization
        serializable_hidden_states = convert_hidden_states_to_list(all_hidden_states)

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
                    "hidden_states": serializable_hidden_states,
                    "confidence_score": score
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(tokens),
                "total_tokens": len(prompt) + len(tokens),
            },
        }
    except Exception as e:
        logger.error(f"Async completion generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_chat_completion(prompt, request, model, tokenizer):
    created = int(time.time())
    request_id = f"chatcmpl-{uuid.uuid4()}"

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    tokens = []
    all_hidden_states = [] if request.with_hidden_states else None

    stop_id_sequences = []
    if request.stop:
        stop_words = request.stop if isinstance(request.stop, list) else [request.stop]
        stop_id_sequences = [tokenizer.encode(stop) for stop in stop_words]

    try:
        async for gen_output in async_generate_step(
                prompt=prompt,
                model=model,
                temp=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size,
                logit_bias=request.logit_bias,
                with_hidden_states=request.with_hidden_states,
                max_tokens=request.max_tokens
        ):
            (token, _, hidden_states) = gen_output

            detokenizer.add_token(token)
            tokens.append(token)

            stop_condition = stopping_criteria(tokens, stop_id_sequences, tokenizer.eos_token_id)
            if stop_condition.stop_met:
                break

            await asyncio.sleep(0)

        detokenizer.finalize()
        text = detokenizer.text

        score = None
        if request.with_hidden_states and hidden_states is not None:
            # averaging the hidden states of the prompt tokens for reducing the data transfer overhead.
            avg_hs = hidden_states.mean(axis=1)
            if request.remote_score:
                model_key = "qwen" if request.model.lower().__contains__("qwen") else "llama"
                scorer = _confidence_scorers.get(model_key)
                if scorer:
                    score = await asyncio.to_thread(scorer.calculate_confidence, avg_hs.tolist())
            else:
                all_hidden_states.append(avg_hs)

        # Convert hidden states to regular Python lists before JSON serialization
        serializable_hidden_states = convert_hidden_states_to_list(all_hidden_states)

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": text,
                        "hidden_states": serializable_hidden_states,
                        "confidence_score": score
                    },
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
    except Exception as e:
        logger.error(f"Async chat completion generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    try:
        import uvicorn
        args = parse_args()
        app, server_config, logger = create_app(args)
        logger.info(f"Starting server on {server_config.host}:{server_config.port}")
        uvicorn.run(app, host=server_config.host, port=server_config.port)
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()