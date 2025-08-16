import os
import logging
from datetime import datetime
import argparse
import json
import time
import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Set

import mlx.core as mx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import asyncio

from gbx_lm.utils import generate_step, load
from gbx_lm.sample_utils import make_sampler
from gbx_lm.prompt_cache import PromptCache
from gbx_lm.infer_opt import eminf_generate, eminf_generate_step

# Try to import mlx_lm for mlx-community models
try:
    import mlx_lm
    HAVE_MLX_LM = True
except ImportError:
    HAVE_MLX_LM = False


PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"


from .server_utils import (
    stopping_criteria,
    sequence_overlap,
)

# Global configurations
server_config = None
model_provider = None
logger = None

# Global UE confidence scorers
UE_MODELS = {
    "qwen-2.5-7b": "qwen2.5",
    "llama-3-8b": "llama-3"
}

_confidence_scorers = {}

# {system_prompt_hash: PromptCache}
_base_caches = {}


def get_system_prompt_from_messages(messages: List[Dict[str, str]]) -> Optional[str]:
    """Extract the system prompt from the message list"""
    for msg in messages:
        if msg.get("role") == "system":
            return msg.get("content", "")
    return None

def calculate_cached_tokens(
        tokens_processed: int,
        total_tokens: int,
        cache_hit: bool,
) -> int:
    """Calculate the number of cached tokens, in accordance with OpenAI's 128 increment rule"""
    if not cache_hit or total_tokens < 1024:
        return 0

    # Calculate the actual cached tokens
    cached_tokens = total_tokens - tokens_processed
    # Align by 128 increments (in accordance with OpenAI rules)
    return max(1024, (cached_tokens // 128) * 128)

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

def get_model_key(request_model: str) -> str:
    """
    Determine the corresponding model key based on the requested model name

    Args:
        request_model: The model name in the request
    Returns:
        str: The matched model key
    """
    request_model = request_model.lower()

    # First try exact match
    if request_model in UE_MODELS:
        return request_model

    # If no exact match, try standardizing the model name format
    model_families = {
        "qwen-2.5-7b": ["qwen2.5-7b", "qwen-2.5-7b"],
        "llama-3-8b": ["llama3-8b", "llama-3-8b"]
    }

    for standard_name, variants in model_families.items():
        if any(variant in request_model for variant in variants):
            return standard_name

    # If no match is found, raise exception
    raise ValueError(f"Error: Unsupported model: {request_model}")


def is_qwen3_model(model_name: str) -> bool:
    """
    Check if the model is a Qwen3 model which supports enable_thinking parameter.

    Args:
        model_name: The model name to check

    Returns:
        bool: True if it's a Qwen3 model, False otherwise
    """
    model_name_lower = model_name.lower()
    return any([
        "qwen3-" in model_name_lower,
        "qwen-3-" in model_name_lower
    ])

def parse_args():
    parser = argparse.ArgumentParser(description="MLX FastAPI Server.")
    # Server configuration
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind the server")
    parser.add_argument("--port", type=int, default=11688, help="Port number to run the server")
    # Model configuration
    parser.add_argument("--model", type=str, help="The path to the MLX model weights, tokenizer, and config")
    parser.add_argument("--model_list", type=str, nargs="+", help="List of model paths to serve")
    parser.add_argument("--adapter_path", type=str, help="Optional path for the trained adapter weights and config.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Enable trusting remote code for tokenizer")
    parser.add_argument("--chat_template", type=str, default="", help="Specify a chat template for the tokenizer")
    parser.add_argument("--use_default_chat_template", action="store_true", help="Use the default chat template")
    parser.add_argument("--eos_token", type=str, default="<|eot_id|>", help="End of sequence token for tokenizer")
    parser.add_argument("--ue_parameter_path", type=str, default="db/router.db", help="Path to the method parameters database")
    parser.add_argument("--base_system_prompts", type=str, nargs="*",
                        default=["You are a helpful AI assistant.", "You are nice friend of human."],
                        help="List of base system prompts for pre-caching")

    return parser.parse_args()


class ServerConfig:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, **kwargs):
        self.host = host
        self.port = port
        self.model_config = argparse.Namespace(**kwargs)

        # to ensure thread safe
        self.model_locks = {}

        self.base_system_prompts = kwargs.get("base_system_prompts", ["You are a helpful AI assistant."])
        # Session-level cache storage: {prompt_cache_key: PromptCache}
        self.session_caches = {}

        # Store the list of models to serve
        self.models_to_serve: Set[str] = set()

        if kwargs.get("model_list"):
            self.models_to_serve.update(kwargs["model_list"])
        elif kwargs.get("model"):
            self.models_to_serve.add(kwargs["model"])

    def get_model_lock(self, model_path: str):
        """Get the lock for the specified model, creating it if it does not exist."""
        if model_path not in self.model_locks:
            self.model_locks[model_path] = asyncio.Lock()
        return self.model_locks[model_path]

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

            # Check if this is an mlx-community model and if mlx-lm is available
            is_mlx_community = model_path.startswith("mlx-community/") and HAVE_MLX_LM

            if is_mlx_community:
                # Use mlx-lm's load function
                logger.info(f"Using mlx_lm to load model from {model_path}")
                model, tokenizer = mlx_lm.load(
                    model_path,
                    adapter_path=(adapter_path if adapter_path else self.cli_args.adapter_path),
                    tokenizer_config={"trust_remote_code": True}
                )
            else:
                # Original GreenBitAI model loading logic
                logger.info(f"Using gbx-lm to load model from {model_path}")
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

            # Pre-cache base system prompts
            if server_config is not None:
                for i, system_prompt in enumerate(server_config.base_system_prompts):
                    prompt_hash = hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()[:8]

                    if prompt_hash not in _base_caches:
                        try:
                            prompt_cache_obj = PromptCache()
                            prompt_cache_obj.cache_system_prompt(model, system_prompt, tokenizer)
                            _base_caches[prompt_hash] = prompt_cache_obj
                            logger.info(f"Pre-cached base system prompt {i + 1} with hash: {prompt_hash}")
                        except Exception as e:
                            logger.warning(f"Failed to pre-cache base system prompt {i + 1}: {e}")

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
    global logger, server_config
    logger = setup_logging()

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
        eos_token=args.eos_token,
        base_system_prompts=args.base_system_prompts
    )

    try:
        from .routing import ConfidenceScorer
        for model_family, model_id in UE_MODELS.items():
            scorer = ConfidenceScorer(
                parameters_path=args.ue_parameter_path,
                model_id=model_id
            )
            _confidence_scorers[model_family] = scorer
    except Exception as e:
        logger.warning(f"Loading confidence scorers: {str(e)}")

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

    # Initialize model provider
    model_provider = ModelProvider(server_config.model_config)

    def validate_and_get_model_path(model_name: str) -> str:
        """Validate if the requested model is available and return its path."""
        if model_name not in server_config.models_to_serve:
            available_models = list(server_config.models_to_serve)
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not found. Available models: {available_models}"
            )
        return model_name

    @app.post("/v1/completions", response_model=Dict)
    async def create_completion(request: CompletionRequest):
        try:
            if request.use_eminf:
                raise HTTPException(
                    status_code=400,
                    detail="EMINF optimization is only supported for chat completions, not text completions"
                )

            # Verify and get the model path
            model_path = validate_and_get_model_path(request.model)
            model_lock = server_config.get_model_lock(model_path)

            async with model_lock:
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

    @app.post("/v1/chat/completions", response_model=Dict)
    async def create_chat_completion(request: ChatCompletionRequest):
        try:
            # Verify and get the model path
            model_path = validate_and_get_model_path(request.model)
            model_lock = server_config.get_model_lock(model_path)

            async with model_lock:
                model, tokenizer = model_provider.load(model_path)

                prompt = None
                if is_qwen3_model(model_path):
                    enable_thinking = request.enable_thinking
                    prompt = tokenizer.apply_chat_template(
                        request.messages,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking
                    )
                else:
                    prompt = tokenizer.apply_chat_template(
                        request.messages,
                        add_generation_prompt=True,
                    )

                if not isinstance(prompt, mx.array):
                    if isinstance(prompt, str):
                        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                            tokenizer.bos_token
                        )
                        prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
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

    # Add root endpoint for API information
    @app.get("/")
    async def root():
        try:
            return {
                "api": "GBX-Model API",
                "version": "1.0",
                "models": list(server_config.models_to_serve),
                "endpoints": [
                    "/v1/completions",
                    "/v1/chat/completions",
                    "/v1/models",
                    "/v1/prompt_cache_status",
                    "/v1/prompt_cache/{cache_key}"
                ]
            }
        except Exception as e:
            logger.error(f"Root endpoint request failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/v1/models")
    async def list_models():
        """List available models in OpenAI API format."""
        try:
            models = []
            for model_name in server_config.models_to_serve:
                models.append({
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "gbx-lm"
                })

            return {
                "object": "list",
                "data": models
            }
        except Exception as e:
            logger.error(f"Models list request failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/v1/prompt_cache_status")
    async def get_prompt_cache_status():
        """View the current prompt cache status"""
        try:
            base_cache_info = []
            for prompt_hash, prompt_cache_obj in _base_caches.items():
                info = {
                    "prompt_hash": prompt_hash,
                    "system_tokens_count": len(prompt_cache_obj.system_tokens),
                    "type": "base_cache"
                }
                base_cache_info.append(info)

            session_cache_info = []
            for cache_key, prompt_cache_obj in server_config.session_caches.items():
                info = {
                    "prompt_cache_key": cache_key,
                    "system_cached": prompt_cache_obj.system_cached,
                    "system_tokens_count": len(prompt_cache_obj.system_tokens),
                    "conversation_tokens_count": len(prompt_cache_obj.tokens_no_gen),
                    "type": "session_cache"
                }
                session_cache_info.append(info)

            return {
                "base_caches": base_cache_info,
                "session_caches": session_cache_info,
                "total_base_caches": len(_base_caches),
                "total_session_caches": len(server_config.session_caches),
                "note": "Base caches are pre-computed and shared, session caches are per-request"
            }
        except Exception as e:
            logger.error(f"Failed to get cache status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/v1/prompt_cache/{cache_key}")
    async def delete_prompt_cache(cache_key: str):
        """Delete a specific prompt cache by its key"""
        try:
            if cache_key in server_config.session_caches:
                del server_config.session_caches[cache_key]
                logger.info(f"Deleted prompt cache with key: {cache_key}")
                return {
                    "message": f"Successfully deleted prompt cache: {cache_key}",
                    "deleted_key": cache_key
                }
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Prompt cache key '{cache_key}' not found"
                )
        except Exception as e:
            logger.error(f"Failed to delete prompt cache '{cache_key}': {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

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
    use_eminf: bool = False


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
    enable_thinking: Optional[bool] = None  # Qwen3 Model only feature
    prompt_cache_key: Optional[str] = None
    use_eminf: bool = False


def copy_prompt_cache(source_cache_obj: PromptCache, model) -> PromptCache:
    """
    Creates a new copy of an existing PromptCache to avoid recalculating the system prompt.

    Args:
        source_cache_obj: Source PromptCache object
        model: Target model
    Returns:
        PromptCache: New copy of PromptCache
    """
    new_cache_obj = PromptCache()

    new_cache_obj.model_key = source_cache_obj.model_key
    new_cache_obj.system_cached = source_cache_obj.system_cached
    new_cache_obj.system_tokens = source_cache_obj.system_tokens.copy()
    new_cache_obj.tokens_no_gen = source_cache_obj.tokens_no_gen.copy()

    try:
        from .models.cache import make_prompt_cache
        new_cache_obj.cache = make_prompt_cache(model)

        if source_cache_obj.cache and source_cache_obj.system_cached:
            # copy each layer cache state
            for i, (src_cache, new_cache) in enumerate(zip(source_cache_obj.cache, new_cache_obj.cache)):
                if hasattr(src_cache, 'state') and hasattr(new_cache, 'state'):
                    # copy KV cache state
                    new_cache.state = [mx.array(state) for state in src_cache.state]
                    new_cache.offset = src_cache.offset

            logger.info(f"Successfully copied cache state without recomputation")

    except Exception as e:
        logger.warning(f"Failed to copy cache state, falling back to recomputation: {e}")
        new_cache_obj.cache = None

    return new_cache_obj

def handle_prompt_cache(request, model, tokenizer, prompt):
    """
    Handling the prompt cache's hierarchical caching logic

    Returns:
        tuple: (tokens_to_process, prompt_cache, cache_hit, cache_obj, original_prompt_len)
    """
    cache_hit = False
    tokens_to_process = prompt
    prompt_cache = None
    original_prompt_len = len(prompt)
    cache_obj = None

    if request.prompt_cache_key:
        try:
            system_prompt = get_system_prompt_from_messages(request.messages)

            if system_prompt:
                # First check the session cache
                cache_obj = server_config.session_caches.get(request.prompt_cache_key)

                if cache_obj is None:
                    # Then check the base cache
                    prompt_hash = hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()[:8]
                    base_cache = _base_caches.get(prompt_hash)

                    if base_cache:
                        cache_obj = copy_prompt_cache(base_cache, model)
                        logger.info(f"Copied cache from base cache with hash: {prompt_hash}")
                    else:
                        # Finally check if there is the same system prompt in the existing session cache
                        existing_cache_for_same_prompt = None

                        test_tokens = tokenizer.apply_chat_template(
                            [{"role": "system", "content": system_prompt}],
                            add_generation_prompt=False, enable_thinking=False
                        )

                        for existing_key, existing_cache in server_config.session_caches.items():
                            if (existing_cache.system_cached and
                                    len(existing_cache.system_tokens) > 0):
                                if test_tokens == existing_cache.system_tokens:
                                    existing_cache_for_same_prompt = existing_cache
                                    logger.info(
                                        f"Found existing session cache with same system prompt: {existing_key}")
                                    break

                        if existing_cache_for_same_prompt:
                            cache_obj = copy_prompt_cache(existing_cache_for_same_prompt, model)
                            logger.info(f"Copied cache state from existing session cache, no recomputation needed")
                        else:
                            cache_obj = PromptCache()
                            cache_obj.cache_system_prompt(model, system_prompt, tokenizer)
                            logger.info(
                                f"Created new cache with fresh computation for key: {request.prompt_cache_key}")

                    server_config.session_caches[request.prompt_cache_key] = cache_obj

                tokens_no_gen = tokenizer.apply_chat_template(
                    request.messages, add_generation_prompt=False, enable_thinking=False
                )
                tokens_with_gen = list(prompt)
                model_key = getattr(model, "model_key", id(model))

                tokens_to_process, prompt_cache, cache_hit = cache_obj.get_prompt_cache(
                    model, tokens_with_gen, tokens_no_gen, model_key
                )

                if cache_hit:
                    logger.info(
                        f"Cache HIT for '{request.prompt_cache_key}'! Processing {len(tokens_to_process)}/{original_prompt_len} tokens")
                else:
                    logger.info(f"Cache MISS for '{request.prompt_cache_key}' - processing all {original_prompt_len} tokens")
            else:
                logger.warning(f"prompt_cache_key provided but no system prompt found")
        except Exception as e:
            logger.warning(f"Prompt cache failed for '{request.prompt_cache_key}': {e}")
            cache_hit = False

    if not isinstance(tokens_to_process, mx.array):
        tokens_to_process = mx.array(tokens_to_process)

    return tokens_to_process, prompt_cache, cache_hit, cache_obj, original_prompt_len

async def stream_completion(prompt, request, model, tokenizer):
    created = int(time.time())
    request_id = f"cmpl-{uuid.uuid4()}"

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    tokens = []
    is_first_chunk = True

    stop_id_sequences = []
    if request.stop:
        stop_words = request.stop if isinstance(request.stop, list) else [request.stop]
        stop_id_sequences = [tokenizer.encode(stop) for stop in stop_words]

    async for (token, _, hidden_states) in async_generate_step(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        temp=request.temperature,
        top_p=request.top_p,
        with_hidden_states=request.with_hidden_states,
        max_tokens=request.max_tokens,
        use_eminf=request.use_eminf
    ):
        token = token
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

        # Add usage info in first chunk
        if is_first_chunk:
            response["usage"] = {
                "input_tokens": len(prompt),
                "output_tokens": len(tokens),
                "total_tokens": len(prompt) + len(tokens)
            }
            is_first_chunk = False

        yield f"data: {json.dumps(response)}\n\n"

    # Final chunk with complete usage stats
    final_response = {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "text": "",
                "index": 0,
                "logprobs": None,
                "finish_reason": "max_tokens" if len(tokens) == request.max_tokens else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(tokens),
            "total_tokens": len(prompt) + len(tokens)
        }
    }
    yield f"data: {json.dumps(final_response)}\n\n"
    yield "data: [DONE]\n\n"

async def stream_chat_completion(prompt, request, model, tokenizer):
    created = int(time.time())
    request_id = f"chatcmpl-{uuid.uuid4()}"

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    tokens = []
    is_first_chunk = True
    generated_text_parts = []

    stop_id_sequences = []
    if request.stop:
        stop_words = request.stop if isinstance(request.stop, list) else [request.stop]
        stop_id_sequences = [tokenizer.encode(stop) for stop in stop_words]

    # Using cache processing functions
    tokens_to_process, prompt_cache, cache_hit, cache_obj, original_prompt_len = handle_prompt_cache(
        request, model, tokenizer, prompt
    )

    if request.use_eminf:
        request._cache_obj = cache_obj  # Temporarily store cache_obj in the request object

    async for (token, _, hidden_states) in async_generate_step(
        prompt=tokens_to_process,
        model=model,
        tokenizer=tokenizer,
        temp=request.temperature,
        top_p=request.top_p,
        with_hidden_states=request.with_hidden_states,
        max_tokens=request.max_tokens,
        prompt_cache=prompt_cache,
        use_eminf=request.use_eminf,
        request=request
    ):
        detokenizer.add_token(token)
        tokens.append(token)

        stop_condition = stopping_criteria(tokens, stop_id_sequences, tokenizer.eos_token_id)
        if stop_condition.stop_met:
            break

        if any(sequence_overlap(tokens, sequence) for sequence in stop_id_sequences):
            continue

        new_text = detokenizer.last_segment
        generated_text_parts.append(new_text)

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

        # Add usage info in first chunk
        if is_first_chunk:
            # Calculate cache token number
            cached_tokens = calculate_cached_tokens(
                len(tokens_to_process), original_prompt_len, cache_hit
            )
            response["usage"] = {
                "input_tokens": original_prompt_len,
                "output_tokens": len(tokens),
                "total_tokens": original_prompt_len + len(tokens),
                "prompt_tokens_details": {
                    "cached_tokens": cached_tokens
                }
            }
            is_first_chunk = False

        yield f"data: {json.dumps(response)}\n\n"

    # Update the cache status after the generation is completed
    generated_text = "".join(generated_text_parts)
    if request.prompt_cache_key and cache_hit and cache_obj:
        try:
            updated_messages = request.messages + [{"role": "assistant", "content": generated_text}]
            cache_obj.update_after_step(updated_messages, tokenizer)
        except Exception as e:
            logger.warning(f"Failed to update stream prompt cache: {e}")

    # Calculate the final number of cache tokens
    cached_tokens = calculate_cached_tokens(
        len(tokens_to_process), original_prompt_len, cache_hit
    )

    # Final chunk with complete usage stats
    final_response = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "delta": {"role": "assistant", "content": ""},
                "index": 0,
                "finish_reason": "max_tokens" if len(tokens) == request.max_tokens else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": original_prompt_len,
            "completion_tokens": len(tokens),
            "total_tokens": original_prompt_len + len(tokens),
            "prompt_tokens_details": {
                "cached_tokens": cached_tokens
            }
        }
    }
    yield f"data: {json.dumps(final_response)}\n\n"
    yield "data: [DONE]\n\n"


async def async_eminf_generate_step(
        prompt, model, tokenizer, temp, top_p, with_hidden_states, max_tokens,
        prompt_cache=None, request=None, cache_obj=None
):
    """
    Async wrapper for EMINF generation that yields tokens incrementally.
    """
    try:
        # Get input_ids for EMINF
        if isinstance(prompt, mx.array):
            input_ids = prompt.tolist()
        else:
            input_ids = list(prompt)

        # Reconstruct input_ids_no_gen without generation prompt
        if request and hasattr(request, 'messages'):
            enable_thinking = getattr(request, 'enable_thinking', False) if hasattr(request,
                                                                                    'enable_thinking') else False
            input_ids_no_gen = tokenizer.apply_chat_template(
                request.messages,
                add_generation_prompt=False,
                enable_thinking=enable_thinking
            )
        else:
            # Fallback: estimate by removing typical generation prompt tokens
            input_ids_no_gen = input_ids.copy()  # This is suboptimal but necessary fallback

        # Use streaming EMINF generation
        for token_data in eminf_generate_step(
                model, tokenizer, input_ids, input_ids_no_gen,
                max_tokens=max_tokens, prompt_cache=cache_obj, use_cache=cache_obj is not None
        ):
            yield token_data
            await asyncio.sleep(0)  # Allow other tasks to run

    except Exception as e:
        logger.error(f"EMINF generation failed: {str(e)}")

async def async_generate_step(
        prompt, model, tokenizer, temp, top_p, with_hidden_states, max_tokens,
        prompt_cache=None, use_eminf=False, request=None
):
    """Wrap the synchronous generate_step as an async generator."""
    sampler = make_sampler(temp, top_p)

    # Determine if we're using a mlx-community model
    is_mlx_community = hasattr(model, '_loaded_with_mlx_lm') or 'mlx_community' in str(model.__class__)

    if is_mlx_community and HAVE_MLX_LM:
        # Check if EMINF is requested for mlx-community models
        if use_eminf:
            logger.warning(
                "EMINF optimization is not supported for mlx-community models, falling back to standard generation")

        # Use mlx_lm's generate_step for mlx-community models
        for token_data in mlx_lm.generate.generate_step(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                sampler=sampler,
        ):
            token, logprobs = token_data
            if token in tokenizer.eos_token_ids:
                break

            yield (token, logprobs, None)  # mlx_lm doesn't support hidden_states currently
            await asyncio.sleep(0)
    else:
        if use_eminf:
            # Use EMINF optimization
            logger.info("Using EMINF optimization for generation")

            async for token_data in async_eminf_generate_step(
                    prompt=prompt,
                    model=model,
                    tokenizer=tokenizer,
                    temp=temp,
                    top_p=top_p,
                    with_hidden_states=with_hidden_states,
                    max_tokens=max_tokens,
                    prompt_cache=prompt_cache,
                    request=request,
                    cache_obj=getattr(request, '_cache_obj', None)
            ):
                yield token_data
        else:
            # Use original generate_step for GreenBitAI models
            for token, logprobs, hidden_states in generate_step(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    with_hidden_states=with_hidden_states,
                    prompt_cache=prompt_cache,
            ):
                if token in tokenizer.eos_token_ids:
                    break

                yield (token, logprobs, hidden_states)
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
        async for (token, _, hidden_states) in async_generate_step(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            temp=request.temperature,
            top_p=request.top_p,
            with_hidden_states=request.with_hidden_states,
            max_tokens=request.max_tokens,
            use_eminf=request.use_eminf
        ):
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
                model_key = get_model_key(request.model)
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
                    "finish_reason": "max_tokens" if len(tokens) == request.max_tokens else "stop",
                    "hidden_states": serializable_hidden_states,
                    "confidence_score": score
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(tokens),
                "total_tokens": len(prompt) + len(tokens),
                "prompt_tokens_details": {
                    "cached_tokens": 0  # completions doesn't support cache yet.
                }
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

    tokens_to_process, prompt_cache, cache_hit, cache_obj, original_prompt_len = handle_prompt_cache(
        request, model, tokenizer, prompt
    )

    try:
        if request.use_eminf:
            request._cache_obj = cache_obj  # Temporarily store cache_obj in the request object

        async for (token, _, hidden_states) in async_generate_step(
            prompt=tokens_to_process,
            model=model,
            tokenizer=tokenizer,
            temp=request.temperature,
            top_p=request.top_p,
            with_hidden_states=request.with_hidden_states,
            max_tokens=request.max_tokens,
            prompt_cache=prompt_cache,
            use_eminf=request.use_eminf,
            request=request
        ):
            detokenizer.add_token(token)
            tokens.append(token)

            stop_condition = stopping_criteria(tokens, stop_id_sequences, tokenizer.eos_token_id)
            if stop_condition.stop_met:
                break

            await asyncio.sleep(0)

        detokenizer.finalize()
        text = detokenizer.text

        # update prompt cache state
        if request.prompt_cache_key and cache_hit and cache_obj:
            try:
                updated_messages = request.messages + [{"role": "assistant", "content": text}]
                cache_obj.update_after_step(updated_messages, tokenizer)
            except Exception as e:
                logger.warning(f"Failed to update prompt cache: {e}")

        score = None
        if request.with_hidden_states and hidden_states is not None:
            # averaging the hidden states of the prompt tokens for reducing the data transfer overhead.
            avg_hs = hidden_states.mean(axis=1)
            if request.remote_score:
                model_key = get_model_key(request.model)
                scorer = _confidence_scorers.get(model_key)
                if scorer:
                    score = await asyncio.to_thread(scorer.calculate_confidence, avg_hs.tolist())
            else:
                all_hidden_states.append(avg_hs)

        # Convert hidden states to regular Python lists before JSON serialization
        serializable_hidden_states = convert_hidden_states_to_list(all_hidden_states)

        # Calculate cache token number
        cached_tokens = calculate_cached_tokens(
            len(tokens_to_process), original_prompt_len, cache_hit
        )

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
                    "finish_reason": "max_tokens" if len(tokens) == request.max_tokens else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": original_prompt_len,
                "completion_tokens": len(tokens),
                "total_tokens": original_prompt_len + len(tokens),
                "prompt_tokens_details": {
                    "cached_tokens": cached_tokens
                }
            },
        }
    except Exception as e:
        logger.error(f"Async chat completion generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    try:
        import uvicorn
        args = parse_args()

        # Check if mlx-lm is available
        if not HAVE_MLX_LM:
            print("WARNING: mlx-lm is not installed. Support for mlx-community models will be disabled.")
            print("To enable mlx-community models, install mlx-lm: pip install mlx-lm")

        app, server_config, logger = create_app(args)
        logger.info(f"Starting server on {server_config.host}:{server_config.port}")
        uvicorn.run(app, host=server_config.host, port=server_config.port)
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()