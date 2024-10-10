from __future__ import annotations

import json
from typing import Any, Callable, Iterator, List, Mapping, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

import mlx.core as mx
from gbx_lm import load, generate
from gbx_lm.generate import load_kv_cache_from_file

DEFAULT_MODEL_ID = "GreenBitAI/Llama-3-8B-layer-mix-bpw-4.0-mlx"


class GBXPipeline(LLM):
    """GBX Pipeline API.

    To use, you should have the ``gbx-lm`` python package installed.

    Example using from_model_id:
        .. code-block:: python

            from gbx_lm.langchain import GBXPipeline
            pipe = GBXPipeline.from_model_id(
                model_id="GreenBitAI/Llama-3-8B-layer-mix-bpw-4.0-mlx",
                pipeline_kwargs={"max_tokens": 10, "temp": 0.7},
            )
    Example passing model and tokenizer in directly:
        .. code-block:: python

            from gbx_lm.langchain import GBXPipeline
            from gbx_lm import load
            model_id="GreenBitAI/Llama-3-8B-layer-mix-bpw-4.0-mlx"
            model, tokenizer = load(model_id)
            pipe = GBXPipeline(model=model, tokenizer=tokenizer)
    """

    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model: Any  #: :meta private:
    """Model."""
    tokenizer: Any  #: :meta private:
    """Tokenizer."""
    tokenizer_config: Optional[dict] = None
    """
        Configuration parameters specifically for the tokenizer.
        Defaults to an empty dictionary.
    """
    adapter_file: Optional[str] = None
    """
        Path to the adapter file. If provided, applies LoRA layers to the model.
        Defaults to None.
    """
    lazy: bool = False
    """
        If False eval the model parameters to make sure they are
        loaded in memory before returning, otherwise they will be loaded
        when needed. Default: ``False``
    """
    pipeline_kwargs: Optional[dict] = None
    """
    Keyword arguments passed to the pipeline. Defaults include:
        - temp (float): Temperature for generation, default is 0.7.
        - max_tokens (int): Maximum tokens to generate, default is 100.
        - verbose (bool): Whether to output verbose logging, default is False.
        - formatter (Optional[Callable]): A callable to format the output.
          Default is None.
        - repetition_penalty (Optional[float]): The penalty factor for
          repeated sequences, default is None.
        - repetition_context_size (Optional[int]): Size of the context
          for applying repetition penalty, default is None.
        - top_p (float): The cumulative probability threshold for
          top-p filtering, default is 1.0.

    """
    max_kv_size: Optional[int] = None
    """Maximum size of the key-value cache"""
    cache_history: Optional[List[Tuple[mx.array, mx.array]]] = None
    """cache history of pre-saved KV cache"""

    class Config:
        extra = "forbid"

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        tokenizer_config: Optional[dict] = None,
        adapter_file: Optional[str] = None,
        lazy: bool = False,
        pipeline_kwargs: Optional[dict] = None,
        kv_cache_file: Optional[str] = None, # A file containing saved KV caches to avoid recomputing them
        **kwargs: Any,
    ) -> GBXPipeline:
        """Construct the pipeline object from model_id and task."""

        # Load the kv cache and metadata if a kv cache file is provided
        cache_history = None
        metadata = None
        if kv_cache_file is not None:
            cache_history, metadata = load_kv_cache_from_file(kv_cache_file)
        max_kv_size = None
        if cache_history is not None:
            max_kv_size = metadata["max_kv_size"]
            max_kv_size = int(max_kv_size) if max_kv_size.isdigit() else None

        if tokenizer_config is not None:
            tokenizer_config = tokenizer_config
        else:
            tokenizer_config = (
                {} if cache_history is None else json.loads(metadata["tokenizer_config"])
            )

        if adapter_file:
            model, tokenizer = load(model_id, tokenizer_config, adapter_file, lazy)
        else:
            model, tokenizer = load(model_id, tokenizer_config, lazy=lazy)

        if metadata is not None:
            tokenizer.chat_template = metadata["chat_template"]

        _pipeline_kwargs = pipeline_kwargs or {}

        return cls(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            tokenizer_config=tokenizer_config,
            adapter_file=adapter_file,
            lazy=lazy,
            pipeline_kwargs=_pipeline_kwargs,
            max_kv_size=max_kv_size,
            cache_history=cache_history,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "tokenizer_config": self.tokenizer_config,
            "adapter_file": self.adapter_file,
            "lazy": self.lazy,
            "pipeline_kwargs": self.pipeline_kwargs,
            "max_kv_size": self.max_kv_size,
            "cache_history": self.cache_history
        }

    @property
    def _llm_type(self) -> str:
        return "gbx_pipeline"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        pipeline_kwargs = kwargs.get("pipeline_kwargs", self.pipeline_kwargs) or {}

        temp: float = pipeline_kwargs.get("temp", 0.7)
        max_tokens: int = pipeline_kwargs.get("max_tokens", 100)
        verbose: bool = pipeline_kwargs.get("verbose", False)
        formatter: Optional[Callable] = pipeline_kwargs.get("formatter", None)
        repetition_penalty: Optional[float] = pipeline_kwargs.get(
            "repetition_penalty", None
        )
        repetition_context_size: Optional[int] = pipeline_kwargs.get(
            "repetition_context_size", None
        )
        top_p: float = pipeline_kwargs.get("top_p", 1.0)
        # Treat the prompt as a suffix assuming that the prefix is in the
        # stored kv cache.
        if self.cache_history is not None:
            test_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "<query>"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = prompt[test_prompt.index("<query>") :]

        return generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            temp=temp,
            max_tokens=max_tokens,
            verbose=verbose,
            formatter=formatter,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            top_p=top_p,
            max_kv_size=self.max_kv_size,
            cache_history=self.cache_history
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        try:
            from gbx_lm import generate_step

        except ImportError:
            raise ImportError(
                "Could not import gbx_lm python package. "
                "Please install it with `pip install gbx_lm`."
            )

        pipeline_kwargs = kwargs.get("pipeline_kwargs", self.pipeline_kwargs) or {}

        temp: float = pipeline_kwargs.get("temp", 0.7)
        max_new_tokens: int = pipeline_kwargs.get("max_tokens", 100)
        repetition_penalty: Optional[float] = pipeline_kwargs.get(
            "repetition_penalty", None
        )
        repetition_context_size: Optional[int] = pipeline_kwargs.get(
            "repetition_context_size", None
        )
        top_p: float = pipeline_kwargs.get("top_p", 1.0)
        # Treat the prompt as a suffix assuming that the prefix is in the
        # stored kv cache.
        if self.cache_history is not None:
            test_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "<query>"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = prompt[test_prompt.index("<query>") :]

        prompt = self.tokenizer.encode(prompt, return_tensors="np")

        prompt_tokens = mx.array(prompt[0])

        eos_token_id = self.tokenizer.eos_token_id
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        for (token, prob, _), n in zip(
            generate_step(
                prompt=prompt_tokens,
                model=self.model,
                temp=temp,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                top_p=top_p,
                max_kv_size=self.max_kv_size,
                cache_history=self.cache_history
            ),
            range(max_new_tokens),
        ):
            # identify text to yield
            text: Optional[str] = None
            detokenizer.add_token(token)
            detokenizer.finalize()
            text = detokenizer.last_segment

            # yield text, if any
            if text:
                chunk = GenerationChunk(text=text)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)

            # break if stop sequence found
            if token == eos_token_id or (stop is not None and text in stop):
                break