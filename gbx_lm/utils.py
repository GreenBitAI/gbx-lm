# Initial code base from https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm under the MIT License.
# Additional code from GreenBitAI is licensed under the Apache 2.0 License.

import glob
import json
import logging
import time
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError
from transformers import PreTrainedTokenizer
import transformers

# Local imports
from .models.base import KVCache, RotatingKVCache
from .sample_utils import categorical_sampling, min_p_sampling, top_p_sampling
from .tokenizer_utils import TokenizerWrapper, load_tokenizer
from .tuner.utils import apply_lora_layers
from .models import qllama, qmixtral, qgemma, qqwen2, qphi3, qstarcoder2
from .models.quantized_linear_gba import QuantizedLinear
import re

# Constants
MODEL_MAPPING = {
    "llama": qllama,
    "mistral": qllama,  # mistral is compatible with llama
    "mixtral": qmixtral,
    "gemma": qgemma,
    "qwen2": qqwen2,
    "phi3": qphi3,
    "starcoder2": qstarcoder2
}

MAX_FILE_SIZE_GB = 5


class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]

    if model_type not in MODEL_MAPPING:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    arch = MODEL_MAPPING[model_type]
    return arch.Model, arch.ModelArgs


def get_model_path(path_or_hf_repo: str, token=None, revision: Optional[str] = None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        try:
            model_path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    revision=revision,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                        "*.txt",
                    ],
                    token=token
                )
            )
        except RepositoryNotFoundError:
            raise ModelNotFoundError(
                f"Model not found for path or HF repo: {path_or_hf_repo}.\n"
                "Please make sure you specified the local path or Hugging Face"
                " repo id correctly.\nIf you are trying to access a private or"
                " gated Hugging Face repo, make sure you are authenticated:\n"
                "https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login"
            ) from None
    return model_path


def apply_repetition_penalty(logits: mx.array, generated_tokens: Any, penalty: float):
    """
    Apply repetition penalty to specific logits based on the given context.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        logits (mx.array): The logits produced by the language model.
        generated_tokens (any): A list of N previous tokens.
        penalty (float): The repetition penalty factor to be applied.

    Returns:
        logits (mx.array): Logits with repetition penalty applied to generated tokens.
    """
    if len(generated_tokens) > 0:
        indices = mx.array([token for token in generated_tokens])
        selected_logits = logits[:, indices]
        selected_logits = mx.where(
            selected_logits < 0, selected_logits * penalty, selected_logits / penalty
        )
        logits[:, indices] = selected_logits
    return logits


def make_kv_caches(
    model: nn.Module, max_kv_size: Optional[int] = None
) -> List[Union[KVCache, RotatingKVCache]]:
    if hasattr(model, "make_cache"):
        return model.make_cache()

    kv_heads = (
        [model.n_kv_heads] * len(model.layers)
        if isinstance(model.n_kv_heads, int)
        else model.n_kv_heads
    )
    if max_kv_size is not None:
        return [
            RotatingKVCache(model.head_dim, n, max_size=max_kv_size, keep=4)
            for n in kv_heads
        ]
    else:
        return [KVCache(model.head_dim, n) for n in kv_heads]


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    temp: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    logit_bias: Optional[Dict[int, float]] = None,
    prefill_step_size: int = 512,
    max_kv_size: Optional[int] = None,
    cache_history: Optional[List[Tuple[mx.array, mx.array]]] = None,
    with_hidden_states: bool = False
) -> Generator[Tuple[int, mx.array, Optional[mx.array]], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): The minimum value (scaled by the top token's
          probability) that a token probability must have to be considered.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
          be filtered by min_p sampling.
        logit_bias (dictionary, optional): Additive logit bias.
        prefill_step_size (int): Step size for processing the prompt.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        with_hidden_states (bool): If ``True``, return hidden states. Default: ``False``.

    Yields:
        Generator[Tuple[mx.array, mx.array], None, None]: A generator producing
          one token and a vector of log probabilities.
    """

    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        if logit_bias:
            indices = mx.array(list(logit_bias.keys()))
            values = mx.array(list(logit_bias.values()))
            logits[:, indices] += values
        logprobs = logits - mx.logsumexp(logits)

        if temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            if top_p > 0 and top_p < 1.0:
                token = top_p_sampling(logits, top_p, temp)
            elif min_p != 0.0:
                token = min_p_sampling(logits, min_p, min_tokens_to_keep, temp)
            else:
                token = categorical_sampling(logits, temp)

        return token, logprobs

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )

    y = prompt

    # Create the KV cache for generation
    cache = make_kv_caches(model, max_kv_size)

    if cache_history is not None:
        if len(cache_history) != len(cache):
            raise ValueError("Wrong number of layers in the cache history")

        # Set the history in the cache objects and evaluate them to prepare for
        # generation.
        for c, h in zip(cache, cache_history):
            c.update_and_fetch(h[0], h[1])
        mx.eval([c.state for c in cache])

    repetition_context = prompt.tolist()

    if repetition_context_size:
        repetition_context = repetition_context[-repetition_context_size:]

    def _step(y, hidden_states=False):
        nonlocal repetition_context
        prompt_hidden_states = None
        if hidden_states:
            logits, prompt_hidden_states = model(y[None], cache=cache, hidden_states=hidden_states)
        else:
            logits = model(y[None], cache=cache)
        logits = logits[:, -1, :]

        if repetition_penalty:
            logits = apply_repetition_penalty(
                logits, repetition_context, repetition_penalty
            )
            y, logprobs = sample(logits)
            repetition_context.append(y.item())
        else:
            y, logprobs = sample(logits)

        if repetition_context_size:
            if len(repetition_context) > repetition_context_size:
                repetition_context = repetition_context[-repetition_context_size:]
        return y, logprobs.squeeze(0), prompt_hidden_states

    while y.size > prefill_step_size:
        model(y[:prefill_step_size][None], cache=cache)
        mx.eval([c.state for c in cache])
        y = y[prefill_step_size:]

    y, logprobs, h_states = _step(y, hidden_states=with_hidden_states)

    mx.async_eval(y)
    while True:
        next_y, next_logprobs, _ = _step(y)
        mx.async_eval(next_y)
        yield y.item(), logprobs, h_states
        y, logprobs = next_y, next_logprobs


def stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 100,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The ma
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        Generator[Tuple[mx.array, mx.array]]: A generator producing text.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    detokenizer.reset()
    for (token,  logprobs, _), n in zip(
        generate_step(prompt_tokens, model, **kwargs),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)

        # Yield the last segment if streaming
        yield detokenizer.last_segment

    detokenizer.finalize()
    yield detokenizer.last_segment


def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 100,
    verbose: bool = False,
    formatter: Optional[Callable] = None,
    with_hidden_states: bool = False,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    Generate a complete response from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       max_tokens (int): The maximum number of tokens. Default: ``100``.
       verbose (bool): If ``True``, print tokens and timing information.
           Default: ``False``.
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       with_hidden_states (bool): If ``True``, return hidden states. Default: ``False``.
       kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if verbose:
        print("=" * 10)
        print("Prompt:", prompt)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    tic = time.perf_counter()
    detokenizer.reset()

    all_hidden_states = [] if with_hidden_states else None

    for (token, logprobs, hidden_states), n in zip(
        generate_step(prompt_tokens, model, with_hidden_states=with_hidden_states, **kwargs),
        range(max_tokens),
    ):
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)

        if verbose:
            if formatter:
                # We have to finalize so that the prob corresponds to the last segment
                detokenizer.finalize()
                formatter(detokenizer.last_segment, mx.exp(logprobs[token]).item())
            else:
                print(detokenizer.last_segment, end="", flush=True)

    if with_hidden_states and hidden_states is not None:
        all_hidden_states.append(hidden_states)

    token_count = n + 1
    detokenizer.finalize()

    if verbose:
        gen_time = time.perf_counter() - tic
        print(detokenizer.last_segment, flush=True)
        print("=" * 10)
        if token_count == 0:
            print("No tokens generated for this prompt")
            return
        prompt_tps = prompt_tokens.size / prompt_time
        gen_tps = (token_count - 1) / gen_time
        print(f"Prompt: {prompt_tokens.size} tokens, {prompt_tps:.3f} tokens-per-sec")
        print(f"Generation: {token_count} tokens, {gen_tps:.3f} tokens-per-sec")
        peak_mem = mx.metal.get_peak_memory() / 2**30
        print(f"Peak memory: {peak_mem:.3f} GB")

    if with_hidden_states:
        return detokenizer.text, all_hidden_states
    else:
        return detokenizer.text


def get_parameter_usage_info(weights):
    """
    Determines whether double quantization and q_perm are used for scales and zeros within the given weights.

    This function iterates through the keys of the weights dictionary, checking for specific
    keys related to quantization statistics, scales, and zeros. If any of these keys are present,
    it indicates that double quantization is applied to either scales or zeros, or both.

    Parameters:
        weights (dict): A dictionary containing model weights and potentially quantization
                      information.

    Returns:
        bool: True if double quantization is used for either scales or zeros, False otherwise.
        bool: True if q_perm is used, False otherwise.
    """
    use_double_quantization = False
    use_q_perm = False
    for k, v in weights.items():
        if 'qstatistic' in k or 'qscales_scales' in k or 'qzeros_scales' in k or 'qscales_zeros' in k or 'qzeros_zeros' in k:
            use_double_quantization = True
        if 'q_perm' in k:
            use_q_perm = True
    return use_double_quantization, use_q_perm


def extract_bits_and_group_size(s):
    """
    Extracts quantization bits and group size from a given string based on predefined patterns.

    This function searches for the patterns that represent 'bits' and 'group size' in the input string.
    The patterns are defined as 'w<number>' for bits and 'g<number>' for group size. If these patterns
    are found, the function extracts and converts them to integers. If not found, the values default to None.

    Parameters:
    - s (str): The input string containing the information about bits and group size.

    Returns:
    - tuple: A tuple containing two elements, (bits, group_size), extracted from the string.
             Each element is an integer if the pattern is found, otherwise None.
    """
    # Regex patterns to find bits and group_size
    bits_pattern = r'w(\d+)'
    group_size_pattern = r'g(\d+)'

    # Search for patterns in the string
    bits_match = re.search(bits_pattern, s)
    group_size_match = re.search(group_size_pattern, s)

    # Extract and convert to integers if found, else default to None
    bits = int(bits_match.group(1)) if bits_match else None
    group_size = int(group_size_match.group(1)) if group_size_match else None

    return bits, group_size


def load_config(model_path: Path) -> dict:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    return config


def load_model(
    model_path: Path,
    lazy: bool = False,
    model_config: dict = {},
    bits: int = 4,
    group_size: int = 64,
    is_conversion: bool = False,
    get_model_classes: Callable[[dict], Tuple[Type[nn.Module], Type]] = _get_classes,
) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        model_config(dict, optional): Configuration parameters for the model.
            Defaults to an empty dictionary.
        bits (int): bits for quantization
        group_size (int): group size used in quantization
        is_conversion (bool): if it is for conversion
        get_model_classes (Callable[[dict], Tuple[Type[nn.Module], Type]], optional):
            A function that returns the model class and model args class given a config.
            Defaults to the _get_classes function.

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """

    # ======== load strategy.json file ========= #
    strategy = None
    try:
        with open(model_path / "quant_strategy.json", "r") as f:
            strategy = json.load(f)["measurement"]
    except FileNotFoundError:
        logging.info(f"[WARNING] Strategy config file not found in {model_path}")

    # ===== load quantization config file ====== #
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
            quantization = config.get("quantization", None)
            if quantization == None:
                quantization = {"group_size": group_size, "bits": bits}
    except FileNotFoundError:
        logging.info(f"[WARNING] Quantization config file not found in {model_path}")
        raise

    config.update(model_config)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))

    if not weight_files:
        # Try weight for back-compat
        weight_files = glob.glob(str(model_path / "weight*.safetensors"))

    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # get if uses double quantization, a technique to reduce the model size
    use_double_quantization, use_q_perm = get_parameter_usage_info(weights)
    if not use_q_perm:
        assert quantization['group_size'] in [32, 64,
                                              128], f"The group size value ({group_size}) must be 32, 64 or 128."
    if is_conversion:
        info_message = "[INFO] This model {} double quantization.".format(
            "USES" if use_double_quantization else "DOES NOT use")
        print(info_message)

    ## ==== needs to do this layout adaption for loading GBA weights ==== ##
    if is_conversion:
        print("[INFO] Transposing qweight, possibly also scales and zeros to meet mlx format ...")
        for k, v in weights.items():
            if 'qweight' in k:
                weights[k] = v.transpose().astype(mx.uint32)
            if not use_double_quantization and ('scales' in k or 'zeros' in k):
                weights[k] = v.transpose()
    ## ===================================================================##

    model_class, model_args_class = get_model_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # update QuantizedLinear layers using quantization parameters.
    QuantizedLinear.reinit_module(
        model,
        **quantization,
        strategy=strategy,
        use_double_quantization = use_double_quantization,
        use_q_perm = use_q_perm
    )

    model.load_weights(list(weights.items()), strict=False)

    # If double quantization used in GBA models, fp16 scales and zeros will be created for supporting mlx format.
    if use_double_quantization:
        QuantizedLinear.prepare_scales_zeros(
            model
        )
    if is_conversion:
        # zeros -> -zeros and release some attributes after loading and adapting params
        QuantizedLinear.post_processing_and_release(
            model
        )

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model


def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    model_config={},
    adapter_path: Optional[str] = None,
    lazy: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    model = load_model(model_path, lazy, model_config)
    if adapter_path is not None:
        model = apply_lora_layers(model, adapter_path)
        model.eval()
    tokenizer = load_tokenizer(model_path, tokenizer_config)

    return model, tokenizer


def fetch_from_hub(
    model_path: str,
    lazy: bool = False,
    token=None,
    group_size=64,
    bits=4,
    is_conversion=False,
) -> Tuple[nn.Module, dict, PreTrainedTokenizer]:
    """
    Fetches a model, its configuration, and tokenizer from the Hugging Face Hub.

    This function downloads a pre-trained model along with its configuration and tokenizer based on the provided
    model path. It also supports quantization parameters to potentially adjust the model loading process for
    conversion purposes.

    Parameters:
    - model_path (str): The path or identifier of the model on the Hugging Face Hub.
    - token: An optional authentication token for private models or higher request rates. Defaults to None.
    - group_size (int): The group size used for quantization, affecting how the model is loaded. Defaults to 64.
    - bits (int): The number of quantization bits, affecting the precision of the loaded model. Defaults to 4.
    - is_conversion (bool): A flag indicating whether the model is being fetched for conversion purposes. Affects model loading. Defaults to False.

    Returns:
    - Tuple containing:
      - A dictionary representing the model ready for use.
      - A dictionary containing the model's configuration.
      - The PreTrainedTokenizer associated with the model.

    The function leverages the Transformers library to obtain the configuration and tokenizer, and assumes the
    existence of a `load_model` function tailored for loading and possibly quantizing the model based on the given
    parameters.
    """

    model_path = get_model_path(model_path, token=token)
    config = transformers.AutoConfig.from_pretrained(model_path, token=token, trust_remote_code=True)
    model = load_model(model_path, lazy=lazy, bits=bits, group_size=group_size, is_conversion=is_conversion)
    tokenizer_config = {"token": token, "trust_remote_code": True}
    tokenizer = load_tokenizer(model_path, tokenizer_config_extra=tokenizer_config)
    return model, config.to_dict(), tokenizer, model_path


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def upload_to_hub(path: str, upload_repo: str, hf_path: str, token: Optional[str] = None):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
        hf_path (str): Path to the original Hugging Face model.
        token: Optional(str): token used for accessing Hugging Face repo.
    """
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    from . import __version__

    card = ModelCard.load(hf_path, token=token)
    card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"]
    card.data.base_model = hf_path
    card.text = dedent(
        f"""
        # {upload_repo}
        
        This quantized low-bit model [{upload_repo}](https://huggingface.co/{upload_repo}) was converted to MLX format from [`{hf_path}`](https://huggingface.co/{hf_path}) using gbx-lm version **{__version__}**.
        Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.
        
        ## Use with mlx
        
        ```bash
        pip install gbx-lm
        ```
        
        ```python
        from gbx_lm import load, generate
        
        model, tokenizer = load("{upload_repo}")
        response = generate(model, tokenizer, prompt="hello", verbose=True)
        ```
        """
    )
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi(token=token)
    api.create_repo(repo_id=upload_repo, exist_ok=True, token=token, private=True, repo_type='model')
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
        token = token,
        multi_commits=True,
        multi_commits_verbose=True,
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def save_weights(
    save_path: Union[str, Path],
    weights: Dict[str, Any],
    *,
    donate_weights: bool = False,
) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    if donate_weights:
        weights.clear()
        del weights

    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # Clean unused keys
    config.pop("_name_or_path", None)

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)