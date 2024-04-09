import glob
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import time

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoConfig

# Local imports
from .models import qllama, qmixtral, qgemma, qqwen2
from .models.quantized_linear_gba import QuantizedLinear
import re

# Constants
MODEL_MAPPING = {
    "llama": qllama,
    "mistral": qllama,  # mistral is compatible with llama
    "mixtral": qmixtral,
    "gemma": qgemma,
    "qwen2": qqwen2
}

linear_class_predicate = (
    lambda m: isinstance(m, nn.Linear)
    and m.weight.shape[0]
    != 8  # avoid quantizing gate layers, otherwise we have to re-quant and upload all the mixtral models
)


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


def get_model_path(path_or_hf_repo: str, token=None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
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
    return model_path


def top_p_sampling(logits: mx.array, top_p: float, temperature: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: The logits from the model's output.
        top_p: The cumulative probability threshold for top-p filtering.
        temperature: Temperature parameter for softmax distribution reshaping.
    Returns:
        token selected based on the top-p criterion.
    """
    if (
        logits.dtype == mx.bfloat16
    ):  # workaround for unable to load kernel contiguous_scan_inclusive_sum_bfloat16_bfloat16
        logits = logits.astype(mx.float32)

    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    probs = mx.softmax(logits / temperature, axis=-1)

    # sort probs in ascending order
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = probs[..., sorted_indices.squeeze(0)]

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # select tokens with cumulative probs below threshold
    top_probs = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_probs,
        mx.zeros_like(sorted_probs),
    )

    sorted_token = mx.random.categorical(mx.log(top_probs))
    token = sorted_indices.squeeze(0)[sorted_token]

    return token


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


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    temp: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling, if 0 the argmax is used.
        repetition_penalty (float, optional): The penalty factor for repeating tokens.
        repetition_context_size (int, optional): The number of tokens to consider for repetition penalty (default 20).
        top_p (float, optional): Nulceus sampling, higher means model considers more less likely words

    Yields:
        Generator[Tuple[mx.array, mx.array]]: A generator producing
        one token and probability per call.
    """

    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        softmax_logits = mx.softmax(logits)

        if temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            if top_p > 0 and top_p < 1.0:
                token = top_p_sampling(logits, top_p, temp)
            else:
                token = mx.random.categorical(logits * (1 / temp))

        prob = softmax_logits[0, token]
        return token, prob

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )

    y = prompt
    cache = None

    repetition_context = prompt.tolist()

    if repetition_context_size:
        repetition_context = repetition_context[-repetition_context_size:]

    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]

        if repetition_penalty:
            logits = apply_repetition_penalty(
                logits, repetition_context, repetition_penalty
            )
            y, prob = sample(logits)
            repetition_context.append(y.item())
        else:
            y, prob = sample(logits)

        if repetition_context_size:
            if len(repetition_context) > repetition_context_size:
                repetition_context = repetition_context[-repetition_context_size:]
        yield y, prob


def generate(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    temp: float = 0.6,
    max_tokens: int = 100,
    verbose: bool = False,
    formatter: Optional[Callable] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    top_p: float = 1.0,
) -> str:
    """
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temp (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       verbose (bool): If ``True``, print tokens and timing information
           (default ``False``).
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
       repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.
    """

    if verbose:
        print("=" * 10)
        print("Prompt:", prompt)

    prompt_tokens = mx.array(tokenizer.encode(prompt))

    tic = time.perf_counter()
    tokens = []
    token_strings = []
    skip = 0
    REPLACEMENT_CHAR = "\ufffd"

    for (token, prob), n in zip(
        generate_step(
            prompt_tokens,
            model,
            temp,
            repetition_penalty,
            repetition_context_size,
            top_p,
        ),
        range(max_tokens),
    ):
        token = token.item()
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()
        if token == tokenizer.eos_token_id:
            break
        tokens.append(token)

        if verbose:
            s = tokenizer.decode(tokens)
            if formatter:
                formatter(s[skip:], prob.item())
                skip = len(s)
            elif s[-1] != REPLACEMENT_CHAR:
                print(s[skip:], end="", flush=True)
                skip = len(s)
            # Reset token cache at line break
            if s[-1] == "\n":
                tokens = []
                token_strings.append(s)
                skip = 0

    token_count = n + 1
    token_strings.append(tokenizer.decode(tokens).replace(REPLACEMENT_CHAR, ""))

    if verbose:
        print(token_strings[-1][skip:], flush=True)
        gen_time = time.perf_counter() - tic
        print("=" * 10)
        if token_count == 0:
            print("No tokens generated for this prompt")
            return
        prompt_tps = prompt_tokens.size / prompt_time
        gen_tps = (token_count - 1) / gen_time
        print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
        print(f"Generation: {gen_tps:.3f} tokens-per-sec")

    return "".join(token_strings)


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


def load_model(model_path: Path, bits: int=4, group_size: int=64, is_conversion: bool = False, lazy: bool = False) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        bits (int): bits for quantization
        group_size (int): group size used in quantization
        is_conversion (bool): if it is for conversion

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

    weight_files = glob.glob(str(model_path / "*.safetensors"))
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

    model_class, model_args_class = _get_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if  hasattr(model, "sanitize"):
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
        # zeros -> -zeros
        QuantizedLinear.post_processing_and_release(
            model
        )

    if not lazy:
        mx.eval(model.parameters())

    mx.eval()
    return model


def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    adapter_file: Optional[str] = None,
    lazy: bool = False,
) -> Tuple[nn.Module, PreTrainedTokenizer]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        adapter_file (str, optional): Path to the adapter file. If provided, applies LoRA layers to the model.
            Defaults to None.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
    Returns:
        Tuple[nn.Module, PreTrainedTokenizer]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    model = load_model(model_path, lazy=lazy)
    if adapter_file is not None:
        raise NotImplementedError('Error: LoRA layers not implemented yet.')

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_config)
    return model, tokenizer
