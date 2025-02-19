# Copyright © 2023-2024 Apple Inc.
# Additional code from GreenBitAI is licensed under the Apache 2.0 License.

from typing import List, NamedTuple, Optional, Sequence, Union

class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int


def stopping_criteria(
    tokens: List[int],
    stop_id_sequences: List[List[int]],
    eos_token_id: Union[int, None] = None,
) -> StopCondition:
    """
    Determines whether the token generation should stop based on predefined
    conditions.

    Args:
        tokens (List[int]): The current sequence of generated tokens.
        stop_id_sequences (List[List[[int]]): A list of integer lists, each
          representing a sequence of token IDs. If the end of the `tokens`
          list matches any of these sequences, the generation should stop.
        eos_token_id (Union[int, None]): The token ID that represents the
        eos_token_id (Union[int, None]): The token ID that represents the
          end-of-sequence. If the last token in `tokens` matches this, the
          generation should stop.

    Returns:
        StopCondition: A named tuple indicating whether the stop condition has
          been met (`stop_met`) and how many tokens should be trimmed from the
          end if it has (`trim_length`).
    """
    if tokens and tokens[-1] == eos_token_id:
        return StopCondition(stop_met=True, trim_length=1)

    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if tokens[-len(stop_ids) :] == stop_ids:
                return StopCondition(stop_met=True, trim_length=len(stop_ids))

    return StopCondition(stop_met=False, trim_length=0)


def sequence_overlap(s1: Sequence, s2: Sequence) -> bool:
    """
    Checks if a suffix of s1 has overlap with a prefix of s2

    Args:
        s1 (Sequence): The first sequence
        s2 (Sequence): The second sequence

    Returns:
        bool: If the two sequences have overlap
    """
    max_overlap = min(len(s1), len(s2))
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))


def convert_chat(messages: List[dict], role_mapping: Optional[dict] = None):
    default_role_mapping = {
        "system_prompt": (
            "A chat between a curious user and an artificial intelligence "
            "assistant. The assistant follows the given rules no matter what."
        ),
        "system": "ASSISTANT's RULE: ",
        "user": "USER: ",
        "assistant": "ASSISTANT: ",
        "stop": "\n",
    }
    role_mapping = role_mapping if role_mapping is not None else default_role_mapping

    prompt = ""
    for line in messages:
        role_prefix = role_mapping.get(line["role"], "")
        stop = role_mapping.get("stop", "")
        content = line.get("content", "")
        prompt += f"{role_prefix}{content}{stop}"

    prompt += role_mapping.get("assistant", "")
    return prompt.rstrip()


def convert_model_name_to_url_path(model_name: str) -> str:
    """
    Convert a model name to a URL-safe path segment.
    Examples:
        "GreenBitAI/Llama-3-8B-instruct-layer-mix-bpw-4.0-mlx" ->
        "GreenBitAI-Llama-3-8B-instruct-layer-mix-bpw-4.0-mlx"

    Args:
        model_name: Original model name/path

    Returns:
        URL-safe version of the model name
    """
    # Replace forward slashes with dashes
    url_safe_name = model_name.replace("/", "-")

    # Remove any special characters that might cause issues in URLs
    # Keep alphanumeric characters, dashes, and underscores
    url_safe_name = "".join(c for c in url_safe_name
                            if c.isalnum() or c in "-_")

    # Remove any repeated dashes
    while "--" in url_safe_name:
        url_safe_name = url_safe_name.replace("--", "-")

    # Remove leading or trailing dashes
    url_safe_name = url_safe_name.strip("-")

    return url_safe_name


def get_model_endpoint_path(model_name: str, endpoint_type: str) -> str:
    """
    Generate the full API endpoint path for a given model and endpoint type.

    Args:
        model_name: The name of the model
        endpoint_type: Either "completions" or "chat/completions"

    Returns:
        Full API endpoint path
    """
    safe_name = convert_model_name_to_url_path(model_name)
    return f"/v1/{safe_name}/{endpoint_type}"