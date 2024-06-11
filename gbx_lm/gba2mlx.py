import argparse
import copy
from shutil import copy as file_copy
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Any

import mlx.core as mx
import mlx.nn as nn
import transformers
from mlx.utils import tree_flatten

from .utils import get_model_path, load_model, extract_bits_and_group_size
from .tokenizer_utils import load_tokenizer

MAX_FILE_SIZE_GB = 5


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )

    parser.add_argument("--hf-path", type=str, help="Path to the Hugging Face model.")
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "--q-group-size", help="Group size used in the quantized model", type=int, default=256
    )
    parser.add_argument(
        "--q-bits", help="Bits per weight for quantization.", type=int, default=4
    )
    parser.add_argument(
        "--hf-token", type=str, default="", help="hf token"
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    return parser


def fetch_from_hub(
    model_path: str,
    lazy: bool = False,
    token=None,
    group_size=64,
    bits=4,
    is_conversion=False,
) -> Tuple[nn.Module, dict, transformers.PreTrainedTokenizer]:
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


def get_quantized_parameters(
    model: nn.Module, config: dict, q_group_size: int, q_bits: int
) -> tuple:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The quantized model.
        config (dict): Model configuration.
        q_group_size (int): Group size for quantization.
        q_bits (int): Bits per weight for quantization.

    Returns:
        tuple: Tuple containing quantized weights and config.
    """
    quantized_config = copy.deepcopy(config)
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
    quantized_weights = dict(tree_flatten(model.parameters()))
    return quantized_weights, quantized_config


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
    card.text = f"""
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
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi(token=token)
    api.create_repo(repo_id=upload_repo, exist_ok=True, token=token, private=True, repo_type='model')
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
        token = token
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def check_upload_repo_string(repo_string: str):
    """
    Verify the repopsitory name for model upload.

    Args:
        repo_string (str): a valid Hugging Face repo name.
    """
    # Define an example that meets the requirements
    example = "ExampleHFRepoName/example-folder-name-mlx"

    # Check if the string contains "/"
    if "/" not in repo_string:
        return False, f"Upload repo string must contain a '/' separating the HF repo name and the folder name. For example, '{example}'."

    # Split the string into HF repo name and folder name
    hf_repo_name, folder_name = repo_string.split("/", 1)

    # Check if the folder name ends with "-mlx"
    if not folder_name.endswith("-mlx"):
        return False, f"The folder name must end with '-mlx'. For example, '{example}'."

    # If all checks pass, return True and an empty string
    return True, ""


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


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    q_group_size: int = 256,
    q_bits: int = 4,
    upload_repo: str = None,
    hf_token: str = None
):
    """
    Converts a model from GreenBitAI's HuggingFace hub to a quantized version for MLX platform.

    Parameters:
        - hf_path (str): The path or identifier to the Hugging Face model hub.
        - mlx_path (str): The directory path where the quantized model will be saved. Defaults to 'mlx_model'.
        - q_group_size (int): The group size for quantization. Defaults to 256.
        - q_bits (int): The number of bits for quantization. Defaults to 4.
        - upload_repo: The Hugging Face repo to upload the model to.
    Returns:
    - None

    Note: The function assumes the availability of certain utility functions such as `extract_bits_and_group_size`,
    `fetch_from_hub`, `get_quantized_parameters`, and `make_shards` to perform its tasks. It also uses
    'mx.save_safetensors' for saving quantized weights which implies a dependency on a specific library (mxnet).
    """
    print("[INFO] Loading ...")
    is_conversion = True
    q_bits, q_group_size = extract_bits_and_group_size(hf_path)
    model, config, tokenizer, model_path = fetch_from_hub(hf_path, token=hf_token, group_size=q_group_size, bits=q_bits, is_conversion=is_conversion)
    weights, config = get_quantized_parameters(model, config, q_group_size, q_bits)

    print("[INFO] Creating dir '{}' for saving ...".format(mlx_path))
    mlx_path = Path(mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Saving quantized weights ...")
    del model
    save_weights(mlx_path, weights, donate_weights=True)

    tokenizer.save_pretrained(mlx_path)

    print("[INFO] Saving config file ...")
    save_config(config, config_path=mlx_path / "config.json")

    print("[INFO] Saving quant strategy config file ...")
    try:
        # check if file exists
        with open(model_path / "quant_strategy.json", "r") as f:
            file_copy(model_path / "quant_strategy.json", mlx_path / "quant_strategy.json")
            print(f"[INFO] Strategy config file copied to {mlx_path}")
    except FileNotFoundError:
        print(f"[INFO] Strategy config file not found in {model_path}")

    print(f"[INFO] Converted model files have been saved in {mlx_path.absolute()}/")
    for file in mlx_path.iterdir():
        if file.is_file():
            print("[INFO] |----" + file.name)

    if upload_repo is not None:
        print("[INFO] Upload saved model to {}.".format(upload_repo))
        try:
            is_valid, message = check_upload_repo_string(upload_repo)
            if is_valid:
                upload_to_hub(mlx_path, upload_repo, hf_path, token=hf_token)
                print("[INFO] Model updload finished!")
            else:
                print(f"[Error] {message}")
        except Exception as e:
            # Handle any other exceptions that might occur
            print(f"[Error] An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))
