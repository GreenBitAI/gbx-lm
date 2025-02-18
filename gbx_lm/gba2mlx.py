# Initial code base from https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm under the MIT License.
# Additional code from GreenBitAI is licensed under the Apache 2.0 License.
import argparse
import copy
from shutil import copy as file_copy
from pathlib import Path

import mlx.nn as nn
from mlx.utils import tree_flatten

from .utils import extract_bits_and_group_size, save_config, fetch_from_hub, save_weights, upload_to_hub


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
