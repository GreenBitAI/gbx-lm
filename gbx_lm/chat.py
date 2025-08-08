# Copyright © 2023-2024 Apple Inc.
# Additional code from GreenBitAI is licensed under the Apache 2.0 License.

import argparse

import mlx.core as mx

from .models.cache import make_prompt_cache
from .sample_utils import make_sampler
from .utils import load, stream_generate

DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "GreenBitAI/Llama-3.2-3B-Instruct-layer-mix-bpw-4.0-mlx"
DEFAULT_SYSTEM_PROMPT = "You are Libra, a helpful and friendly AI assistant. You aim to provide clear and useful responses to help users with their questions and tasks."


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="PRNG seed")
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--calibration-method",
        type = str,
        choices = ["eminf", "temp", "sled", "sledeminf", "sledtemp", "eminfsled", "tempsled"],
        default = None,
        help = "Calibration method to use for the model"
    )
    parser.add_argument(
        "--evolution-rate",
        type=float,
        default=1.2,
        help="Evolution rate for SLED-based methods",
    )
    parser.add_argument(
        "--evolution-scale",
        type=int,
        default=10,
        help="Evolution scale (top-k) for SLED-based methods",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Alpha parameter for entropy-based methods. Recommended 0.65 for eminf and 0.5 for ATS",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Delta parameter for temperature scaling methods",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Threshold parameter for entropy minimization",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load(
        args.model,
        adapter_path=args.adapter_path,
        tokenizer_config={"trust_remote_code": True},
    )

    print(f"[INFO] Starting chat session with {args.model}. To exit, enter 'q'.")
    prompt_cache = make_prompt_cache(model, args.max_kv_size)
    messages = [{"role": "system", "content": args.system_prompt}]

    while True:
        query = input(">> ")
        if query == "q":
            break
        messages.append({"role": "user", "content": query})
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if args.calibration_method is not None:
            method_kwargs = {
                "max_new_tokens": args.max_tokens,
                "temperature": args.temp,
                "top_p": args.top_p,
            }
            if args.calibration_method in ["sled", "sledeminf", "sledtemp", "eminfsled", "tempsled"]:
                method_kwargs.update({
                    "evolution_rate": args.evolution_rate,
                    "evolution_scale": args.evolution_scale,
                })
            if args.calibration_method in ["eminf", "sledeminf", "eminfsled"]:
                method_kwargs.update({
                    "alpha": args.alpha,
                    "threshold": args.threshold,
                })
            
            if args.calibration_method in ["temp", "sledtemp", "tempsled"]:
                method_kwargs.update({
                    "alpha": args.alpha,
                    "delta": args.delta,
                })

            from .utils import calibrated_generate
            response = calibrated_generate(
                model,
                tokenizer,
                prompt,
                method = args.calibration_method,
                verbose = args.verbose,
                **method_kwargs
            )
            print(response.text, flush=True, end="")
        else:
            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                sampler=make_sampler(args.temp, args.top_p),
                prompt_cache=prompt_cache,
            ):
                print(response.text, flush=True, end="")
        print()


if __name__ == "__main__":
    main()
