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
        type=str,
        choices=["eminf", "temp", "sled", "sledeminf", "sledtemp", "eminfsled", "tempsled"],
        default=None,
        help="Calibration method to use for generation",
    )
    parser.add_argument(
        "--enable-cache",
        action="store_true",
        help="Enable prompt caching for better performance in multi-turn conversations",
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
    
    # Initialize cache for both calibration and regular generation
    prompt_cache_obj = None
    mlx_cache = None
    
    if args.enable_cache:
        from .prompt_cache import PromptCache
        prompt_cache_obj = PromptCache()
        print("Pre-caching system prompt...")
        prompt_cache_obj.cache_system_prompt(model, args.system_prompt, tokenizer)
    else:
        # Always create MLX cache for regular chat functionality
        mlx_cache = make_prompt_cache(model, args.max_kv_size)
    
    messages = [{"role": "system", "content": args.system_prompt}]

    while True:
        query = input(">> ")
        if query == "q":
            break
        messages.append({"role": "user", "content": query})
        
        if args.calibration_method is not None:
            # For calibration methods, use tokenized input
            input_ids_with_gen = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
            input_ids_no_gen = tokenizer.apply_chat_template(messages, add_generation_prompt=False, enable_thinking=False)
            
            from .utils import calibrated_generate
            response = calibrated_generate(
                model,
                tokenizer,
                input_ids_with_gen,
                method=args.calibration_method,
                verbose=True,
                prompt_cache=prompt_cache_obj,
                use_cache=args.enable_cache,
                system_prompt=args.system_prompt,
                input_ids_no_gen=input_ids_no_gen
            )
            print(response, flush=True, end="")
            
            # Update messages with the response
            messages.append({"role": "assistant", "content": response})
            
            # Update cache after generation
            if args.enable_cache and prompt_cache_obj:
                prompt_cache_obj.update_after_step(messages, tokenizer)
        else:
            # For regular generation with cache optimization
            if args.enable_cache and prompt_cache_obj:
                # Use PromptCache for prefix detection and incremental processing
                input_ids_with_gen = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
                input_ids_no_gen = tokenizer.apply_chat_template(messages, add_generation_prompt=False, enable_thinking=False)
                
                # Use get_input_ids for cache optimization
                from .calibration import get_input_ids
                model_key = getattr(model, "model_key", id(model))
                optimized_tokens, cache_hit, cache_to_use = get_input_ids(
                    prompt_cache_obj, model, input_ids_with_gen, input_ids_no_gen, model_key, use_cache=True
                )
                
                response_text = ""
                for response in stream_generate(
                    model,
                    tokenizer,
                    optimized_tokens,
                    max_tokens=args.max_tokens,
                    sampler=make_sampler(args.temp, args.top_p),
                    prompt_cache=cache_to_use,
                ):
                    print(response.text, flush=True, end="")
                    response_text += response.text
            else:
                # Regular generation without cache optimization
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                
                response_text = ""
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=args.max_tokens,
                    sampler=make_sampler(args.temp, args.top_p),
                    prompt_cache=mlx_cache,
                ):
                    print(response.text, flush=True, end="")
                    response_text += response.text
            
            # Update messages with the response
            messages.append({"role": "assistant", "content": response_text})
            
            # Update cache after generation
            if args.enable_cache and prompt_cache_obj:
                prompt_cache_obj.update_after_step(messages, tokenizer)
        
        print()


if __name__ == "__main__":
    main()
