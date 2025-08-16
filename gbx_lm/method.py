import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from gbx_lm.models.cache import make_prompt_cache


def get_input_ids(prompt_cache, model, ids_with_gen, ids_no_gen, model_key, use_cache=True):
    """
    Helper function to apply prompt cache to input_ids for generation methods.
    Returns: (new_input_ids, use_cache, cache)
    Args:
        use_cache: Whether to use prompt caching (independent switch)
    """
    if not use_cache or prompt_cache is None:
        cache = make_prompt_cache(model)
        return ids_with_gen, False, cache
        
    new_tokens, cache, cache_hit = prompt_cache.get_prompt_cache(model, ids_with_gen, ids_no_gen, model_key)
    
    if cache_hit:
        print(f"Use input cache 🐈 Processing {len(new_tokens)} tokens instead of {len(ids_with_gen)}")
        return new_tokens if len(new_tokens) > 0 else ids_with_gen, True, cache
    else:
        print(f"No cache benefit - processing all tokens")
        return ids_with_gen, False, cache

def eminf_optimize(logits, alpha=0.65, num_steps=None, threshold=0.05):
    def entropy_calculation(x):
        probs = mx.softmax(x, axis=-1)
        return -mx.sum(probs * mx.log(probs + 1e-10))
    def step_allocation(H_init, H_target, min_step = 3, max_step = 15):
        r = float(mx.maximum(0.0, H_init - H_target)/H_target)
        s = r/(1.0+r)
        steps = int(min_step+max_step *s)   
        return steps
    current_logits = logits.astype(mx.float32)
    initial_alpha = alpha
    initial_entropy = entropy_calculation(current_logits)
    best_entropy = initial_entropy
    best_logits = current_logits
    inner_iterations = 0
    target_entropy = mx.maximum(threshold, 0.1 * initial_entropy)
    if num_steps is None:
        num_steps = step_allocation(initial_entropy, threshold)
    inner_start = time.perf_counter()
    for _ in range(num_steps):
        inner_iterations += 1
        current_entropy = entropy_calculation(current_logits)
        if float(current_entropy) < float(target_entropy):
            break
        current_logp = nn.log_softmax(current_logits, axis = -1)
        current_p = mx.exp(current_logp)
        gradient = current_p * (mx.sum(current_p * (current_logp+1.0)) - (current_logp+1.0))
        gradient_norm = mx.sqrt(mx.sum(gradient ** 2))
        
        if float(gradient_norm) > 0:
            gradient = gradient / gradient_norm
            new_logits = current_logits - initial_alpha * gradient
            new_entropy = entropy_calculation(new_logits)
            if float(new_entropy) < float(current_entropy):
                current_logits = new_logits
                if float(new_entropy) < float(best_entropy):
                    best_logits = new_logits
                    best_entropy = new_entropy
            else:
                initial_alpha *= 0.65
                if initial_alpha < 1e-4:
                    break
        else:
            break
    return best_logits, best_entropy

def eminf_generate(model, tokenizer, input_ids, input_ids_no_gen, max_tokens, num_steps = None, alpha = 0.65, threshold = 0.05, max_kv_size = None, prompt_cache=None, use_cache=True):
    # Get input IDs with cache optimization
    model_key = getattr(model, "model_key", id(model))
    cached_input_ids, cache_hit, cache = get_input_ids(prompt_cache, model, input_ids, input_ids_no_gen, model_key, use_cache)
    generated_ids = list(cached_input_ids)
    original_length = len(cached_input_ids)
    output = model(mx.array([cached_input_ids]), cache = cache, hidden_states = False)
    mx.eval([c.state for c in cache])
    mx.clear_cache()

    output = mx.stop_gradient(output)
    logits = output[:, -1, :].astype(mx.float32)

    for i in range(max_tokens):
        try:
            response = ""
            best_logits, best_entropy = eminf_optimize(logits, alpha = alpha, num_steps = num_steps, threshold = threshold)
            final_probs = nn.softmax(best_logits, axis = -1)
            probs_np = np.array(final_probs).flatten()
            probs_np = probs_np / probs_np.sum()
            next_token = np.random.choice(len(probs_np), p=probs_np)
            generated_ids.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break
            new_tokens = mx.array([[next_token]])
            response += tokenizer.decode([next_token])
            print(response, flush=True, end="")
            output2 = model(new_tokens, cache = cache, hidden_states = False)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
            output2 = mx.stop_gradient(output2)
            logits = output2[:, -1, :].astype(mx.float32)
        except Exception as e:
            print(f"EMINF error at step {i}: {e}")
            break
    return generated_ids[original_length:]

def generate_response(model, tokenizer, messages, model_name, max_tokens, prompt_cache=None, use_cache=True):
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking = False)
    input_ids_no_gen = tokenizer.apply_chat_template(messages, add_generation_prompt=False, enable_thinking = False)
        
    generated_ids = eminf_generate(model, tokenizer, input_ids, input_ids_no_gen, max_tokens=max_tokens, prompt_cache=prompt_cache, use_cache=use_cache)
    response = tokenizer.decode(np.array(generated_ids), skip_special_tokens=True)
    response = response.strip()
    return response