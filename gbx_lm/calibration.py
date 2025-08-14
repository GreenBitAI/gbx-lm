import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from .models.cache import make_prompt_cache
from .sample_utils import make_sampler, make_logits_processors

decoding_config = {
    "eminf": {
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 2048,
        "use_eminf": True,
        "use_temp_scaling": False,
        "use_sled": False,
        "use_sledeminf": False,
        "use_sledtemp": False,
        "use_eminfsled": False,
        "use_tempsled": False
    },
    "temp": {
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 2048,
        "use_eminf": False,
        "use_temp_scaling": True,
        "use_sled": False,
        "use_sledeminf": False,
        "use_sledtemp": False,
        "use_eminfsled": False,
        "use_tempsled": False,
        "alpha": 0.6,
        "delta": 0.05
    },
    "mlx": {
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 2048,
        "use_eminf": False,
        "use_temp_scaling": False,
        "use_sled": False,
        "use_sledeminf": False,
        "use_sledtemp": False,
        "use_eminfsled": False,
        "use_tempsled": False
    },
    "sled": {
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 2048,
        "use_eminf": False,
        "use_temp_scaling": False,
        "use_sled": True,
        "evolution_rate": 1.5,
        "evolution_scale": 15,
        "evolution_lower_bound": -100.0,
        "use_sledeminf": False,
        "use_sledtemp": False,
        "use_eminfsled": False,
        "use_tempsled": False
    },
    "sledeminf": {
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 2048,
        "use_eminf": False,
        "use_temp_scaling": False,
        "use_sled": False,
        "use_sledeminf": True,
        "use_sledtemp": False,
        "use_eminfsled": False,
        "use_tempsled": False,
        "evolution_rate": 1.2,
        "evolution_scale": 10,
        "evolution_lower_bound": -100.0
    },
    "sledtemp":{
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 2048,
        "use_eminf": False,
        "use_temp_scaling": False,
        "use_sled": False,
        "use_sledeminf": False,
        "use_sledtemp": True,
        "use_eminfsled": False,
        "use_tempsled": False,
        "alpha": 0.5,
        "delta": 0.05
    },
    "eminfsled": {
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 2048,
        "use_eminf": False,
        "use_temp_scaling": False,
        "use_sled": False,
        "use_sledeminf": False,
        "use_sledtemp": False,
        "use_eminfsled": True,
        "use_tempsled": False,
        "alpha": 0.65,
        "threshold": 0.05,
        "num_steps": None,
        "evolution_rate": 1.2,
        "evolution_scale": 10,
        "evolution_lower_bound": -100.0
    },
    "tempsled": {
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 2048,
        "use_eminf": False,
        "use_temp_scaling": False,
        "use_sled": False,
        "use_sledeminf": False,
        "use_sledtemp": False,
        "use_eminfsled":False,
        "use_tempsled": True,
        "alpha": 0.6,
        "delta": 0.05
    }
}
def get_input_ids(prompt_cache, model, ids_with_gen, ids_no_gen, model_key, use_cache=True):
    """
    Helper function to apply prompt cache to input_ids for generation methods.
    Returns: (new_input_ids, use_cache, cache)
    Args:
        use_cache: Whether to use prompt caching (independent switch)
    """
    if not use_cache or prompt_cache is None:
        cache = make_prompt_cache(model)
        print(f"Create new cache (caching {'disabled' if not use_cache else 'not available'})")
        return ids_with_gen, False, cache
        
    new_tokens, cache, cache_hit = prompt_cache.get_prompt_cache(model, ids_with_gen, ids_no_gen, model_key)
    
    if cache_hit:
        print(f"Use input cache 🐈 Processing {len(new_tokens)} tokens instead of {len(ids_with_gen)}")
        return new_tokens if len(new_tokens) > 0 else ids_with_gen, True, cache
    else:
        print(f"No cache benefit - processing all tokens")
        return ids_with_gen, False, cache

def sledeminf_generate(model, tokenizer, input_ids, input_ids_no_gen, max_token = 2048, num_steps = None, alpha = 0.65, threshold = 0.05, max_kv_size = None, bench = None, prompt_cache=None, use_cache=True):
    # Get input IDs with cache optimization
    model_key = getattr(model, "model_key", id(model))
    cached_input_ids, cache_hit, cache = get_input_ids(prompt_cache, model, input_ids, input_ids_no_gen, model_key, use_cache)
    
    def entropy_calculation(x):
        probs = nn.softmax(x, axis = -1)
        return -mx.sum(probs * mx.log(probs + 1e-10))
    def step_allocation(H_init, H_target, min_step = 3, max_step = 15):
        r = float(mx.maximum(0.0, H_init - H_target)/H_target)
        s = r/(1.0+r)
        steps = int(min_step+max_step *s)   
        return steps
    generated_ids = list(cached_input_ids)
    original_length = len(cached_input_ids)
    forward_start = time.perf_counter()
    input_array = mx.array([cached_input_ids])
    final_states, all_hidden_states = model(input_array, cache = cache, hidden_states = True)
    mx.eval([c.state for c in cache])
    mx.clear_cache()
    forward_time = time.perf_counter() - forward_start
    if bench:
        bench.set_forward_time(forward_time)
    total_inner_steps = 0
    for i in range(max_token):
        if i % 1000 == 0:
            print(f"Sledeminf step {i+1}/{max_token}")
        try:
            evolved_logits = sled_logits_from_hidden(model, all_hidden_states = all_hidden_states, optimized_logits = final_states[0,-1].astype(mx.float32), k = 10, alpha = 1.2, lower_bound = -100.0, early_layers = None)
            current_logits = evolved_logits
            initial_alpha = alpha
            initial_entropy = entropy_calculation(evolved_logits) 
            best_entropy = initial_entropy
            best_logits = evolved_logits
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
            inner_time = time.perf_counter() - inner_start
            
            if bench:
                bench.add_inner_time(inner_time)
                bench.add_inner_steps(inner_iterations)
            
            total_inner_steps += inner_iterations
            avg_inner_time = inner_time / inner_iterations if inner_iterations > 0 else 0
            method_start = time.perf_counter()
            next_token = mx.argmax(best_logits, axis = -1).item()
            method_time = time.perf_counter() - method_start
            
            if bench:
                bench.set_method_time(method_time)
                bench.set_avg_inner_time(avg_inner_time)
                bench.add_tokens(1)
            
            generated_ids.append(next_token)
            
            if next_token == tokenizer.eos_token_id:
                break
            new_tokens = mx.array([[next_token]])
            final_states, all_hidden_states = model(new_tokens, cache = cache, hidden_states = True)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
            
        except Exception as e:
            print(f"SLEDEMINF error at step {i}: {e}")
            break
    return generated_ids[original_length:]
def sledtemp_generate(model, tokenizer, input_ids, input_ids_no_gen, max_tokens=2048, alpha = 0.5, delta = 0.05, max_kv_size = None, bench = None, prompt_cache=None, use_cache=True):
    # Get input IDs with cache optimization
    model_key = getattr(model, "model_key", id(model))
    cached_input_ids, cache_hit, cache = get_input_ids(prompt_cache, model, input_ids, input_ids_no_gen, model_key, use_cache)
    
    generated_ids = list(cached_input_ids)
    original_length = len(cached_input_ids)
    forward_start = time.perf_counter()
    input_array = mx.array([cached_input_ids])
    final_states, all_hidden_states = model(input_array, cache = cache, hidden_states = True)
    mx.eval([c.state for c in cache])
    mx.clear_cache()
    forward_time = time.perf_counter() - forward_start
    
    if bench:
        bench.set_forward_time(forward_time)
    
    for i in range(max_tokens):
        if i % 1000 == 0:
            print(f"Sledtemp step {i+1}/{max_tokens}")
        try:
            evolved_logits = sled_logits_from_hidden(model, all_hidden_states = all_hidden_states, optimized_logits = final_states[0,-1], k = 10, alpha = 1.2, lower_bound = -100.0, early_layers = None)
            evolved_logits = evolved_logits.astype(mx.float32)
            logits = temp_scaling(evolved_logits, alpha = alpha, delta = delta, bench = bench)
            probs = nn.softmax(logits, axis = -1)
            next_token = mx.random.categorical(mx.log(probs), num_samples=1)[0]
            generated_ids.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id:
                break
            new_tokens = mx.array([[next_token.item()]])
            final_states, all_hidden_states = model(new_tokens, cache = cache, hidden_states = True)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
        except Exception as e:
            print(f"Sledtemp error at step {i}: {e}")
            break
    return generated_ids[original_length:]
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
def eminfsled_generate(model, tokenizer, input_ids, input_ids_no_gen, max_token=2048, num_steps=None, alpha=0.65, threshold=0.05, evolution_rate=1.2, evolution_scale=10, evolution_lower_bound=-100.0, max_kv_size=None, bench=None, prompt_cache=None, use_cache=True):
    # Get input IDs with cache optimization
    model_key = getattr(model, "model_key", id(model))
    cached_input_ids, cache_hit, cache = get_input_ids(prompt_cache, model, input_ids, input_ids_no_gen, model_key, use_cache)
    
    generated_ids = list(cached_input_ids)
    original_length = len(cached_input_ids)
    input_array = mx.array([cached_input_ids])
    final_states, all_hidden_states = model(input_array, cache = cache, hidden_states = True)
    mx.eval([c.state for c in cache])
    mx.clear_cache()
    for i in range(max_token):
        if i % 1000 == 0:
            print(f"EMINFSLED step {i+1}/{max_token}")
        try:
            original_logits = final_states[0,-1].astype(mx.float32)
            optimized_logits, _ = eminf_optimize(original_logits, alpha = alpha, num_steps = num_steps, threshold = threshold)
            optimized_logits = optimized_logits.astype(mx.float32)
            evolved_logits = sled_logits_from_hidden(
                model, 
                all_hidden_states=all_hidden_states, 
                optimized_logits=optimized_logits,
                k=evolution_scale,
                alpha=evolution_rate,
                lower_bound=evolution_lower_bound,
                early_layers=None
            )
            next_token = mx.argmax(evolved_logits, axis=-1).item()
            generated_ids.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break
            next_token_array = mx.array([[next_token]])
            final_states, all_hidden_states = model(
                next_token_array,
                cache=cache,
                hidden_states=True,
            )
            mx.eval([c.state for c in cache])
            mx.clear_cache()
        except Exception as e:
            print(f"EMINFSLED error at step {i}: {e}")
            break
    return generated_ids[original_length:]

def tempsled_generate(model, tokenizer, input_ids, input_ids_no_gen, max_tokens=2048, alpha = 0.5, delta = 0.05, max_kv_size = None, bench = None, prompt_cache=None, use_cache=True):
    # Get input IDs with cache optimization
    model_key = getattr(model, "model_key", id(model))
    cached_input_ids, cache_hit, cache = get_input_ids(prompt_cache, model, input_ids, input_ids_no_gen, model_key, use_cache)
    
    generated_ids = list(cached_input_ids)
    original_length = len(cached_input_ids)
    input_array = mx.array([cached_input_ids])
    final_states, all_hidden_states = model(input_array, cache = cache, hidden_states = True)
    mx.eval([c.state for c in cache])
    mx.clear_cache()
    for i in range(max_tokens):
        if i % 1000 == 0:
            print(f"TEMPSLED step {i+1}/{max_tokens}")
        try:
            original_logits = final_states[0,-1].astype(mx.float32)
            logits = temp_scaling(original_logits, alpha = alpha, delta = delta, bench = bench)
            evolved_logits = sled_logits_from_hidden(model, all_hidden_states = all_hidden_states, optimized_logits = logits)
            next_token = mx.argmax(evolved_logits, axis=-1).item()
            generated_ids.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break
            new_tokens = mx.array([[next_token]])
            final_states, all_hidden_states = model(new_tokens, cache = cache, hidden_states = True)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
        except Exception as e:
            print(f"TEMPSLED error at step {i}: {e}")
            break
    return generated_ids[original_length:]

def eminf_generate(model, tokenizer, input_ids, input_ids_no_gen, max_token = 2048, num_steps = None, alpha = 0.65, threshold = 0.05, max_kv_size = None, bench=None, prompt_cache=None, use_cache=True):
    # Get input IDs with cache optimization
    model_key = getattr(model, "model_key", id(model))
    cached_input_ids, cache_hit, cache = get_input_ids(prompt_cache, model, input_ids, input_ids_no_gen, model_key, use_cache)
    
    generated_ids = list(cached_input_ids)
    original_length = len(cached_input_ids)
    
    forward_start = time.perf_counter()
    output, _ = model(mx.array([cached_input_ids]), cache = cache, hidden_states = False)
    mx.eval([c.state for c in cache])
    mx.clear_cache()

    # Handle both single output and tuple output
    if isinstance(output, tuple):
        output = output[0]  # Take the first element (logits)
    output = mx.stop_gradient(output)
    logits = output[:, -1, :].astype(mx.float32)
    forward_time = time.perf_counter() - forward_start
    
    if bench:
        bench.set_forward_time(forward_time)
    
    for i in range(max_token):
        if i % 1000 == 0:
            print(f"EMINF Step {i+1}/{max_token}")
        try:
            best_logits, best_entropy = eminf_optimize(logits, alpha = alpha, num_steps = num_steps, threshold = threshold)
            final_probs = nn.softmax(best_logits, axis = -1)
            probs_np = np.array(final_probs).flatten()
            probs_np = probs_np / probs_np.sum()
            next_token = np.random.choice(len(probs_np), p=probs_np)
            generated_ids.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break
            new_tokens = mx.array([[next_token]])
            output2, _ = model(new_tokens, cache = cache, hidden_states = False)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
            if isinstance(output2, tuple):
                output2 = output2[0] 
            output2 = mx.stop_gradient(output2)
            logits = output2[:, -1, :].astype(mx.float32)
        except Exception as e:
            print(f"EMINF error at step {i}: {e}")
            break
    return generated_ids[original_length:]

def temp_scaling(logits, alpha=0.6, delta=0.05, max_init=1.99, min_init=0.01, iterations=15, tolerance=1e-4, bench=None):
    P_initial = nn.softmax(logits, axis=-1)
    H_initial = -mx.sum(P_initial *nn.log_softmax(logits, axis = -1), axis = -1)
    H_target = mx.maximum(delta, alpha * H_initial)
    final_temp = 1.0
    inner_iterations = 0
    
    if float(H_initial) > delta:    
        low_temp = min_init
        high_temp = max_init
        inner_start = time.perf_counter()
        for _ in range(iterations):
            inner_iterations += 1
            mid_temp = (low_temp + high_temp) / 2.0
            current_logits = logits/mid_temp
            current_P = nn.softmax(current_logits, axis=-1)
            current_H = -mx.sum(current_P * nn.log_softmax(current_logits, axis = -1), axis = -1)
            if mx.abs(float(current_H - H_target)) < tolerance:
                final_temp = mid_temp
                break
            elif float(current_H) < float(H_target):
                low_temp = mid_temp
            else:
                high_temp = mid_temp
            final_temp = mid_temp
        inner_time = time.perf_counter() - inner_start
        
        if bench:
            bench.add_inner_time(inner_time)
            bench.add_inner_steps(inner_iterations)
    
    final_logits = logits/final_temp
    return final_logits

def temp_generate(model, tokenizer, input_ids, input_ids_no_gen, max_tokens=4096, alpha=0.6, delta=0.05, max_kv_size=None, bench=None, prompt_cache=None, use_cache=True):
    # Get input IDs with cache optimization
    model_key = getattr(model, "model_key", id(model))
    cached_input_ids, cache_hit, cache = get_input_ids(prompt_cache, model, input_ids, input_ids_no_gen, model_key, use_cache)
    
    generated_ids = list(cached_input_ids)
    original_length = len(cached_input_ids)
    output, _ = model(mx.array([cached_input_ids]), cache = cache, hidden_states = False)
    mx.eval([c.state for c in cache])
    mx.clear_cache()
    
    # Handle both single output and tuple output
    if isinstance(output, tuple):
        output = output[0]  # Take the first element (logits)
    output = mx.stop_gradient(output)
    logits = output[:, -1, :].astype(mx.float32)
    for i in range(max_tokens):
        if i % 1000 == 0:
            print(f"Generating token {i} of {max_tokens}")
        logits = temp_scaling(
            logits,
            alpha=alpha,
            delta=delta,
            bench=bench,
        )
        probs = nn.softmax(logits, axis=-1)
        next_token = mx.random.categorical(mx.log(probs), num_samples=1)[0]
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated_ids.append(next_token.item())
        
        new_tokens = mx.array([[next_token.item()]])
        output2, _ = model(new_tokens, cache = cache, hidden_states = False)
        mx.eval([c.state for c in cache])
        mx.clear_cache()
        if isinstance(output2, tuple):
            output2 = output2[0]
        output2 = mx.stop_gradient(output2)
        logits = output2[:, -1, :].astype(mx.float32)
    return generated_ids[original_length:]

def project_logits(model, h_lastpos):
    """Project hidden states to logits using the model's output projection"""
    if model.args.tie_word_embeddings:
        return model.model.embed_tokens.as_linear(h_lastpos)
    else:
        return model.lm_head(h_lastpos)

def sled_logits_from_hidden(model,
                            all_hidden_states,
                            optimized_logits,
                            k=15,
                            alpha=1.5,
                            tau=1.0,
                            lower_bound=-100.0,
                            early_layers=None):
    """
    USE SLED evolution method to logits from hidden states
    Return evolved logits for the last token position
    """
    pN = mx.softmax(optimized_logits / tau, axis=-1)
    topk_idx = mx.argsort(pN)[-k:]
    seq_i = -1
    mature_idx = len(all_hidden_states) - 1 #get the hidden state of last token in all layers
    logits_N = optimized_logits
    logitsN_top = logits_N[topk_idx]
    pN_top = mx.softmax(logitsN_top / tau, axis=-1)
    
    if early_layers is None:
        early_layers = list(range(1, mature_idx))
    if getattr(model.args,"tie_word_embeddings",False):
        W_top = model.model.embed_tokens.weight[topk_idx]
    else:
        W_top = model.lm_head.weight[topk_idx]
   # phase 1: calculate mi for each early layer
    kI = mx.eye(k)
    m_sum = mx.zeros(k)
    layer_weights = []
    per_layer_m = []
    for li in early_layers:
        h_last = all_hidden_states[li][0, seq_i] #hidden state at layer li
        logits_n_top= h_last @W_top.T #project to logits ONLY FOR TOPK TOKENS
        pn_top = mx.softmax(logits_n_top / tau, axis=-1)
        a = logits_n_top - logitsN_top #difference for topk logits
        a_norm = mx.linalg.norm(a)+ 1e-8
        a_hat = a / a_norm # a
        grad = pn_top[None, :] - kI # change of KL(P_ei, P_n)
        grad_norm = mx.linalg.norm(grad, axis=1) + 1e-8
        cos = mx.sum(a_hat[None, :] * grad, axis=1) / grad_norm 
        mi = mx.maximum(cos, 0.0) ** 2 #mi should be max(cossim(logitsn-logitsN, Pn-Pei),0)**2
        per_layer_m.append(mi)
        layer_weights.append(mx.sum(mi))
    # phase 2: aggregate by layer weights
    total_w = sum(layer_weights) + 1e-8
    for m_i, w in zip(per_layer_m, layer_weights):
        m_sum += (w / total_w) * (m_i / (mx.sum(m_i) + 1e-8))
    m_top = m_sum
    # phase 3: single-step gradient descent on final logits 
    g_top = (pN_top - m_top) / tau
    evolved_top = logitsN_top - alpha * g_top
    evolved = mx.array(optimized_logits)
    evolved = evolved.at[topk_idx].add(evolved_top - logitsN_top)
    return evolved

def sled_generate(model, tokenizer, input_ids, input_ids_no_gen, max_tokens=2048, 
                  evolution_rate=1.2, evolution_scale=10, evolution_lower_bound=-100, 
                  max_kv_size=None, bench=None, prompt_cache=None, use_cache=True):
    # Get input IDs with cache optimization
    model_key = getattr(model, "model_key", id(model))
    cached_input_ids, cache_hit, cache = get_input_ids(prompt_cache, model, input_ids, input_ids_no_gen, model_key, use_cache)
    
    generated_ids = list(cached_input_ids)
    original_length = len(cached_input_ids)
    
    input_array = mx.array([cached_input_ids])
    final_hidden, all_hidden_states = model(
        input_array,
        cache=cache,
        hidden_states=True,
    )
    mx.eval([c.state for c in cache])
    mx.clear_cache()
    
    for i in range(max_tokens):
        if i % 1000 == 0:
            print(f"SLED Step {i+1}/{max_tokens}")
        
        try:
            evolved_logits = sled_logits_from_hidden(
                model, all_hidden_states=all_hidden_states, optimized_logits = final_hidden[0,-1].astype(mx.float32),
                k=evolution_scale, alpha=evolution_rate, 
                lower_bound=evolution_lower_bound, early_layers=None,
            )
            next_token = mx.argmax(evolved_logits, axis=-1).item()
            
            generated_ids.append(next_token)
            
            if next_token == tokenizer.eos_token_id:
                break
                
            next_token_array = mx.array([[next_token]])
            final_hidden, all_hidden_states = model(
                next_token_array,
                cache=cache,
                hidden_states=True,
            )
            mx.eval([c.state for c in cache])
            mx.clear_cache()
            
        except Exception as e:
            print(f"SLED error at step {i}: {e}")
            break
    
    return generated_ids[original_length:]

def gbxlm_generate(model, tokenizer, prompt, temperature, top_p, max_tokens, repetition_penalty = 1.0, bench=None):
    sampler = make_sampler(temp =temperature, top_p = top_p)
    logits_processors =make_logits_processors(repetition_penalty = repetition_penalty)
    generation_start = time.perf_counter()
    from .utils import generate
    response = generate(model, tokenizer, prompt, sampler=sampler, logits_processors=logits_processors, max_tokens=max_tokens)
    generation_time = time.perf_counter() - generation_start
    
    if bench:
        full_response_tokens = len(tokenizer.encode(str(response)))
        generated_tokens = full_response_tokens
        bench.add_tokens(generated_tokens)
        bench.set_method_time(generation_time)
    
    return str(response).strip()

def generate_response(model, tokenizer, messages, model_name, bench=None, prompt_cache=None, use_cache=True):
    config = decoding_config[model_name]
    
    if bench:
        bench.start()
    
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking = False)
    input_ids_no_gen = tokenizer.apply_chat_template(messages, add_generation_prompt=False, enable_thinking = False)
        
    if config["use_eminf"]:
        generated_ids = eminf_generate(model, tokenizer, input_ids, input_ids_no_gen, max_token = config["max_new_tokens"], bench=bench, prompt_cache=prompt_cache, use_cache=use_cache)
        response = tokenizer.decode(np.array(generated_ids), skip_special_tokens=True)
        response = response.strip()
    elif config["use_temp_scaling"]:
        generated_ids = temp_generate(model, tokenizer, input_ids, input_ids_no_gen, max_tokens = config["max_new_tokens"], alpha = config["alpha"], delta = config["delta"], bench=bench, prompt_cache=prompt_cache, use_cache=use_cache)
        response = tokenizer.decode(np.array(generated_ids), skip_special_tokens=True)
        response = response.strip()
    elif config["use_sled"]:
        generated_ids = sled_generate(
            model, tokenizer, input_ids, input_ids_no_gen, 
            max_tokens=config["max_new_tokens"],
            evolution_rate=config["evolution_rate"],
            evolution_scale=config["evolution_scale"],
            evolution_lower_bound=config["evolution_lower_bound"],
            bench=bench,
            prompt_cache=prompt_cache,
            use_cache=use_cache
        )
        response = tokenizer.decode(np.array(generated_ids), skip_special_tokens=True)
        response = response.strip()
    elif config["use_sledeminf"]:
        generated_ids = sledeminf_generate(model, tokenizer, input_ids, input_ids_no_gen, max_token = config["max_new_tokens"], bench=bench, prompt_cache=prompt_cache, use_cache=use_cache)
        response = tokenizer.decode(np.array(generated_ids), skip_special_tokens=True)
        response = response.strip()
    elif config["use_eminfsled"]:
        generated_ids = eminfsled_generate(
            model, tokenizer, input_ids, input_ids_no_gen, 
            max_token=config["max_new_tokens"], 
            num_steps=config["num_steps"], 
            alpha=config["alpha"], 
            threshold=config["threshold"],
            evolution_rate=config.get("evolution_rate", 1.2),
            evolution_scale=config.get("evolution_scale", 10),
            evolution_lower_bound=config.get("evolution_lower_bound", -100.0),
            bench=bench,
            prompt_cache=prompt_cache,
            use_cache=use_cache
        )
        response = tokenizer.decode(np.array(generated_ids), skip_special_tokens=True)
        response = response.strip()
    elif config["use_sledtemp"]:
        generated_ids = sledtemp_generate(model, tokenizer, input_ids, input_ids_no_gen, max_tokens = config["max_new_tokens"], alpha = config["alpha"], delta = config["delta"], bench=bench, prompt_cache=prompt_cache, use_cache=use_cache)
        response = tokenizer.decode(np.array(generated_ids), skip_special_tokens=True)
        response = response.strip()
    elif config["use_tempsled"]:
        generated_ids = tempsled_generate(model, tokenizer, input_ids, input_ids_no_gen, max_tokens = config["max_new_tokens"], alpha = config["alpha"], delta = config["delta"], bench=bench, prompt_cache=prompt_cache, use_cache=use_cache)
        response = tokenizer.decode(np.array(generated_ids), skip_special_tokens=True)
        response = response.strip()
    if bench:
        bench.stop()
    return response if response else "Error: No response generated"

def generate_gbx_response(prompt: str, *, model: str, gbx_mode: str, bench=None) -> str:
    """Generate response using GBX models with different variants"""
    try:
        if model == "Qwen/Qwen3-8B-MLX-bf16":
            from mlx_lm import load
            model_config = {"rope_scaling": None}
            model_obj, tokenizer = load(model, model_config = model_config)
        else:
            from gbx_lm import load
            model_obj, tokenizer = load(model)

        messages = [{"role": "user", "content": prompt}]
        response = generate_response(model_obj, tokenizer, messages, gbx_mode, bench=bench)
        return response
        
    except Exception as e:
        raise ValueError(f"GBX generation failed: {e}")