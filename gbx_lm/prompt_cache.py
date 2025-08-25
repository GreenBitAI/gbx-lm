import mlx.core as mx
from gbx_lm.models.cache import make_prompt_cache
from gbx_lm.examples.config import ModelConfig
class PromptCache:
    """
    This prompt cache has some differences from the origianl MLX prompt cache.
    1.Generation prompt handling: the original one's comparison is not robust, generation templates lead to different common prefix matching, 
    I use two different token sequences to handle the generation prompt:
       - tokens_no_gen: [system, user1, assistant1, user2] (for caching)
       - tokens_with_gen: [system, user1, assistant1, user2, <generation_prompt>] (for inference)
    The original approach can't distinguish between these, leading to cache invalidation when the generation prompt is added/removed.
    2. I add pre-cache system prompt
    """
    
    def __init__(self, model_name: str, quantize: bool = False, qbit=None, q_group_size=None):
        self.cache = None
        # store the conversation tokens WITHOUT generation prompt for consistent caching
        self.tokens_no_gen = [] 
        self.model_key = None
        self.model_name = model_name
        self.model_config = ModelConfig.get_config(model_name)
        # track if system prompt is cached (optimization for multi-conversation reuse)
        self.system_cached = False
        self.system_tokens = []
        self.quantize = quantize # to control whether to quantize cache or not
        self.qbit = qbit
        self.q_group_size = q_group_size

    def _common_prefix(self, a, b):
        """Find the length of the common prefix between two token sequences."""
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i
    
    def _new_cache(self, model):
        """
        Create a new prompt cache for the given model.
        """
        base_cache = make_prompt_cache(model)
        return base_cache
    
    def _quantize_cache(self, cache):
        if not self.quantize or cache is None:
            return cache
        quantized_cache = []
        for c in cache:
        # don't add max-kv-size when making a prompt cache, because that will build rotatingkvcache and trigger NotImplementedError
        # rotatingkvcache doesn't support quantization
            if hasattr(c, "to_quantized"):
                quantized_cache.append(c.to_quantized(self.q_group_size, self.qbit))
            else:
                quantized_cache.append(c)
        return quantized_cache
    
    def cache_system_prompt(self, model, system_prompt, tokenizer):
        """
        Pre-cache the system prompt to ensure it's always available.
        """
        model_key = getattr(model, "model_key", id(model))
        
        if self.system_cached and self.model_key == model_key:
            print("System prompt already cached")
            return

        system_messages = [{"role": "system", "content": system_prompt}]
        self.system_tokens = tokenizer.apply_chat_template(
            system_messages, add_generation_prompt=False, enable_thinking=False
        )
        self.model_key = model_key

        # pre-compute the system prompt through the model: pre-cache system prompt
        base_cache = self._new_cache(model)
        system_array = mx.array([self.system_tokens])
        model(system_array, cache=base_cache)
        mx.eval([c.state for c in base_cache])
        mx.clear_cache()
        
        quantized_cache = self._quantize_cache(base_cache)
        self.cache = quantized_cache

        self.system_cached = True
        self.tokens_no_gen = list(self.system_tokens)
        print(f"System prompt cached: {len(self.system_tokens)} tokens")

    def get_prompt_cache(self, model, tokens_with_gen, tokens_no_gen, model_key):
        """
        Get the prompt cache for the given model and tokens.
        Args:
            model: LLM model
            tokens_with_gen: full token sequence including generation prompt (for inference)
            tokens_no_gen: token sequence without generation prompt (for caching)
            model_key: unique identifier for the model
        Returns:
            tuple: (tokens_to_process, cache, cache_hit)
            - tokens_to_process: tokens that need to be processed by the model
            - cache: the prompt cache to use
            - cache_hit: boolean indicating if we got a cache benefit
        """
        # IF there is no existing cache OR the model changed to another one - start new cache
        if self.cache is None or self.model_key != model_key:
            base_cache = self._new_cache(model)
            self.cache = self._quantize_cache(base_cache)
            self.model_key = model_key
            self.tokens_no_gen = list(tokens_no_gen)
            self.system_cached = False
            return tokens_with_gen, self.cache, False

        # If system prompt is cached but current tokens don't start with it, start new cache
        if self.system_cached:
            if not tokens_no_gen[:len(self.system_tokens)] == self.system_tokens:
                base_cache = self._new_cache(model)
                self.cache = self._quantize_cache(base_cache)
                self.tokens_no_gen = list(tokens_no_gen)
                self.system_cached = False
                return tokens_with_gen, self.cache, False

        # Find how much of the cached conversation matches the current conversation
        prefix_len = self._common_prefix(self.tokens_no_gen, tokens_no_gen)

        # If conversation diverged from cache
        if prefix_len < len(self.tokens_no_gen):
            # If we have system cache and the divergence is after system prompt
            # use the system cache and just process the new conversation
            if self.system_cached and prefix_len >= len(self.system_tokens):
                # only process the new conversation part + generation suffix
                new_tokens_no_gen = tokens_no_gen[prefix_len:] #the new tokens should start after the common prefix
                gen_suffix_len = len(tokens_with_gen) - len(tokens_no_gen) 
                gen_suffix = tokens_with_gen[-gen_suffix_len:] if gen_suffix_len > 0 else []
                tokens_to_process = list(new_tokens_no_gen) + list(gen_suffix)
                self.tokens_no_gen = list(tokens_no_gen)
                return tokens_to_process, self.cache, True
            else:
                # If the cache miss completely - start over
                base_cache = self._new_cache(model)
                self.cache = self._quantize_cache(base_cache)
                self.tokens_no_gen = list(tokens_no_gen)
                self.system_cached = False
                return tokens_with_gen, self.cache, False

        # If the cache hit - current conversation extends the cached conversation
        # only process the new tokens + generation suffix
        new_tokens_no_gen = tokens_no_gen[prefix_len:]
        gen_suffix_len = len(tokens_with_gen) - len(tokens_no_gen)
        gen_suffix = tokens_with_gen[-gen_suffix_len:] if gen_suffix_len > 0 else []
        tokens_to_process = list(new_tokens_no_gen) + list(gen_suffix)

        self.tokens_no_gen = list(tokens_no_gen)
        return tokens_to_process, self.cache, True

    def update_after_step(self, generated_token_ids, messages, tokenizer):
        """
        update tokens no gen
        """
        if self.model_config["use_update_after_step"]: #if the model is qwen3/llama3 series models, this will be true
            generation_tokens = self.model_config["generation_tokens"] #the generation tokens are "<imstart>"
            self.tokens_no_gen.extend(generation_tokens)
            self.tokens_no_gen.extend(generated_token_ids) #update self.tokens_no_gen with the newly generated assistant response tokens
        else:
            self.tokens_no_gen = tokenizer.apply_chat_template(messages, add_generation_prompt=False, enable_thinking=False)
            #if the model is deepseek series models, to avoid the effect of thinking part, we can directly use apply_chat_template to update self.tokens_no_gen
        self.system_cached = (
            len(self.system_tokens) > 0 and
            self.tokens_no_gen[:len(self.system_tokens)] == self.system_tokens
        )
