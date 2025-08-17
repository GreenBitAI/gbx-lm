import mlx.core as mx
from gbx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

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
    
    def __init__(self):
        self.cache = None
        # store the conversation tokens WITHOUT generation prompt for consistent caching
        self.tokens_no_gen = [] 
        self.model_key = None
        # track if system prompt is cached (optimization for multi-conversation reuse)
        self.system_cached = False
        self.system_tokens = []

    def _common_prefix(self, a, b):
        """Find the length of the common prefix between two token sequences."""
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

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
        self.cache = make_prompt_cache(model)
        self.model_key = model_key

        # pre-compute the system prompt through the model: pre-cache system prompt
        system_array = mx.array([self.system_tokens])
        model(system_array, cache=self.cache)
        mx.eval([c.state for c in self.cache])
        mx.clear_cache()
        
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
            self.cache = make_prompt_cache(model)
            self.model_key = model_key
            self.tokens_no_gen = list(tokens_no_gen)
            self.system_cached = False
            return tokens_with_gen, self.cache, False

        # If system prompt is cached but current tokens don't start with it, start new cache
        if self.system_cached:
            if not tokens_no_gen[:len(self.system_tokens)] == self.system_tokens:
                self.cache = make_prompt_cache(model)
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
                new_tokens_no_gen = tokens_no_gen[len(self.system_tokens):]
                gen_suffix_len = len(tokens_with_gen) - len(tokens_no_gen) 
                gen_suffix = tokens_with_gen[-gen_suffix_len:] if gen_suffix_len > 0 else []
                tokens_to_process = list(new_tokens_no_gen) + list(gen_suffix)
                self.tokens_no_gen = list(tokens_no_gen)
                return tokens_to_process, self.cache, True
            else:
                # If the cache miss completely - start over
                self.cache = make_prompt_cache(model)
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

    def update_after_step(self, messages, tokenizer, enable_thinking=False):
        """
        After generation, trim the generation suffix(e.g. <assistant>, /n, etc.) so the cache matches no-gen prefix.
        Without this step, the cache would contain generation prompt tokens that don't match the actual conversation state, causing cache misses.
        """
        if not self.cache:
            return
        try:
            tokens_with_gen = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=enable_thinking
            )
            tokens_no_gen = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, enable_thinking=enable_thinking
            )
            # calculate tokens to trim (the generation prompt)
            num_to_trim = len(tokens_with_gen) - len(tokens_no_gen)
            if num_to_trim > 0 and can_trim_prompt_cache(self.cache):
                trim_prompt_cache(self.cache, num_to_trim)
        except Exception:
            pass