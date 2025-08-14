from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
class PromptCache:
    """MLX prompt cache for efficient multi-step generation"""
    def __init__(self):
        self.cache = None
        self.tokens_no_gen = [] 
        self.model_key = None
        self.system_cached = False
        self.system_tokens = []

    def _common_prefix(self, a, b):
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def cache_system_prompt(self, model, system_prompt, tokenizer):
        """Pre-cache the system prompt to ensure it's always available"""
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

        import mlx.core as mx
        system_array = mx.array([self.system_tokens])
        model(system_array, cache=self.cache)
        mx.eval([c.state for c in self.cache])
        mx.clear_cache()
        
        self.system_cached = True
        self.tokens_no_gen = list(self.system_tokens)
        print(f"System prompt cached: {len(self.system_tokens)} tokens")

    def get_prompt_cache(self, model, tokens_with_gen, tokens_no_gen, model_key):
        if self.cache is None or self.model_key != model_key:
            self.cache = make_prompt_cache(model)
            self.model_key = model_key
            self.tokens_no_gen = list(tokens_no_gen)
            self.system_cached = False
            return tokens_with_gen, self.cache, False

        if self.system_cached:
            if not tokens_no_gen[:len(self.system_tokens)] == self.system_tokens:
                self.cache = make_prompt_cache(model)
                self.tokens_no_gen = list(tokens_no_gen)
                self.system_cached = False
                return tokens_with_gen, self.cache, False

        prefix_len = self._common_prefix(self.tokens_no_gen, tokens_no_gen)

        if prefix_len < len(self.tokens_no_gen):
            if self.system_cached and prefix_len >= len(self.system_tokens):
                new_tokens_no_gen = tokens_no_gen[len(self.system_tokens):]
                gen_suffix_len = len(tokens_with_gen) - len(tokens_no_gen)
                gen_suffix = tokens_with_gen[-gen_suffix_len:] if gen_suffix_len > 0 else []
                tokens_to_process = list(new_tokens_no_gen) + list(gen_suffix)
                self.tokens_no_gen = list(tokens_no_gen)
                return tokens_to_process, self.cache, True
            else:
                self.cache = make_prompt_cache(model)
                self.tokens_no_gen = list(tokens_no_gen)
                self.system_cached = False
                return tokens_with_gen, self.cache, False

        new_tokens_no_gen = tokens_no_gen[prefix_len:]
        gen_suffix_len = len(tokens_with_gen) - len(tokens_no_gen)
        gen_suffix = tokens_with_gen[-gen_suffix_len:] if gen_suffix_len > 0 else []
        tokens_to_process = list(new_tokens_no_gen) + list(gen_suffix)

        self.tokens_no_gen = list(tokens_no_gen)
        return tokens_to_process, self.cache, True

    def update_after_step(self, messages, tokenizer):
        """After generation, trim the generation suffix so the cache matches no-gen prefix."""
        if not self.cache:
            return
        try:
            tokens_with_gen = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False
            )
            tokens_no_gen = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, enable_thinking=False
            )
            num_to_trim = len(tokens_with_gen) - len(tokens_no_gen)
            if num_to_trim > 0 and can_trim_prompt_cache(self.cache):
                trim_prompt_cache(self.cache, num_to_trim)
        except Exception:
            pass