"""
Tokens' configuration for different models(DeepSeek R1/Qwen3/Llama3)
"""

class ModelConfig:
    MODEL_CONFIGS = {
        # Llama3 series models
        "llama3": {
            "use_update_after_step": True,
            "generation_tokens": [128006, 78191, 128007, 271],
            "description": "tokens thatLlama3 series models used for update_after_step function"
        },
        
        # Qwen3 series models
        "qwen3": {
            "use_update_after_step": True,
            "generation_tokens": [151644, 77091, 198],
            "description": "tokens that Qwen3 series models used for update_after_step function"
        },
        
        # DeepSeek series models
        "deepseek": {
            "use_update_after_step": False,
            "generation_tokens": [],  # DeepSeek uses update method, no specific tokens are needed
            "description": "tokens that DeepSeek series models used for update_after_step function"
        }
    }
    
    @classmethod
    def get_model_type(cls, model_name: str) -> str:
        """
        infer model type from model name
        Args:
            model_name: model name
        Returns:
            model type string
        """
        model_name_lower = model_name.lower()
        #There is a problem: if the model name contains both qwen3 and deepseek, it will be recognized as deepseek first(因为我看公司已有的模型这两个明细同时出现的都是deepseek)
        if "llama3" in model_name_lower or "llama-3" in model_name_lower:
            return "llama3"
        elif "deepseek" in model_name_lower or "deepseek-r1" in model_name_lower:
            return "deepseek"
        elif "qwen3" in model_name_lower or "qwen-3" in model_name_lower:
            return "qwen3"
        else:
            # default to qwen3 configuration
            print(f"Warning: unrecognized model type {model_name}, using default configuration(qwen3)")
            return "qwen3"
    
    @classmethod
    def get_config(cls, model_name: str) -> dict:
        """
        get model configuration
        Args:
            model_name: model name
        Returns:
            model configuration dictionary
        """
        model_type = cls.get_model_type(model_name)
        return cls.MODEL_CONFIGS[model_type].copy()
