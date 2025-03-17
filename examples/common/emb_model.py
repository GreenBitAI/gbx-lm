import json
import os
import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import yaml


def average_pool(last_hidden_state: mx.array, attention_mask: mx.array) -> mx.array:
    last_hidden = mx.multiply(last_hidden_state, attention_mask[..., None])
    return last_hidden.sum(axis=1) / attention_mask.sum(axis=1)[..., None]


class ModelConfig(BaseModel):
    dim: int
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int
    attention_probs_dropout_prob: float
    hidden_dropout_prob: float
    layer_norm_eps: float
    max_position_embeddings: int


class TransformerEncoderLayer(nn.Module):
    """
    A transformer encoder layer with (the original BERT) post-normalization.
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        mlp_dims = mlp_dims or dims * 4
        self.attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.ln1 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.linear1 = nn.Linear(dims, mlp_dims)
        self.linear2 = nn.Linear(mlp_dims, dims)
        self.gelu = nn.GELU()

    def __call__(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        add_and_norm = self.ln1(x + attention_out)

        ff = self.linear1(add_and_norm)
        ff_gelu = self.gelu(ff)
        ff_out = self.linear2(ff_gelu)
        x = self.ln2(ff_out + add_and_norm)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None
    ):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(dims, num_heads, mlp_dims)
            for i in range(num_layers)
        ]

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x


class BertEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.token_type_embeddings = nn.Embedding(2, config.dim)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.dim
        )
        self.norm = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(
            mx.broadcast_to(mx.arange(input_ids.shape[1]), input_ids.shape)
        )
        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = position + words + token_types
        return self.norm(embeddings)


class Bert(nn.Module):
    def __init__(self, config: ModelConfig):
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(
            num_layers=config.num_hidden_layers,
            dims=config.dim,
            num_heads=config.num_attention_heads,
        )
        self.pooler = nn.Linear(config.dim, config.dim)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array,
        attention_mask: mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        x = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            # convert 0's to -infs, 1's to 0's, and make it broadcastable
            attention_mask = mx.log(attention_mask)
            attention_mask = mx.expand_dims(attention_mask, (1, 2))

        y = self.encoder(x, attention_mask)
        return y, mx.tanh(self.pooler(y[:, 0]))


class Model:
    def __init__(self) -> None:
        # Setup paths for mlx-bge-small-en
        self.mlx_model_path = self.setup_mlx_model()

        # Setup path for bge-small-en
        self.model_path = self.setup_bge_model()

        # Load the config file
        config_path = os.path.join(self.mlx_model_path, "config.json")
        with open(config_path) as f:
            model_config = ModelConfig(**json.load(f))

        self.model = Bert(model_config)

        # Load model weights
        weights_path = os.path.join(self.mlx_model_path, "bge-small-en.npz")
        self.model.load_weights(weights_path)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def setup_mlx_model(self):
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        repo_id = "Jaward/mlx-bge-small-en"

        # Download config.json
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=cache_dir)

        # Download model weights
        weights_path = hf_hub_download(repo_id=repo_id, filename="bge-small-en.npz", cache_dir=cache_dir)

        return os.path.dirname(config_path)

    def setup_bge_model(self):
        # Try to get model path from environment variable
        bge_path = os.environ.get('BGE_SMALL_EN_PATH')

        if not bge_path:
            # If env var is not set, try to read from config file
            config_file = 'config.yaml'
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    bge_path = config.get('BGE_SMALL_EN_PATH')

        if not bge_path or not os.path.exists(bge_path):
            # If BGE_SMALL_EN_PATH is not set, check the current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            potential_bge_path = os.path.join(current_dir, 'bge-small-en')

            if os.path.exists(potential_bge_path):
                bge_path = potential_bge_path
            else:
                return "BAAI/bge-small-en"

        return bge_path

    def run(self, input_text: List[str]) -> mx.array:
        tokens = self.tokenizer(input_text, return_tensors="np", padding=True)
        tokens = {key: mx.array(v) for key, v in tokens.items()}

        last_hidden_state, _ = self.model(**tokens)

        embeddings = average_pool(
            last_hidden_state, tokens["attention_mask"].astype(mx.float32)
        )
        embeddings = embeddings / mx.linalg.norm(embeddings, ord=2, axis=1)[..., None]

        return embeddings


class BertMLXEmbeddings:
    def __init__(self):
        self.model = Model()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.run(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.model.run([text])
        return embeddings[0].tolist()


# Define a function to check LangChain compatibility and update the class if necessary
def setup_langchain_compatibility():
    try:
        from langchain.embeddings.base import Embeddings
        global BertMLXEmbeddings
        if not issubclass(BertMLXEmbeddings, Embeddings):
            class BertMLXEmbeddingsLangChain(BertMLXEmbeddings, Embeddings):
                pass
            BertMLXEmbeddings = BertMLXEmbeddingsLangChain
        return True
    except ImportError:
        return False

# Run the setup function
is_langchain_compatible = setup_langchain_compatibility()

# Keep the convenience function
def get_bert_mlx_embeddings() -> BertMLXEmbeddings:
    return BertMLXEmbeddings()
