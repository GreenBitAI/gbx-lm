# Copyright © 2024 Apple Inc.
# Additional code from GreenBitAI is licensed under the Apache 2.0 License.

import os
import tempfile
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten
from gbx_lm import utils

HF_MODEL_PATH = "GreenBitAI/Qwen-1.5-0.5B-Chat-layer-mix-bpw-2.2-mlx"

class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        if not os.path.isdir(cls.test_dir):
            os.mkdir(cls.test_dir_fid.name)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_load(self):
        model, _ = utils.load(HF_MODEL_PATH)

        model_lazy, _ = utils.load(HF_MODEL_PATH, lazy=True)

        mx.eval(model_lazy.parameters())

        p1 = model.layers[0].mlp.up_proj.qweight
        p2 = model_lazy.layers[0].mlp.up_proj.qweight
        self.assertTrue(mx.allclose(p1, p2))

    def test_make_shards(self):
        from gbx_lm.models import qllama as llama

        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=2048,
            num_hidden_layers=32,
            intermediate_size=4096,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=30_000,
        )
        model = llama.Model(args)
        weights = tree_flatten(model.parameters())
        gb = sum(p.nbytes for _, p in weights) // 2**30
        shards = utils.make_shards(dict(weights), 1)
        self.assertTrue(gb <= len(shards) <= gb + 1)


if __name__ == "__main__":
    unittest.main()
