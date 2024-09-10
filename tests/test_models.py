# Copyright © 2024 Apple Inc.
# Additional code from GreenBitAI is licensed under the Apache 2.0 License.

import unittest

import mlx.core as mx
from gbx_lm.models.base import KVCache, RotatingKVCache
from gbx_lm.models.quantized_linear_gba import QuantizedLinear
import numpy as np

class TestModels(unittest.TestCase):

    def initialize_quantized_linear(self, model):
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinear):
                # Generate random uint32 values using numpy
                random_values = np.random.randint(0, 2 ** 32, size=module.qweight.shape, dtype=np.uint32)
                # Convert numpy array to mlx array
                random_mlx_array = mx.array(random_values)
                # Update the existing qweight values
                module.qweight[:] = random_mlx_array


    def test_kv_cache(self):
        cache = KVCache(32, 4)

        k = mx.ones((1, 4, 1, 32), mx.float16)
        v = mx.ones((1, 4, 1, 32), mx.float16)

        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up, k))
        self.assertTrue(mx.array_equal(v_up, v))
        self.assertEqual(cache.offset, 1)

        k = mx.ones((1, 4, cache.step, 32), mx.float16)
        v = mx.ones((1, 4, cache.step, 32), mx.float16)
        k_up, v_up = cache.update_and_fetch(k, v)

        expected = mx.ones((1, 4, cache.step + 1, 32), mx.float16)
        self.assertTrue(mx.array_equal(k_up, expected))
        self.assertTrue(mx.array_equal(v_up, expected))
        self.assertEqual(cache.offset, cache.step + 1)

    def test_rotating_kv_cache(self):
        b, h, d = 1, 2, 32
        cache = RotatingKVCache(d, h, max_size=8, step=4)

        k = mx.random.uniform(shape=(b, h, 2, d))
        v = mx.random.uniform(shape=(b, h, 2, d))

        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up, k))
        self.assertTrue(mx.array_equal(v_up, v))
        self.assertEqual(cache.offset, 2)

        k = mx.random.uniform(shape=(b, h, 5, d))
        v = mx.random.uniform(shape=(b, h, 5, d))
        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up[..., 2:, :], k))
        self.assertTrue(mx.array_equal(v_up[..., 2:, :], v))

        k = mx.random.uniform(shape=(b, h, 4, d))
        v = mx.random.uniform(shape=(b, h, 4, d))
        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up[..., -4:, :], k))
        self.assertTrue(mx.array_equal(v_up[..., -4:, :], v))

        idx = 0
        for _ in range(10):
            k = mx.random.uniform(shape=(b, h, 1, d))
            v = mx.random.uniform(shape=(b, h, 1, d))
            k_up, v_up = cache.update_and_fetch(k, v)
            self.assertTrue(mx.array_equal(k_up[..., idx : idx + 1, :], k))
            self.assertTrue(mx.array_equal(v_up[..., idx : idx + 1, :], v))
            idx += 1
            idx %= 8

        # Try with nonzero keep
        cache = RotatingKVCache(d, h, max_size=8, step=4, keep=2)

        # Check a large update
        k = mx.random.uniform(shape=(b, h, 20, d))
        v = mx.random.uniform(shape=(b, h, 20, d))
        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up, k))
        self.assertTrue(mx.array_equal(v_up, v))

        # A bunch of small updates
        self.assertEqual(cache.offset, 20)
        idx = 2
        for i in range(10):
            k = mx.random.uniform(shape=(b, h, 1, d))
            v = mx.random.uniform(shape=(b, h, 1, d))
            k_up, v_up = cache.update_and_fetch(k, v)
            self.assertTrue(mx.array_equal(k_up[..., idx : idx + 1, :], k))
            self.assertTrue(mx.array_equal(v_up[..., idx : idx + 1, :], v))
            self.assertEqual(cache.offset, 21 + i)
            idx += 1
            if idx >= 8:
                idx = 2

    def model_test_runner(self, model, model_type, vocab_size, num_layers):
        self.initialize_quantized_linear(model)

        self.assertEqual(len(model.layers), num_layers)
        self.assertEqual(model.model_type, model_type)

        for t in [mx.float32]:
            inputs = mx.array([[0, 1]])
            outputs = model(inputs)
            self.assertEqual(outputs.shape, (1, 2, vocab_size))
            self.assertEqual(outputs.dtype, t)

            kv_heads = (
                [model.n_kv_heads] * len(model.layers)
                if isinstance(model.n_kv_heads, int)
                else model.n_kv_heads
            )
            cache = [KVCache(model.head_dim, n) for n in kv_heads]

            outputs = model(inputs, cache)
            self.assertEqual(outputs.shape, (1, 2, vocab_size))
            self.assertEqual(outputs.dtype, t)

            outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache)
            self.assertEqual(outputs.shape, (1, 1, vocab_size))
            self.assertEqual(outputs.dtype, t)

    def test_llama(self):
        from gbx_lm.models import qllama

        args = qllama.ModelArgs(
            model_type="llama",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = qllama.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_phi3(self):
        from gbx_lm.models import qphi3

        args = qphi3.ModelArgs(
            model_type="phi3",
            hidden_size=3072,
            num_hidden_layers=32,
            intermediate_size=8192,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32064,
        )
        model = qphi3.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_gemma(self):
        from gbx_lm.models import qgemma

        args = qgemma.ModelArgs(
            model_type="gemma",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            head_dim=128,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
            num_key_value_heads=4,
        )
        model = qgemma.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_mixtral(self):
        from gbx_lm.models import qmixtral

        # Make a baby mixtral, because it will actually do the
        # eval
        args = qmixtral.ModelArgs(
            model_type="mixtral",
            vocab_size=100,
            hidden_size=1024,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts_per_tok=2,
            num_key_value_heads=2,
            num_local_experts=4,
        )
        model = qmixtral.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_qwen2(self):
        from gbx_lm.models import qqwen2

        args = qqwen2.ModelArgs(
            model_type="qwen2",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = qqwen2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_starcoder2(self):
        from gbx_lm.models import qstarcoder2

        args = qstarcoder2.ModelArgs(
            model_type="starcoder2",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        model = qstarcoder2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_llama3_1(self):
        from gbx_lm.models import qllama

        args = qllama.ModelArgs(
            model_type="llama",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
            max_position_embeddings=128,
            mlp_bias=False,
            num_key_value_heads=2,
            rope_scaling={
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        )
        model = qllama.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )


if __name__ == "__main__":
    unittest.main()
