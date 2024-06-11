# Initial code base from https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm under the MIT License.
# Additional code from GreenBitAI is licensed under the Apache 2.0 License.


import math

import mlx.core as mx
import mlx.nn as nn

from gbx_lm.models.quantized_linear_gba import QuantizedLinear as GBA_QLinear
from ..models.switch_layers import QuantizedSwitchLinear, SwitchLinear


class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        scale: float = 10.0,
    ):
        # TODO remove when input_dims and output_dims are attributes
        # on linear and quantized linear
        if isinstance(linear, GBA_QLinear):
            output_dims, input_dims = linear.qweight.shape
            input_dims *= 32 // linear.bits
        elif isinstance(linear, (nn.Linear, nn.QuantizedLinear)):
            output_dims, input_dims = linear.weight.shape
            if isinstance(linear, nn.QuantizedLinear):
                input_dims *= 32 // linear.bits

        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            alpha=alpha,
            dropout=dropout,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def to_linear(self, de_quantize: bool = False):
        linear = self.linear
        bias = "bias" in linear

        # adapted for GBA_QLinear
        q_perm = None
        if isinstance(linear, GBA_QLinear):
            weight = linear.qweight
            if hasattr(linear, 'q_perm'):
                q_perm = linear.q_perm
        else:
            weight = linear.weight

        is_quantized = isinstance(linear, (nn.QuantizedLinear, GBA_QLinear))
        is_gba_quantized = isinstance(linear, GBA_QLinear)

        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype

        if is_quantized:
            dtype = mx.float16
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.zeros if is_gba_quantized else linear.biases,
                linear.group_size,
                linear.bits,
            )

            if isinstance(linear, GBA_QLinear):
                # rearrange weight
                if q_perm is not None:
                    q_perm = q_perm.reshape(-1)
                    weight_n = mx.zeros(weight.shape)
                    # torch scatter like
                    for orig_idx, new_idx in enumerate(q_perm):
                        weight_n[new_idx] = weight[orig_idx]
                    weight = weight_n

        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        lora_b = (self.scale * self.lora_b.T).astype(dtype)
        lora_a = self.lora_a.T.astype(dtype)
        fused_linear.weight = weight + lora_b @ lora_a

        if bias:
            fused_linear.bias = linear.bias

        if is_quantized and not de_quantize:
            QuantizedClass = GBA_QLinear if is_gba_quantized else nn.QuantizedLinear
            args = [fused_linear, linear.group_size, linear.bits]
            if is_gba_quantized:
                args.append(q_perm)
                args.append(linear.channel_scale)
            fused_linear = QuantizedClass.from_linear(*args)

        return fused_linear


    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        scale: float = 10.0,
        bias: bool = False,
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        # Scale for low-rank update
        self.scale = scale * (alpha / r)

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)


class LoRASwitchLinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Module,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        scale: float = 10.0,
    ):
        lora_lin = LoRASwitchLinear(
            input_dims=linear.input_dims,
            output_dims=linear.output_dims,
            num_experts=linear.num_experts,
            r=r,
            alpha=alpha,
            dropout=dropout,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def to_linear(self, de_quantize: bool = False):
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        is_quantized = isinstance(linear, QuantizedSwitchLinear)

        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype

        if is_quantized:
            dtype = mx.float16
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                linear.group_size,
                linear.bits,
            )
        num_experts, output_dims, input_dims = weight.shape
        fused_linear = SwitchLinear(input_dims, output_dims, num_experts, bias=bias)

        lora_b = (self.scale * self.lora_b).astype(dtype)
        lora_a = self.lora_a.reshape(num_experts, -1, input_dims).astype(dtype)
        fused_linear.weight = weight + lora_b @ lora_a
        if bias:
            fused_linear.bias = linear.bias

        if is_quantized and not de_quantize:
            fused_linear = fused_linear.to_quantized(linear.group_size, linear.bits)

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        scale: float = 10.0,
        bias: bool = False,
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = SwitchLinear(input_dims, output_dims, num_experts, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        # Scale for low-rank update
        self.scale = scale * (alpha / r)

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(r * num_experts, input_dims),
        )
        self.lora_b = mx.zeros(shape=(num_experts, output_dims, r))
        self.num_experts = num_experts

    def __call__(self, x, indices):
        shape = x.shape[:-3] + (self.num_experts, -1)

        y = self.linear(x, indices)
        z = (self.dropout(x) @ self.lora_a.T).reshape(shape)
        z = mx.take_along_axis(z, indices[..., None], axis=-2)
        z = z[..., None, :] @ self.lora_b[indices].swapaxes(-2, -1)

        return y + (self.scale * z).astype(x.dtype)
