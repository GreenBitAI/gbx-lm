# Initial code base from https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm under the MIT License.
# Additional code from GreenBitAI is licensed under the Apache 2.0 License.

import math

import mlx.core as mx
import mlx.nn as nn

# TODO: convert to gba_quantized linear layer
class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
    ):
        super().__init__()

        self.scale = math.sqrt(1 / input_dims)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.group_size = group_size
        self.bits = bits
        self.num_experts = num_experts
        self.init_params()
        # Freeze this model's parameters
        self.freeze()
    	if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    def init_params(self):
        self.qweight, self.scales, self.zeros = mx.quantize(
            mx.random.uniform(
                low=-self.scale,
                high=self.scale,
                shape=(self.num_experts, self.output_dims, self.input_dims),
            ),
            group_size=self.group_size,
            bits=self.bits,
        )
        
    def unfreeze(self, *args, **kwargs):
        """Wrap unfreeze so that we unfreeze any layers we might contain but
        our parameters will remain frozen."""
        super().unfreeze(*args, **kwargs)
        self.freeze(recurse=False)
    
    def __call__(self, x, indices):
        x = mx.gather_qmm(
            x,
            self["qweight"],
            self["scales"],
            self["zeros"],
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices):
        x = mx.gather_mm(x, self["weight"].swapaxes(-1, -2), rhs_indices=indices)
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims, output_dims, num_experts, False, group_size, bits
        )
        ql.qweight, ql.scales, ql.zeros = mx.quantize(self.weight, group_size, bits)
        if "bias" in self:
            ql.bias = self.bias

        return ql


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.silu,
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = QuantizedSwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = QuantizedSwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = QuantizedSwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        x_up = self.up_proj(x, indices)
        x_gate = self.gate_proj(x, indices)
        x = self.down_proj(self.activation(x_gate) * x_up, indices)

        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.gelu_approx,
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        x = self.fc1(x, indices)
        x = self.activation(x)
        x = self.fc2(x, indices)

        return x.squeeze(-2)
