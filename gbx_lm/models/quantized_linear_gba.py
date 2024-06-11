import math

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.utils import tree_flatten, tree_map

import torch
import numpy as np

class QuantizedLinear(Module):
    """Applies an affine transformation to the input using a quantized weight matrix.

    It is the quantized equivalent of :class:`mlx.nn.Linear`. For now its
    parameters are frozen and will not be included in any gradient computation
    but this will probably change in the future.

    QuantizedLinear also provides two useful classmethods to convert linear
    layers to QuantizedLinear layers.

    - :meth:`from_linear` returns a QuantizedLinear layer that applies the same
      linear transformation up to the quantization error.
    - :meth:`quantize_module` swaps all the linear layers of the passed module
      with QuantizedLinear ones.

    Args:
        input_dims (int): The dimensionality of the input features
        output_dims (int): The dimensionality of the output features
        bias (bool, optional): If set to ``False`` then the layer will not use
            a bias. (default: True).
        group_size (int, optional): The group size to use for the quantized
            weight. See :func:`~mlx.core.quantize`. (default: 64)
        bits (int, optional): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. (default: 4)
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = False,
        group_size: int = 64,
        bits: int = 4,
    ):
        super().__init__()

        # Quantization config
        self.group_size = group_size
        self.bits = bits
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.double_group_size = 32

        # And bias if needed
        if bias:
            self.bias = mx.zeros((self.output_dims,))
        self.init_params(False, False)
        # Freeze this model's parameters
        self.freeze()

    def init_params(self, use_double_quantization: bool, use_q_perm: bool):
        shape_w = (self.output_dims, self.input_dims // 32 * self.bits)
        shape_sz = (self.output_dims, self.input_dims // self.group_size)

        self.qweight = mx.zeros(
            shape=shape_w,
            dtype=mx.uint32
        )

        self.channel_scale = mx.ones(
            shape=(1, 1, self.input_dims),
            dtype=mx.float16
        )

        if use_q_perm:
            self.q_perm = mx.zeros(self.input_dims, dtype=mx.int16)

        if use_double_quantization:
            shape_qstatistic = (
                math.ceil(self.input_dims / self.group_size),
                math.ceil(self.output_dims / self.double_group_size),
                self.double_group_size
            )
            shape_dsz = (
                math.ceil(self.input_dims / self.group_size),
                math.ceil(self.output_dims / self.double_group_size),
                1
            )
            # parameters only for conversion
            self.qstatistic = mx.zeros(
                shape=shape_qstatistic,
                dtype=mx.uint8
            )
            self.qzeros_zeros = mx.zeros(
                shape=shape_dsz,
                dtype=mx.float16
            )
            self.qzeros_scales = mx.ones(
                shape=shape_dsz,
                dtype=mx.float16
            )
            self.qscales_zeros = mx.zeros(
                shape=shape_dsz,
                dtype=mx.float16
            )
            self.qscales_scales = mx.ones(
                shape=shape_dsz,
                dtype=mx.float16
            )
        else:
            self.scales = mx.ones(
                shape=shape_sz,
                dtype=mx.float16
            )
            self.zeros = mx.zeros(
                shape=shape_sz,
                dtype=mx.float16
            )

    def create_scales_zeros(self):
        # cannot directly convert to torch, use numpy as intermediate buffer
        qstatistic_np = np.array(self.qstatistic)
        qzeros_zeros_np = np.array(self.qzeros_zeros)
        qzeros_scales_np = np.array(self.qzeros_scales)
        qscales_zeros_np = np.array(self.qscales_zeros)
        qscales_scales_np = np.array(self.qscales_scales)

        qstatistic_torch = torch.from_numpy(qstatistic_np)
        qzeros_zeros_torch = torch.from_numpy(qzeros_zeros_np)
        qzeros_scales_torch = torch.from_numpy(qzeros_scales_np)
        qscales_zeros_torch = torch.from_numpy(qscales_zeros_np)
        qscales_scales_torch = torch.from_numpy(qscales_scales_np)

        buffer_shape = (math.ceil(self.input_dims / self.group_size), self.output_dims)
        qscales_torch = (qstatistic_torch & 0xF0) >> 4
        qzeros_torch = qstatistic_torch & 0x0F

        zeros = ((qzeros_torch.to(torch.float16) - qzeros_zeros_torch) * qzeros_scales_torch).view(buffer_shape)
        scales = ((qscales_torch.to(torch.float16) - qscales_zeros_torch) * qscales_scales_torch).view(buffer_shape)

        # get mx array and adapt the layout according to qweight.shape
        self.scales = mx.array(scales.numpy()).transpose()
        self.zeros = mx.array(zeros.numpy()).transpose()

        # after creating fp16 scales and zeros, we release qscale* and qzero* parameters
        self['qstatistic'] = None
        self['qzeros_zeros'] = None
        self['qzeros_scales'] = None
        self['qscales_zeros'] = None
        self['qscales_scales'] = None

    def set_bias_and_weight(self):
        # There is a small error in the mlx document. Although the document indicates
        # q_weight * scale - zero, however '+' is used in the actual calculation.
        # Therefore, here we need to add the negative sign manually.
        self.zeros = -self.zeros

        # check if no q_perm Assignment we release it
        if hasattr(self, 'q_perm'):
            # to 3D in order to use mx.take_along_axis to re-arrange x's permutation in forward
            self.q_perm = self.q_perm.reshape(1, 1, -1)

    def unfreeze(self, *args, **kwargs):
        """Wrap unfreeze so that we unfreeze any layers we might contain but
        our parameters will remain frozen."""
        super().unfreeze(*args, **kwargs)
        self.freeze(recurse=False)

    def _extra_repr(self):
        in_dims, out_dims = self.qweight.shape
        in_dims *= 32 // self.bits

        assert self.input_dims == in_dims, f"input_dims mismatch of the quantized model."

        return (
            f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self}, "
            f"group_size={self.group_size}, bits={self.bits}"
        )

    def __call__(self, x):
        # mul channel_scale
        x = mx.multiply(x, self.channel_scale)
        # # array rearrangement if necessary
        if hasattr(self, 'q_perm'):
            x = mx.take_along_axis(x, self.q_perm, axis=-1)
        # quantized matmul
        x = mx.quantized_matmul(
            x,
            self.qweight,
            scales=self.scales,
            biases=self.zeros,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if "bias" in self:
            x = x + self.bias
        return x


    @classmethod
    def reinit_module(
        cls,
        model: Module,
        group_size: int = 64,
        bits: int = 4,
        strategy: dict = None,
        use_double_quantization: bool = False,
        use_q_perm: bool = False,
        gba_linear_class_predicate=lambda m: isinstance(m, QuantizedLinear),
    ):
        def _run_if_q_gba_linear(m):
            """
            update group size, bits and re-init params
            """
            if gba_linear_class_predicate(m):
                m.group_size = group_size
                m.bits = bits
                m.init_params(use_double_quantization, use_q_perm)
            return m

        def _assign_attributs(model: Module):
            """
            read bits from strategy and update QuantizedLinear layers
            """
            for name, child in model.named_modules():
                if isinstance(child, QuantizedLinear):
                    # read bits and group size from strategy
                    layer_number = name.split('.')[2]
                    strategy_per_block = strategy["model.layers.{}".format(layer_number)]

                    for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'qkv_proj', 'gate_up_proj']:
                        if key in name:
                            try:
                                strg = strategy_per_block[key]
                                break
                            except KeyError:
                                pass
                    child.bits = strg["bits"][0]
                    child.group_size = strg["group_size"][str(child.bits)]
                    assert child.group_size in [32, 64, 128], f"The group size value ({child.group_size}) must be 32, 64 or 128."
                    # print(f'[DEBUG]: {name}: Updated QuantizedLinear bits to {child.bits}, group_size to {child.group_size}')

                    # re-init params
                    child.init_params(use_double_quantization, use_q_perm)

        if strategy is None:
            leaves = model.leaf_modules()
            leaves = tree_map(_run_if_q_gba_linear, leaves, is_leaf=Module.is_module)
            model.update_modules(leaves)
        else:
            _assign_attributs(model)


    @classmethod
    def prepare_scales_zeros(
        cls,
        model: Module,
        gba_linear_class_predicate=lambda m: isinstance(m, QuantizedLinear),
    ):
        """Creates fp16 scales and zeros, releases double quantization parameters."""
        def _run_if_q_gba_linear(m):
            if gba_linear_class_predicate(m):
                m.create_scales_zeros()
            return m

        leaves = model.leaf_modules()
        leaves = tree_map(_run_if_q_gba_linear, leaves, is_leaf=Module.is_module)
        model.update_modules(leaves)


    @classmethod
    def post_processing_and_release(
        cls,
        model: Module,
        gba_linear_class_predicate=lambda m: isinstance(m, QuantizedLinear),
    ):
        """
        Changes all zeros to -zeros.

        mlx quantize method: wq = round((w - zeros) / scale)
        but its dequantize: w = wq * scsle - zeros
        Two functions are not correctly alighned in the documentation.
        After our investifation, we have to change zeros from GBA model to -zeros:
        """
        def _run_if_q_gba_linear(m):
            if gba_linear_class_predicate(m):
                m.set_bias_and_weight()
            return m

        leaves = model.leaf_modules()
        leaves = tree_map(_run_if_q_gba_linear, leaves, is_leaf=Module.is_module)
        model.update_modules(leaves)


    @classmethod
    def from_linear(cls, linear_layer: Module, group_size: int = 64, bits: int = 4, q_perm = None, channel_scale = None):
        """Create a QuantizedLinear layer from the parameters of a provided
        linear layer."""
        output_dims, input_dims = linear_layer.weight.shape
        ql = cls(input_dims, output_dims, False, group_size, bits)

        if q_perm is not None:
            q_perm = q_perm.reshape(-1, 1)
            weight = mx.take_along_axis(linear_layer.weight, q_perm, axis=0)
            ql.q_perm = q_perm.reshape(1, 1, -1) # prepare shape for inference

        if channel_scale is not None:
            ql.channel_scale = channel_scale

        ql.qweight, ql.scales, ql.zeros = mx.quantize(
            weight, group_size, bits
        )
        if "bias" in linear_layer:
            ql.bias = linear_layer.bias

        return ql