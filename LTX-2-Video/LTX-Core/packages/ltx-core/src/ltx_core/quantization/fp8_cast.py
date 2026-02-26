import torch

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.model.transformer.model import LTXModel

BLOCK_SIZE = 1024


def calculate_weight_float8(target_weights: torch.Tensor, original_weights: torch.Tensor) -> torch.Tensor:
    result = _fused_add_round_launch(target_weights, original_weights, seed=0).to(target_weights.dtype)
    target_weights.copy_(result, non_blocking=True)
    return target_weights


def _fused_add_round_launch(target_weight: torch.Tensor, original_weight: torch.Tensor, seed: int) -> torch.Tensor:
    # Lazy import triton - only available on CUDA platforms
    import triton  # noqa: PLC0415

    from ltx_core.loader.kernels import fused_add_round_kernel  # noqa: PLC0415

    if original_weight.dtype == torch.float8_e4m3fn:
        exponent_bits, mantissa_bits, exponent_bias = 4, 3, 7
    elif original_weight.dtype == torch.float8_e5m2:
        exponent_bits, mantissa_bits, exponent_bias = 5, 2, 15  # noqa: F841
    else:
        raise ValueError("Unsupported dtype")

    if target_weight.dtype != torch.bfloat16:
        raise ValueError("target_weight dtype must be bfloat16")

    # Calculate grid and block sizes
    n_elements = original_weight.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    fused_add_round_kernel[grid](
        original_weight,
        target_weight,
        seed,
        n_elements,
        exponent_bias,
        mantissa_bits,
        BLOCK_SIZE,
    )
    return target_weight


def _naive_weight_or_bias_downcast(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    """
    Downcast the weight or bias to the float8_e4m3fn dtype.
    """
    return [KeyValueOperationResult(key, value.to(dtype=torch.float8_e4m3fn))]


def _upcast_and_round(
    weight: torch.Tensor, dtype: torch.dtype, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.Tensor:
    """
    Upcast the weight to the given dtype and optionally apply stochastic rounding.
    Input weight needs to have float8_e4m3fn or float8_e5m2 dtype.
    """
    if not with_stochastic_rounding:
        return weight.to(dtype)
    return _fused_add_round_launch(torch.zeros_like(weight, dtype=dtype), weight, seed)


def _replace_fwd_with_upcast(layer: torch.nn.Linear, with_stochastic_rounding: bool = False, seed: int = 0) -> None:
    """
    Replace linear.forward and rms_norm.forward with a version that:
      - upcasts weight and bias to input's dtype
      - if the layer has a weight_scale (for FP8+scale checkpoints), applies it
      - returns F.linear or F.rms_norm calculated in that dtype
    """

    layer.original_forward = layer.forward

    def new_linear_forward(*args, **_kwargs) -> torch.Tensor:
        # assume first arg is the input tensor
        x = args[0]
        w_up = _upcast_and_round(layer.weight, x.dtype, with_stochastic_rounding, seed)
        # HOTFIX: LTX-2/FP8 checkpoints store a per-tensor weight_scale alongside weights.
        # Applying this scale restores the correct numerical range that was lost during FP8 quantization.
        # Without this, the model produces pure noise even though FP8 bits are loaded correctly.
        if hasattr(layer, 'weight_scale') and layer.weight_scale is not None:
            scale = layer.weight_scale
            if scale.dtype != x.dtype:
                scale = scale.to(x.dtype)
            w_up = w_up * scale
        b_up = None

        if layer.bias is not None:
            b_up = _upcast_and_round(layer.bias, x.dtype, with_stochastic_rounding, seed)

        return torch.nn.functional.linear(x, w_up, b_up)

    layer.forward = new_linear_forward


def _amend_forward_with_upcast(
    model: torch.nn.Module, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.nn.Module:
    """
    Replace the forward method of the model's Linear layers to forward
    with upcast and optional stochastic rounding.

    NOTE: For FP8 scaled checkpoints (LTX-2 19B), call `inject_weight_scales(model, raw_sd)`
    AFTER load_state_dict to register weight_scale buffers before running inference.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            _replace_fwd_with_upcast(m, with_stochastic_rounding, seed)
    return model


def inject_weight_scales(model: torch.nn.Module, raw_state_dict: dict[str, torch.Tensor]) -> None:
    """
    Register per-tensor weight_scale factors from a raw safetensors state dict as non-persistent
    buffers on the corresponding nn.Linear modules.

    This MUST be called AFTER load_state_dict since load_state_dict silently discards keys
    that have no matching parameter/buffer in the model (weight_scale is not a registered
    nn.Linear attribute). This function retrofits them as buffers so the patched forward()
    in _replace_fwd_with_upcast can access them.

    Call pattern:
        model.load_state_dict(sd, strict=False, assign=True)
        inject_weight_scales(model, raw_sd)  # <-- call this immediately after
    """
    scale_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        scale_key = f"{name}.weight_scale"
        if scale_key not in raw_state_dict:
            continue
        if hasattr(module, "weight_scale"):
            continue  # already registered
        scale = raw_state_dict[scale_key]
        module.register_buffer("weight_scale", scale, persistent=False)
        scale_count += 1
    if scale_count:
        import logging
        logging.getLogger(__name__).info(f"[fp8_cast] Registered {scale_count} weight_scale buffers for FP8 inference.")


TRANSFORMER_LINEAR_DOWNCAST_MAP = (
    SDOps("TRANSFORMER_LINEAR_DOWNCAST_MAP")
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_q.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_q.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_k.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_k.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_v.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_v.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_out.0.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_out.0.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".ff.net.0.proj.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".ff.net.0.proj.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".ff.net.2.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".ff.net.2.bias", operation=_naive_weight_or_bias_downcast
    )
)

UPCAST_DURING_INFERENCE = ModuleOps(
    name="upcast_fp8_during_linear_forward",
    matcher=lambda model: isinstance(model, LTXModel),
    mutator=lambda model: _amend_forward_with_upcast(model, False),
)


class UpcastWithStochasticRounding(ModuleOps):
    """
    ModuleOps for upcasting the model's float8_e4m3fn weights and biases to the bfloat16 dtype
    and applying stochastic rounding during linear forward.
    """

    def __new__(cls, seed: int = 0):
        return super().__new__(
            cls,
            name="upcast_fp8_during_linear_forward_with_stochastic_rounding",
            matcher=lambda model: isinstance(model, LTXModel),
            mutator=lambda model: _amend_forward_with_upcast(model, True, seed),
        )
