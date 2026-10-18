# --- fused quant+GEMV fast-path (night experiment §13, docs/TRITON_KERNELS.md) ---
# Rebind dua entry point block-fp8 linear (flashinfer = auto di sm120 +
# flashinfer installed; cutlass = explicit backend). M==1 decode bs=1 →
# fused kernel (quant+GEMV 1 launch); sisanya jatuh ke original.
# Import fused_tm lazy di call pertama (hindari circular import + kompile
# Triton nunggu step decode pertama). try/except: gagal = fallback orig.
_cutlass_orig_13 = cutlass_w8a8_block_fp8_linear_with_fallback
_flashinfer_orig_13 = flashinfer_gemm_w8a8_block_fp8_linear_with_fallback


def _try_fused_m1_13(input, weight, weight_scale, bias):
    x2d = input.view(-1, input.shape[-1])
    if (x2d.shape[0] != 1 or input.dtype != torch.bfloat16
            or weight.shape[0] % 128 != 0 or weight.shape[1] % 128 != 0):
        return None
    try:
        import sys as _s
        if "/root" not in _s.path:
            _s.path.insert(0, "/root")
        from fused_gemv_quant_tm import fused_gemv_quant as _fgvq
        out = _fgvq(x2d, weight, weight_scale)  # (1, N) bf16
        if bias is not None:
            out = out + bias
        return out.view(*input.shape[:-1], weight.shape[0])
    except Exception:
        return None


def cutlass_w8a8_block_fp8_linear_with_fallback(
    input=None, weight=None, block_size=None, weight_scale=None,
    input_scale=None, bias=None, **kw,
):
    if input_scale is None:
        out = _try_fused_m1_13(input, weight, weight_scale, bias)
        if out is not None:
            return out
    return _cutlass_orig_13(input, weight, block_size, weight_scale,
                             input_scale, bias)


def flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(
    input=None, weight=None, block_size=None, weight_scale=None,
    input_scale=None, bias=None, **kw,
):
    if input_scale is None:
        out = _try_fused_m1_13(input, weight, weight_scale, bias)
        if out is not None:
            return out
    return _flashinfer_orig_13(input, weight, block_size, weight_scale,
                                input_scale, bias)
