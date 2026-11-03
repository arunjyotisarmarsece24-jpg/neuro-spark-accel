# okc: problem=muse_bf16_proj
from typing import TypedDict
import torch

try:
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
    HAS_SGLANG = True
except Exception:
    HAS_SGLANG = False
    _bf16_gemm_dispatch_impl = None


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


ROUTER_FP32_MAX_TOKENS = 8


def prepare(weights: Weights):
    state = {}
    for name, w in weights.items():
        w_c = w.contiguous()
        state[name] = w_c
        state[f"{name}_t"] = w_c.t().contiguous()

    unpooler = weights["unpooler"]
    state["unpooler_1024"] = unpooler[:1024].contiguous()
    state["unpooler_1024_t"] = unpooler[:1024].t().contiguous()
    state["unpooler_8192"] = unpooler.contiguous()
    state["unpooler_8192_t"] = unpooler.t().contiguous()

    router = weights["router"]
    state["router_t"] = router.t().contiguous()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()

    return state


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer = cfg["layer"]

    if layer == "router":
        if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
            return torch.mm(x.float(), state["router_t"])
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc

    if layer == "unpooler":
        n = cfg["n"]
        if n == 1024:
            return torch.mm(x, state["unpooler_1024_t"])
        elif n == 8192:
            return torch.mm(x, state["unpooler_8192_t"])
        return torch.mm(x, state["unpooler_t"][:, :n])

    return torch.mm(x, state[f"{layer}_t"])
