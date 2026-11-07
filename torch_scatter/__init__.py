import torch

def _normalize_dim(dim, src):
    if dim < 0:
        dim = src.dim() + dim
    if dim != 0:
        raise NotImplementedError("Local torch_scatter fallback only supports dim=0.")
    return dim

def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
    _normalize_dim(dim, src)
    if index.dim() != 1:
        index = index.view(-1)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0

    out_shape = (dim_size,) + tuple(src.shape[1:])
    if reduce in {"sum", "add", "mean"}:
        out = src.new_zeros(out_shape)
        if index.numel():
            out.index_add_(0, index.to(src.device), src)
        if reduce == "mean":
            count = src.new_zeros((dim_size,))
            if index.numel():
                count.index_add_(0, index.to(src.device), src.new_ones(index.numel()))
            count = count.clamp_min_(1).view((dim_size,) + (1,) * (src.dim() - 1))
            out = out / count
        return out

    if reduce == "max":
        out = src.new_full(out_shape, -float("inf"))
        for row in range(src.size(0)):
            target = int(index[row].item())
            out[target] = torch.maximum(out[target], src[row])
        out[out == -float("inf")] = 0
        return out

    raise NotImplementedError(f"Unsupported scatter reduce: {reduce}")

def gather_csr(*args, **kwargs):
    raise NotImplementedError("Local torch_scatter fallback does not implement gather_csr.")

def segment_csr(*args, **kwargs):
    raise NotImplementedError("Local torch_scatter fallback does not implement segment_csr.")
