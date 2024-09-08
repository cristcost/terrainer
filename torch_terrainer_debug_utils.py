def dbg_print(label, a, b=None):
    if isinstance(a, torch.Tensor):
        a = a.numpy()
    if isinstance(b, torch.Tensor):
        b = b.numpy()

    if b is not None:
        assert a.shape == b.shape, f"Shapes do not match: {a.shape} != {b.shape}"
        assert a.dtype == b.dtype, f"DType do not match: {a.dtype} != {b.dtype}"
        delta = a - b
        delta_str = "∆"
    else:
        delta = a
        delta_str = " "

    print(
        f"{label} dtype: {a.dtype} shapes: {a.shape} {delta_str} min: {delta.min()} {delta_str} max: {delta.max()} {delta_str} mean: {delta.mean()} {delta_str} std: {delta.std()}"
    )