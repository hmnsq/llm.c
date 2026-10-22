ਆਯਾਤ torch

eps = 1e-5

ਵਰਗ LayerNorm:

    @staticmethod
    ਪਰਿਭਾਸ਼ਾ forward(x, w, b):
        B, T, C = x.ਮਾਪ()
        ਔਸਤ = x.ਜੋੜ(-1, keepdim=ਸੱਚ) / C # B,T,1
        xshift = x - ਔਸਤ # B,T,C
        var = (xshift**2).ਜੋੜ(-1, keepdim=ਸੱਚ) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        ਬਾਹਰਲਾ = norm * w + b # B,T,C

        cache = (x, w, ਔਸਤ, rstd)
        ਵਾਪਸ ਬਾਹਰਲਾ, cache

    @staticmethod
    ਪਰਿਭਾਸ਼ਾ backward(dout, cache):
        x, w, ਔਸਤ, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - ਔਸਤ) * rstd
        # gradients for weights, bias
        db = dout.ਜੋੜ((0, 1))
        dw = (dout * norm).ਜੋੜ((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.ਔਸਤ(-1, keepdim=ਸੱਚ) - norm * (dnorm * norm).ਔਸਤ(-1, keepdim=ਸੱਚ)
        dx *= rstd
        ਵਾਪਸ dx, dw, db

# create a small dummy example and check w.r.t PyTorch backward
B = 2
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=ਸੱਚ)
w = torch.randn(C, requires_grad=ਸੱਚ)
b = torch.randn(C, requires_grad=ਸੱਚ)
ਬਾਹਰਲਾ, cache = LayerNorm.forward(x, w, b)

dout = torch.randn(B, T, C)
dx, dw, db = LayerNorm.backward(dout, cache)

# compare to PyTorch autograd
fakeloss = (ਬਾਹਰਲਾ * dout).ਜੋੜ()
fakeloss.backward()
ਛਾਪੋ("dx error:", (x.grad - dx).abs().max().ਵਸਤੂ_2())
ਛਾਪੋ("dw error:", (w.grad - dw).abs().max().ਵਸਤੂ_2())
ਛਾਪੋ("db error:", (b.grad - db).abs().max().ਵਸਤੂ_2())

# for reference checking in C also
x, w, ਔਸਤ, rstd = cache

ਪਰਿਭਾਸ਼ਾ ਲਿਖ(tensor, handle):
    handle.ਲਿਖ(tensor.detach().numpy().astype("float32").tobytes())

# Write to file
ਨਾਲ open('ln.bin', 'wb') ਇਵ file:
    ਲਿਖ(x, file) # (B, T, C)
    ਲਿਖ(w, file) # (C, )
    ਲਿਖ(b, file) # (C, )
    ਲਿਖ(ਬਾਹਰਲਾ, file) # (B, T, C)
    ਲਿਖ(ਔਸਤ, file) # (B, T)
    ਲਿਖ(rstd, file) # (B, T)
    ਲਿਖ(dout, file) # (B, T, C)
    ਲਿਖ(dx, file) # (B, T, C)
    ਲਿਖ(dw, file) # (C, )
    ਲਿਖ(db, file) # (C, )
