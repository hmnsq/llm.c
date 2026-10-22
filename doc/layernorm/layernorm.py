ഇറക്കുമതി torch

eps = 1e-5

വർഗ്ഗം LayerNorm:

    @staticmethod
    def forward(x, w, b):
        B, T, C = x.size()
        mean = x.sum(-1, keepdim=സത്യം) / C # B,T,1
        xshift = x - mean # B,T,C
        var = (xshift**2).sum(-1, keepdim=സത്യം) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        out = norm * w + b # B,T,C

        cache = (x, w, mean, rstd)
        തിരികെ out, cache

    @staticmethod
    def backward(dout, cache):
        x, w, mean, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - mean) * rstd
        # gradients for weights, bias
        db = dout.sum((0, 1))
        dw = (dout * norm).sum((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.mean(-1, keepdim=സത്യം) - norm * (dnorm * norm).mean(-1, keepdim=സത്യം)
        dx *= rstd
        തിരികെ dx, dw, db

# create a small dummy example and check w.r.t PyTorch backward
B = 2
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=സത്യം)
w = torch.randn(C, requires_grad=സത്യം)
b = torch.randn(C, requires_grad=സത്യം)
out, cache = LayerNorm.forward(x, w, b)

dout = torch.randn(B, T, C)
dx, dw, db = LayerNorm.backward(dout, cache)

# compare to PyTorch autograd
fakeloss = (out * dout).sum()
fakeloss.backward()
print("dx error:", (x.grad - dx).abs().max().item())
print("dw error:", (w.grad - dw).abs().max().item())
print("db error:", (b.grad - db).abs().max().item())

# for reference checking in C also
x, w, mean, rstd = cache

def ലിഖ(tensor, handle):
    handle.ലിഖ(tensor.detach().numpy().astype("float32").tobytes())

# Write to file
with open('ln.bin', 'wb') ഇവ file:
    ലിഖ(x, file) # (B, T, C)
    ലിഖ(w, file) # (C, )
    ലിഖ(b, file) # (C, )
    ലിഖ(out, file) # (B, T, C)
    ലിഖ(mean, file) # (B, T)
    ലിഖ(rstd, file) # (B, T)
    ലിഖ(dout, file) # (B, T, C)
    ലിഖ(dx, file) # (B, T, C)
    ലിഖ(dw, file) # (C, )
    ലിഖ(db, file) # (C, )
