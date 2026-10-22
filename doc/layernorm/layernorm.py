ಆನಯನ torch

eps = 1e-5

ವರ್ಗ LayerNorm:

    @staticmethod
    def forward(x, w, b):
        B, T, C = x.size()
        mean = x.sum(-1, keepdim=ಸತ್ಯ) / C # B,T,1
        xshift = x - mean # B,T,C
        var = (xshift**2).sum(-1, keepdim=ಸತ್ಯ) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        out = norm * w + b # B,T,C

        cache = (x, w, mean, rstd)
        ಪ್ರತ್ಯಾಗಮ out, cache

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
        dx = dnorm - dnorm.mean(-1, keepdim=ಸತ್ಯ) - norm * (dnorm * norm).mean(-1, keepdim=ಸತ್ಯ)
        dx *= rstd
        ಪ್ರತ್ಯಾಗಮ dx, dw, db

# create a small dummy example and check w.r.t PyTorch backward
B = 2
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=ಸತ್ಯ)
w = torch.randn(C, requires_grad=ಸತ್ಯ)
b = torch.randn(C, requires_grad=ಸತ್ಯ)
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

def ಲಿಖ(tensor, handle):
    handle.ಲಿಖ(tensor.detach().numpy().astype("float32").tobytes())

# Write to file
with open('ln.bin', 'wb') ಇವ file:
    ಲಿಖ(x, file) # (B, T, C)
    ಲಿಖ(w, file) # (C, )
    ಲಿಖ(b, file) # (C, )
    ಲಿಖ(out, file) # (B, T, C)
    ಲಿಖ(mean, file) # (B, T)
    ಲಿಖ(rstd, file) # (B, T)
    ಲಿಖ(dout, file) # (B, T, C)
    ಲಿಖ(dx, file) # (B, T, C)
    ಲಿಖ(dw, file) # (C, )
    ಲಿಖ(db, file) # (C, )
