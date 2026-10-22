ಆಮದು torch

eps = 1e-5

ವರ್ಗ LayerNorm:

    @staticmethod
    ವ್ಯಾಖ್ಯಾನ forward(x, w, b):
        B, T, C = x.ಗಾತ್ರ_2()
        ಸರಾಸರಿ = x.ಮೊತ್ತ(-1, keepdim=ನಿಜ) / C # B,T,1
        xshift = x - ಸರಾಸರಿ # B,T,C
        var = (xshift**2).ಮೊತ್ತ(-1, keepdim=ನಿಜ) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        ಹೊರಹೋಗು = norm * w + b # B,T,C

        cache = (x, w, ಸರಾಸರಿ, rstd)
        ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು, cache

    @staticmethod
    ವ್ಯಾಖ್ಯಾನ backward(dout, cache):
        x, w, ಸರಾಸರಿ, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - ಸರಾಸರಿ) * rstd
        # gradients for weights, bias
        db = dout.ಮೊತ್ತ((0, 1))
        dw = (dout * norm).ಮೊತ್ತ((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.ಸರಾಸರಿ(-1, keepdim=ನಿಜ) - norm * (dnorm * norm).ಸರಾಸರಿ(-1, keepdim=ನಿಜ)
        dx *= rstd
        ಹಿಂತಿರುಗಿ dx, dw, db

# create a small dummy example and check w.r.t PyTorch backward
B = 2
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=ನಿಜ)
w = torch.randn(C, requires_grad=ನಿಜ)
b = torch.randn(C, requires_grad=ನಿಜ)
ಹೊರಹೋಗು, cache = LayerNorm.forward(x, w, b)

dout = torch.randn(B, T, C)
dx, dw, db = LayerNorm.backward(dout, cache)

# compare to PyTorch autograd
fakeloss = (ಹೊರಹೋಗು * dout).ಮೊತ್ತ()
fakeloss.backward()
ಮುದ್ರಿಸು("dx error:", (x.grad - dx).abs().max().ಅಂಶ())
ಮುದ್ರಿಸು("dw error:", (w.grad - dw).abs().max().ಅಂಶ())
ಮುದ್ರಿಸು("db error:", (b.grad - db).abs().max().ಅಂಶ())

# for reference checking in C also
x, w, ಸರಾಸರಿ, rstd = cache

ವ್ಯಾಖ್ಯಾನ ಲಿಖ(tensor, handle):
    handle.ಲಿಖ(tensor.detach().numpy().astype("float32").tobytes())

# Write to file
ಜೊತೆ open('ln.bin', 'wb') ಇವ file:
    ಲಿಖ(x, file) # (B, T, C)
    ಲಿಖ(w, file) # (C, )
    ಲಿಖ(b, file) # (C, )
    ಲಿಖ(ಹೊರಹೋಗು, file) # (B, T, C)
    ಲಿಖ(ಸರಾಸರಿ, file) # (B, T)
    ಲಿಖ(rstd, file) # (B, T)
    ಲಿಖ(dout, file) # (B, T, C)
    ಲಿಖ(dx, file) # (B, T, C)
    ಲಿಖ(dw, file) # (C, )
    ಲಿಖ(db, file) # (C, )
