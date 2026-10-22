استورد torch

eps = 1e-5

صنف LayerNorm:

    @staticmethod
    وصف forward(x, w, b):
        B, T, C = x.سائيز()
        سراسري = x.جوڙ(-1, keepdim=سچ) / C # B,T,1
        xshift = x - سراسري # B,T,C
        var = (xshift**2).جوڙ(-1, keepdim=سچ) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        ٻاهريون = norm * w + b # B,T,C

        cache = (x, w, سراسري, rstd)
        أرجع ٻاهريون, cache

    @staticmethod
    وصف backward(dout, cache):
        x, w, سراسري, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - سراسري) * rstd
        # gradients for weights, bias
        db = dout.جوڙ((0, 1))
        dw = (dout * norm).جوڙ((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.سراسري(-1, keepdim=سچ) - norm * (dnorm * norm).سراسري(-1, keepdim=سچ)
        dx *= rstd
        أرجع dx, dw, db

# create a small dummy example and check w.r.t PyTorch backward
B = 2
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=سچ)
w = torch.randn(C, requires_grad=سچ)
b = torch.randn(C, requires_grad=سچ)
ٻاهريون, cache = LayerNorm.forward(x, w, b)

dout = torch.randn(B, T, C)
dx, dw, db = LayerNorm.backward(dout, cache)

# compare to PyTorch autograd
fakeloss = (ٻاهريون * dout).جوڙ()
fakeloss.backward()
ڇاپيو("dx error:", (x.grad - dx).abs().max().شي())
ڇاپيو("dw error:", (w.grad - dw).abs().max().شي())
ڇاپيو("db error:", (b.grad - db).abs().max().شي())

# for reference checking in C also
x, w, سراسري, rstd = cache

وصف اكتب(tensor, handle):
    handle.اكتب(tensor.detach().numpy().astype("float32").tobytes())

# Write to file
سان open('ln.bin', 'wb') كـ file:
    اكتب(x, file) # (B, T, C)
    اكتب(w, file) # (C, )
    اكتب(b, file) # (C, )
    اكتب(ٻاهريون, file) # (B, T, C)
    اكتب(سراسري, file) # (B, T)
    اكتب(rstd, file) # (B, T)
    اكتب(dout, file) # (B, T, C)
    اكتب(dx, file) # (B, T, C)
    اكتب(dw, file) # (C, )
    اكتب(db, file) # (C, )
