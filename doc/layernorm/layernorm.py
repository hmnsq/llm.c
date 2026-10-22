استورد torch

eps = 1e-5

صنف LayerNorm:

    @staticmethod
    تعریف forward(x, w, b):
        B, T, C = x.ناپ()
        اوسط = x.جوڑ(-1, keepdim=سچ) / C # B,T,1
        xshift = x - اوسط # B,T,C
        var = (xshift**2).جوڑ(-1, keepdim=سچ) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        باہرلا = norm * w + b # B,T,C

        cache = (x, w, اوسط, rstd)
        أرجع باہرلا, cache

    @staticmethod
    تعریف backward(dout, cache):
        x, w, اوسط, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - اوسط) * rstd
        # gradients for weights, bias
        db = dout.جوڑ((0, 1))
        dw = (dout * norm).جوڑ((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.اوسط(-1, keepdim=سچ) - norm * (dnorm * norm).اوسط(-1, keepdim=سچ)
        dx *= rstd
        أرجع dx, dw, db

# create a small dummy example and check w.r.t PyTorch backward
B = 2
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=سچ)
w = torch.randn(C, requires_grad=سچ)
b = torch.randn(C, requires_grad=سچ)
باہرلا, cache = LayerNorm.forward(x, w, b)

dout = torch.randn(B, T, C)
dx, dw, db = LayerNorm.backward(dout, cache)

# compare to PyTorch autograd
fakeloss = (باہرلا * dout).جوڑ()
fakeloss.backward()
چھاپو("dx error:", (x.grad - dx).abs().max().شے())
چھاپو("dw error:", (w.grad - dw).abs().max().شے())
چھاپو("db error:", (b.grad - db).abs().max().شے())

# for reference checking in C also
x, w, اوسط, rstd = cache

تعریف اكتب(tensor, handle):
    handle.اكتب(tensor.detach().numpy().astype("float32").tobytes())

# Write to file
نال open('ln.bin', 'wb') كـ file:
    اكتب(x, file) # (B, T, C)
    اكتب(w, file) # (C, )
    اكتب(b, file) # (C, )
    اكتب(باہرلا, file) # (B, T, C)
    اكتب(اوسط, file) # (B, T)
    اكتب(rstd, file) # (B, T)
    اكتب(dout, file) # (B, T, C)
    اكتب(dx, file) # (B, T, C)
    اكتب(dw, file) # (C, )
    اكتب(db, file) # (C, )
