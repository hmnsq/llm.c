استورد torch

eps = 1e-5

صنف LayerNorm:

    @staticmethod
    تعریف forward(x, w, b):
        B, T, C = x.اندازه()
        میانگین = x.جمع(-1, keepdim=درست) / C # B,T,1
        xshift = x - میانگین # B,T,C
        var = (xshift**2).جمع(-1, keepdim=درست) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        بیرون = norm * w + b # B,T,C

        cache = (x, w, میانگین, rstd)
        أرجع بیرون, cache

    @staticmethod
    تعریف backward(dout, cache):
        x, w, میانگین, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - میانگین) * rstd
        # gradients for weights, bias
        db = dout.جمع((0, 1))
        dw = (dout * norm).جمع((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.میانگین(-1, keepdim=درست) - norm * (dnorm * norm).میانگین(-1, keepdim=درست)
        dx *= rstd
        أرجع dx, dw, db

# create a small dummy example and check w.r.t PyTorch backward
B = 2
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=درست)
w = torch.randn(C, requires_grad=درست)
b = torch.randn(C, requires_grad=درست)
بیرون, cache = LayerNorm.forward(x, w, b)

dout = torch.randn(B, T, C)
dx, dw, db = LayerNorm.backward(dout, cache)

# compare to PyTorch autograd
fakeloss = (بیرون * dout).جمع()
fakeloss.backward()
چاپ("dx error:", (x.grad - dx).abs().max().قلم())
چاپ("dw error:", (w.grad - dw).abs().max().قلم())
چاپ("db error:", (b.grad - db).abs().max().قلم())

# for reference checking in C also
x, w, میانگین, rstd = cache

تعریف اكتب(tensor, handle):
    handle.اكتب(tensor.detach().numpy().astype("float32").tobytes())

# Write to file
با open('ln.bin', 'wb') كـ file:
    اكتب(x, file) # (B, T, C)
    اكتب(w, file) # (C, )
    اكتب(b, file) # (C, )
    اكتب(بیرون, file) # (B, T, C)
    اكتب(میانگین, file) # (B, T)
    اكتب(rstd, file) # (B, T)
    اكتب(dout, file) # (B, T, C)
    اكتب(dx, file) # (B, T, C)
    اكتب(dw, file) # (C, )
    اكتب(db, file) # (C, )
