استورد torch

eps = 1e-5

صنف LayerNorm:

    @staticmethod
    تعریف forward(x, w, b):
        B, T, C = x.اندازه()
        منځنۍ = x.مجموعه(-1, keepdim=ریښتیا) / C # B,T,1
        xshift = x - منځنۍ # B,T,C
        var = (xshift**2).مجموعه(-1, keepdim=ریښتیا) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        بهر = norm * w + b # B,T,C

        cache = (x, w, منځنۍ, rstd)
        أرجع بهر, cache

    @staticmethod
    تعریف backward(dout, cache):
        x, w, منځنۍ, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - منځنۍ) * rstd
        # gradients for weights, bias
        db = dout.مجموعه((0, 1))
        dw = (dout * norm).مجموعه((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.منځنۍ(-1, keepdim=ریښتیا) - norm * (dnorm * norm).منځنۍ(-1, keepdim=ریښتیا)
        dx *= rstd
        أرجع dx, dw, db

# create a small dummy example and check w.r.t PyTorch backward
B = 2
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=ریښتیا)
w = torch.randn(C, requires_grad=ریښتیا)
b = torch.randn(C, requires_grad=ریښتیا)
بهر, cache = LayerNorm.forward(x, w, b)

dout = torch.randn(B, T, C)
dx, dw, db = LayerNorm.backward(dout, cache)

# compare to PyTorch autograd
fakeloss = (بهر * dout).مجموعه()
fakeloss.backward()
چاپ("dx error:", (x.grad - dx).abs().max().توکی())
چاپ("dw error:", (w.grad - dw).abs().max().توکی())
چاپ("db error:", (b.grad - db).abs().max().توکی())

# for reference checking in C also
x, w, منځنۍ, rstd = cache

تعریف اكتب(tensor, handle):
    handle.اكتب(tensor.detach().numpy().astype("float32").tobytes())

# Write to file
سره open('ln.bin', 'wb') كـ file:
    اكتب(x, file) # (B, T, C)
    اكتب(w, file) # (C, )
    اكتب(b, file) # (C, )
    اكتب(بهر, file) # (B, T, C)
    اكتب(منځنۍ, file) # (B, T)
    اكتب(rstd, file) # (B, T)
    اكتب(dout, file) # (B, T, C)
    اكتب(dx, file) # (B, T, C)
    اكتب(dw, file) # (C, )
    اكتب(db, file) # (C, )
