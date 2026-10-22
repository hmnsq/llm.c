درآمد torch

eps = 1e-5

جماعت LayerNorm:

    @staticmethod
    تعریف forward(x, w, b):
        B, T, C = x.سائز()
        اوسط = x.مجموع(-1, keepdim=سچ) / C # B,T,1
        xshift = x - اوسط # B,T,C
        var = (xshift**2).مجموع(-1, keepdim=سچ) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        برآمد = norm * w + b # B,T,C

        cache = (x, w, اوسط, rstd)
        واپس برآمد, cache

    @staticmethod
    تعریف backward(dout, cache):
        x, w, اوسط, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - اوسط) * rstd
        # gradients for weights, bias
        db = dout.مجموع((0, 1))
        dw = (dout * norm).مجموع((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.اوسط(-1, keepdim=سچ) - norm * (dnorm * norm).اوسط(-1, keepdim=سچ)
        dx *= rstd
        واپس dx, dw, db

# create a small dummy example and check w.r.t PyTorch backward
B = 2
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=سچ)
w = torch.randn(C, requires_grad=سچ)
b = torch.randn(C, requires_grad=سچ)
برآمد, cache = LayerNorm.forward(x, w, b)

dout = torch.randn(B, T, C)
dx, dw, db = LayerNorm.backward(dout, cache)

# compare to PyTorch autograd
fakeloss = (برآمد * dout).مجموع()
fakeloss.backward()
چھاپو("dx error:", (x.grad - dx).abs().max().شے_2())
چھاپو("dw error:", (w.grad - dw).abs().max().شے_2())
چھاپو("db error:", (b.grad - db).abs().max().شے_2())

# for reference checking in C also
x, w, اوسط, rstd = cache

تعریف اكتب(tensor, handle):
    handle.اكتب(tensor.detach().numpy().astype("float32").tobytes())

# Write to file
ساتھ open('ln.bin', 'wb') بطور file:
    اكتب(x, file) # (B, T, C)
    اكتب(w, file) # (C, )
    اكتب(b, file) # (C, )
    اكتب(برآمد, file) # (B, T, C)
    اكتب(اوسط, file) # (B, T)
    اكتب(rstd, file) # (B, T)
    اكتب(dout, file) # (B, T, C)
    اكتب(dx, file) # (B, T, C)
    اكتب(dw, file) # (C, )
    اكتب(db, file) # (C, )
