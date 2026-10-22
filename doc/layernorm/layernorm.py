దిగుమతి torch

eps = 1e-5

తరగతి LayerNorm:

    @staticmethod
    నిర్వచనం forward(x, w, b):
        B, T, C = x.కొలత()
        సగటు = x.కూడిక(-1, keepdim=నిజం) / C # B,T,1
        xshift = x - సగటు # B,T,C
        var = (xshift**2).కూడిక(-1, keepdim=నిజం) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        బహిర్గతం = norm * w + b # B,T,C

        cache = (x, w, సగటు, rstd)
        తిరిగి బహిర్గతం, cache

    @staticmethod
    నిర్వచనం backward(dout, cache):
        x, w, సగటు, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - సగటు) * rstd
        # gradients for weights, bias
        db = dout.కూడిక((0, 1))
        dw = (dout * norm).కూడిక((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.సగటు(-1, keepdim=నిజం) - norm * (dnorm * norm).సగటు(-1, keepdim=నిజం)
        dx *= rstd
        తిరిగి dx, dw, db

# create a small dummy example and check w.r.t PyTorch backward
B = 2
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=నిజం)
w = torch.randn(C, requires_grad=నిజం)
b = torch.randn(C, requires_grad=నిజం)
బహిర్గతం, cache = LayerNorm.forward(x, w, b)

dout = torch.randn(B, T, C)
dx, dw, db = LayerNorm.backward(dout, cache)

# compare to PyTorch autograd
fakeloss = (బహిర్గతం * dout).కూడిక()
fakeloss.backward()
ముద్రించు("dx error:", (x.grad - dx).abs().max().అంశం())
ముద్రించు("dw error:", (w.grad - dw).abs().max().అంశం())
ముద్రించు("db error:", (b.grad - db).abs().max().అంశం())

# for reference checking in C also
x, w, సగటు, rstd = cache

నిర్వచనం లిఖ(tensor, handle):
    handle.లిఖ(tensor.detach().numpy().astype("float32").tobytes())

# Write to file
తోడుగా open('ln.bin', 'wb') ఇవ file:
    లిఖ(x, file) # (B, T, C)
    లిఖ(w, file) # (C, )
    లిఖ(b, file) # (C, )
    లిఖ(బహిర్గతం, file) # (B, T, C)
    లిఖ(సగటు, file) # (B, T)
    లిఖ(rstd, file) # (B, T)
    లిఖ(dout, file) # (B, T, C)
    లిఖ(dx, file) # (B, T, C)
    లిఖ(dw, file) # (C, )
    లిఖ(db, file) # (C, )
