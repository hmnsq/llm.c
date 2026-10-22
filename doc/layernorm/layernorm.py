आयात torch

eps = 1e-5

श्रेणी LayerNorm:

    @staticmethod
    def forward(x, w, b):
        B, T, C = x.आकार()
        mean = x.योग_2(-1, keepdim=True) / C # B,T,1
        xshift = x - mean # B,T,C
        var = (xshift**2).योग_2(-1, keepdim=True) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        बाहर = norm * w + b # B,T,C

        cache = (x, w, mean, rstd)
        वापस बाहर, cache

    @staticmethod
    def backward(dout, cache):
        x, w, mean, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - mean) * rstd
        # gradients for weights, bias
        db = dout.योग_2((0, 1))
        dw = (dout * norm).योग_2((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.mean(-1, keepdim=True) - norm * (dnorm * norm).mean(-1, keepdim=True)
        dx *= rstd
        वापस dx, dw, db

# create a small dummy example and check w.r.t PyTorch backward
B = 2
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=True)
w = torch.randn(C, requires_grad=True)
b = torch.randn(C, requires_grad=True)
बाहर, cache = LayerNorm.forward(x, w, b)

dout = torch.randn(B, T, C)
dx, dw, db = LayerNorm.backward(dout, cache)

# compare to PyTorch autograd
fakeloss = (बाहर * dout).योग_2()
fakeloss.backward()
print("dx error:", (x.grad - dx).abs().max().item())
print("dw error:", (w.grad - dw).abs().max().item())
print("db error:", (b.grad - db).abs().max().item())

# for reference checking in C also
x, w, mean, rstd = cache

def म_लिखो(tensor, handle):
    handle.म_लिखो(tensor.detach().numpy().astype("float32").tobytes())

# Write to file
with open('ln.bin', 'wb') as file:
    म_लिखो(x, file) # (B, T, C)
    म_लिखो(w, file) # (C, )
    म_लिखो(b, file) # (C, )
    म_लिखो(बाहर, file) # (B, T, C)
    म_लिखो(mean, file) # (B, T)
    म_लिखो(rstd, file) # (B, T)
    म_लिखो(dout, file) # (B, T, C)
    म_लिखो(dx, file) # (B, T, C)
    म_लिखो(dw, file) # (C, )
    म_लिखो(db, file) # (C, )
