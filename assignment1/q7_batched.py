import numpy as np


def forward(x, W, b, V, c):
    k = np.dot(W, x) + b
    h = 1.0 / (1.0 + np.exp(-k))
    o = np.dot(V, h) + c
    e = np.exp(o)
    y = e / np.sum(e)
    return h, y

def backward(x, t, h, y, V):
    do = y.copy()
    do[t] -= 1.0
    dv = np.outer(do, h)
    dc = do
    dh = np.dot(V.T, do)
    dk = dh * h * (1 - h)
    dw = np.outer(dk, x)
    db = dk
    return dw, db, dv, dc

#batched versions


def forward_batch(X, W, b, V, c):
    K = np.dot(X, W.T) + b                           # BLANK 1: all hidden sums at once, shape (n, 300)
    H = 1.0 / (1.0 + np.exp(-K))
    O = np.dot(H, V.T) + c                           # BLANK 2: all outputs at once, shape (n, 10)
    E = np.exp(O - O.max(axis=1, keepdims=True))     # softmax for every row
    Y = E / E.sum(axis=1, keepdims=True)
    return H, Y

def backward_batch(X, T, H, Y, V):
    n = X.shape[0]
    dO = Y.copy()
    dO[np.arange(n), T] -= 1.0               # question 2 result, for every row
    dO /= n                                  # average the loss over the batch
    dV = np.dot(dO.T, H)                     # BLANK 3: shape (10, 300)
    dc = dO.sum(axis=0)                      # BLANK 4: shape (10,)
    dH = np.dot(dO, V)                       # BLANK 5: shape (n, 300)
    dK = dH * H * (1 - H)
    dW = np.dot(dK.T, X)                     # BLANK 6: shape (300, 784)
    db = dK.sum(axis=0)                      # BLANK 7: shape (300,)
    return dW, db, dV, dc

#check: the batched result must equal the average of the single-image results
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 8
    X = rng.random((n, 784)); T = rng.integers(0, 10, n)
    W = rng.standard_normal((300, 784)) / np.sqrt(784); b = rng.standard_normal(300) * 0.1
    V = rng.standard_normal((10, 300)) / np.sqrt(300); c = rng.standard_normal(10) * 0.1
    H, Y = forward_batch(X, W, b, V, c)
    dW, db, dV, dc = backward_batch(X, T, H, Y, V)
    sW = np.zeros_like(W); sb = np.zeros_like(b); sV = np.zeros_like(V); sc = np.zeros_like(c)
    worst_y = 0.0
    for i in range(n):
        h, y = forward(X[i], W, b, V, c)
        worst_y = max(worst_y, np.abs(y - Y[i]).max())
        gW, gb, gV, gc = backward(X[i], T[i], h, y, V)
        sW += gW / n; sb += gb / n; sV += gV / n; sc += gc / n
    print("largest difference in outputs:  ", worst_y)
    print("largest difference in gradients:", max(np.abs(dW - sW).max(), np.abs(db - sb).max(), np.abs(dV - sV).max(), np.abs(dc - sc).max()))
    print("shapes:", H.shape, Y.shape, dW.shape, db.shape, dV.shape, dc.shape)
