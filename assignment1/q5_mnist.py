import numpy as np
from data import load_mnist

# shapes:  x (784,)   W (300, 784)   b (300,)   V (10, 300)   c (10,)

def forward(x, W, b, V, c):
    k = np.dot(W, x) + b
    h = 1.0/(1.0 + np.exp(-k))
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

# ---------- data ----------
(xtrain, ytrain), (xval, yval), num_cls = load_mnist()
xtrain = xtrain / 255.0          # pixel values are 0 to 255, so bring them to 0 to 1
xval = xval / 255.0

# ---------- starting weights: small random values, biases 0 ----------
np.random.seed(0)
W = np.random.randn(300, 784) / np.sqrt(784)
b = np.zeros(300)
V = np.random.randn(10, 300) / np.sqrt(300)
c = np.zeros(10)

lr = 0.01
epochs = 1 

# ---------- training loop ----------
for epoch in range(epochs):
    running = 0.
    for n in range(len(xtrain)):
        x, t = xtrain[n], ytrain[n]
        h, y = forward(x, W, b, V, c)
        running += -np.log(y[t])
        dW, db, dV, dc = backward(x, t, h, y, V)
        W -= lr * dW
        b -= lr * db
        V -= lr * dV
        c -= lr * dc
        if (n + 1) % 5000 == 0:
            print("epoch", epoch, " instance", n + 1, " average loss:", running / 5000)
            running = 0.