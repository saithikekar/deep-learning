import math
import random
from data import load_synth
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0/(1.0 + math.exp(-z))

def forward(x, w, b, v, c):
    #hidden layer
    k = [0.,0.,0.]
    for j in range(3):
        for i in range(2):
            k[j] += w[i][j] * x[i]
        k[j] += b[j]

    #2 sigmoid activation
    h = [0.,0.,0.]
    for j in range(3):
        h[j] = sigmoid(k[j])

    #forward pass, part 2: outputs, softmax, loss
    o = [0.,0.]
    for i in range(2):
        for j in range(3):
            o[i] += h[j] * v[j][i]
        o[i] += c[i]

    #softmax
    e = [math.exp(o[0]), math.exp(o[1])]
    total = e[0] + e[1]
    y = [e[0] / total, e[1] / total]

    return h, y

def backward(x, target, h, y, v):
    #backward pass, part 1
    do = [0., 0.]
    for i in range(2):
        if i == target:
            do[i] = y[i] - 1
        else:
            do[i] = y[i]

    #backward pass, part 2: second layer weights and biases
    dv = [[0., 0.], [0., 0.], [0., 0.]]
    for j in range(3):
        for i in range(2):
            dv[j][i] = h[j] * do[i]
    dc = [0., 0.]
    for i in range(2):
        dc[i] = do[i]

    dh = [0., 0., 0.]
    for j in range(3):
        for i in range(2):
            dh[j] += v[j][i] * do[i]

    dk = [0., 0., 0.]
    for j in range(3):
        dk[j] = dh[j] * h[j] * (1 - h[j])

    dw = [[0., 0., 0.], [0., 0., 0.]]
    for i in range(2):
        for j in range(3):
            dw[i][j] = x[i] * dk[j]

    db = [0., 0., 0.]
    for j in range(3):
        db[j] = dk[j]

    return dv, dc, dw, db

# ---------- data ----------
(xtrain, ytrain), (xval, yval), num_cls = load_synth()
# look at the data before training, to check the range of the values
print("shape:", xtrain.shape)
print("first 5 inputs:\n", xtrain[:5])
print("first 10 labels:", ytrain[:10])
print("min:", xtrain.min(), " max:", xtrain.max())
print("mean:", xtrain.mean(), " std:", xtrain.std())

input("Data printed. Press Enter to start training...")

xtrain = xtrain.tolist()      # plain Python lists for the network
ytrain = ytrain.tolist()

# ---------- starting weights: random for weights, 0 for biases ----------
random.seed(0)
w = [[random.gauss(0, 1) for j in range(3)] for i in range(2)]
b = [0., 0., 0.]
v = [[random.gauss(0, 1) for i in range(2)] for j in range(3)]
c = [0., 0.]

lr=0.01
epochs=10
losses = []
# ---------- training loop ----------
for epoch in range(epochs):
    total_loss = 0.
    for x, t in zip(xtrain, ytrain):
        h, y = forward(x, w, b, v, c)
        loss = -math.log(y[t])
        total_loss += loss
        dv, dc, dw, db = backward(x, t, h, y, v)

    # gradient descent: move each parameter a small step against its derivative
        for i in range(2):
            for j in range(3):
                w[i][j] -= lr * dw[i][j]          
        for j in range(3):
            b[j] -= lr * db[j]                 
        for j in range(3):
            for i in range(2):
                v[j][i] -= lr * dv[j][i]      
        for i in range(2):
            c[i] -= lr * dc[i]                 

    print("epoch", epoch, " average loss:", total_loss / len(xtrain))
    losses.append(total_loss / len(xtrain))


plt.plot(range(1, epochs + 1), losses, marker="o")
plt.xlabel("epoch")
plt.ylabel("average training loss")
plt.title("Training loss on the synthetic data")
plt.savefig("q4_loss.png", dpi=150, bbox_inches="tight")
plt.show()