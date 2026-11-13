import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt
from data import load_mnist


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


EPOCHS = 5
LOG_EVERY = 500                  # one point on the loss curve per 500 instances
RESULTS_FILE = "q8_results.json"


def evaluate(X, Y, W, b, V, c):
    """Average loss and accuracy of the network on a whole data set."""
    total_loss, correct = 0.0, 0
    for x, t in zip(X, Y):
        h, y = forward(x, W, b, V, c)
        total_loss += -np.log(y[t])
        if np.argmax(y) == t:
            correct += 1
    return total_loss / len(X), correct / len(X)


def train(xtr, ytr, xev, yev, lr, seed, label):
    """Train for EPOCHS epochs with SGD, one instance at a time."""
    rng = np.random.default_rng(seed)
    n_in = xtr.shape[1]
    W = rng.standard_normal((300, n_in)) / np.sqrt(n_in)
    b = np.zeros(300)
    V = rng.standard_normal((10, 300)) / np.sqrt(300)
    c = np.zeros(10)

    curve, train_loss, eval_loss, eval_acc = [], [], [], []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(xtr))      # new random order every epoch
        running, epoch_total = 0.0, 0.0
        for step, n in enumerate(order):
            x, t = xtr[n], ytr[n]
            h, y = forward(x, W, b, V, c)
            loss = -np.log(y[t])
            running += loss
            epoch_total += loss
            dW, db, dV, dc = backward(x, t, h, y, V)
            W -= lr * dW
            b -= lr * db
            V -= lr * dV
            c -= lr * dc
            if (step + 1) % LOG_EVERY == 0:
                curve.append(running / LOG_EVERY)
                running = 0.0
        ev_loss, ev_acc = evaluate(xev, yev, W, b, V, c)
        train_loss.append(epoch_total / len(xtr))
        eval_loss.append(ev_loss)
        eval_acc.append(ev_acc)
        print(f"{label}  epoch {epoch + 1}  train loss {train_loss[-1]:.4f}  "
              f"{'test' if 'final' in label else 'val'} loss {ev_loss:.4f}  accuracy {ev_acc:.4f}", flush=True)
    return {"curve": curve, "train_loss": train_loss, "eval_loss": eval_loss, "eval_acc": eval_acc}


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def run(results, key, xtr, ytr, xev, yev, lr, seed):
    """Run one training, or reuse it if it was already finished earlier."""
    if key not in results:
        results[key] = train(xtr, ytr, xev, yev, lr, seed, key)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f)
    else:
        print(key, "already done, reusing saved result")
    return results[key]


# data (validation split)
(xtrain, ytrain), (xval, yval), num_cls = load_mnist()
xtrain = xtrain / 255.0
xval = xval / 255.0
results = load_results()
epochs_axis = list(range(1, EPOCHS + 1))

# experiment 1: loss curve, and training loss against validation loss 
base = run(results, "lr0.01_seed0", xtrain, ytrain, xval, yval, 0.01, 0)

steps = [(i + 1) * LOG_EVERY for i in range(len(base["curve"]))]
plt.figure()
plt.plot(steps, base["curve"])
plt.xlabel("instances seen")
plt.ylabel(f"training loss (average over {LOG_EVERY} instances)")
plt.title("Loss curve, learning rate 0.01")
plt.savefig("q8_loss_curve.png", dpi=150, bbox_inches="tight")

plt.figure()
plt.plot(epochs_axis, base["train_loss"], marker="o", label="training loss")
plt.plot(epochs_axis, base["eval_loss"], marker="o", label="validation loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Training loss and validation loss per epoch")
plt.savefig("q8_train_vs_val.png", dpi=150, bbox_inches="tight")

# experiment 2: three random initialisations, mean and standard deviation
curves = [run(results, f"lr0.01_seed{s}", xtrain, ytrain, xval, yval, 0.01, s)["curve"] for s in (0, 1, 2)]
curves = np.array(curves)
mean, std = curves.mean(axis=0), curves.std(axis=0)
plt.figure()
plt.plot(steps, mean, label="mean of 3 runs")
plt.fill_between(steps, mean - std, mean + std, alpha=0.3, label="plus and minus one standard deviation")
plt.xlabel("instances seen")
plt.ylabel("training loss")
plt.legend()
plt.title("Three random initialisations, learning rate 0.01")
plt.savefig("q8_seeds_mean_std.png", dpi=150, bbox_inches="tight")

# experiment 3: learning rates 
plt.figure()
print("\nlearning rate   final validation loss   final validation accuracy")
for lr in (0.001, 0.003, 0.01, 0.03):
    r = run(results, f"lr{lr}_seed0", xtrain, ytrain, xval, yval, lr, 0)
    plt.plot(epochs_axis, r["eval_loss"], marker="o", label=f"lr = {lr}")
    print(f"{lr:<15} {r['eval_loss'][-1]:<23.4f} {r['eval_acc'][-1]:.4f}")
plt.xlabel("epoch")
plt.ylabel("validation loss")
plt.legend()
plt.title("Validation loss for different learning rates")
plt.savefig("q8_learning_rates.png", dpi=150, bbox_inches="tight")

# experiment 4: final run on the full data 
BEST_LR = None      

if BEST_LR is None:
    print("\nExperiments 1 to 3 are done. Look at the table and the plots,")
    print("then set BEST_LR in the file and run it again for experiment 4.")
else:
    (xfull, yfull), (xtest, ytest), _ = load_mnist(final=True)
    xfull = xfull / 255.0
    xtest = xtest / 255.0
    final = run(results, f"final_lr{BEST_LR}", xfull, yfull, xtest, ytest, BEST_LR, 0)
    print(f"\nFinal test accuracy with lr = {BEST_LR}: {final['eval_acc'][-1]:.4f}")
