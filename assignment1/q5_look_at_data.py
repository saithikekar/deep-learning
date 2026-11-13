from data import load_mnist

(xtrain, ytrain), (xval, yval), num_cls = load_mnist()

print("train shape:", xtrain.shape, " validation shape:", xval.shape)
print("classes:", num_cls)
print("first 10 labels:", ytrain[:10])
print("min:", xtrain.min(), " max:", xtrain.max())
print("mean:", xtrain.mean(), " std:", xtrain.std())