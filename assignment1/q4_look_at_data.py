from data import load_synth

(xtrain, ytrain), (xval, yval), num_cls = load_synth()

print("shape:", xtrain.shape)
print("first 5 inputs:\n", xtrain[:5])
print("first 10 labels:", ytrain[:10])
print("min:", xtrain.min(), " max:", xtrain.max())
print("mean:", xtrain.mean(), " std:", xtrain.std())