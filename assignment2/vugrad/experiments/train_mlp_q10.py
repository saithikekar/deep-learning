from _context import vugrad

import numpy as np

# for running from the command line
from argparse import ArgumentParser

import vugrad as vg
from vugrad.ops import BatchMM, Expand

# Parse command line arguments
parser = ArgumentParser()

parser.add_argument('-D', '--dataset',
                dest='data',
                help='Which dataset to use. [synth, mnist]',
                default='synth', type=str)

parser.add_argument('-b', '--batch-size',
                dest='batch_size',
                help='The batch size (how many instances to use for a single forward/backward pass).',
                default=128, type=int)

parser.add_argument('-e', '--epochs',
                dest='epochs',
                help='The number of epochs (complete passes over the complete training data).',
                default=20, type=int)

parser.add_argument('-l', '--learning-rate',
                dest='lr',
                help='The learning rate. That is, a scalar that determines the size of the steps taken by the '
                     'gradient descent algorithm. 0.1 works well for synth, 0.0001 works well for MNIST.',
                default=0.01, type=float)

# Settings for the Q10 experiments
parser.add_argument('--init', choices=['glorot', 'zeros', 'normal'],
                    default='glorot')
parser.add_argument('--hidden-layers', type=int, choices=[1, 2], default=1)
parser.add_argument('--extra-width', type=int, default=128,
                    help='Size of the second hidden layer, if used.')

args = parser.parse_args()

np.random.seed(42)  #added this line to make sure the results are reproducible. 

## Load the data
if args.data == 'synth':
    (xtrain, ytrain), (xval, yval), num_classes = vg.load_synth()
elif args.data == 'mnist':
    (xtrain, ytrain), (xval, yval), num_classes = vg.load_mnist(final=False, flatten=True)
else:
    raise Exception(f'Dataset {args.data} not recognized.')

print(f'## loaded data:')
print(f'         number of instances: {xtrain.shape[0]} in training, {xval.shape[0]} in validation')
print(f' training class distribution: {np.bincount(ytrain)}')
print(f'     val. class distribution: {np.bincount(yval)}')

num_instances, num_features = xtrain.shape

class MyLinear(vg.Module):
    """
    A linear operation. Applies a matrix transformation and a vector translation.
    """

    def __init__(self, input_size, output_size, init="glorot"):
        super().__init__()

        # Choose how to initialize the weights.
        if init == 'glorot':
            glorot_std = np.sqrt(2.0 / (input_size + output_size))
            w = np.random.randn(output_size, input_size) * glorot_std
        elif init == 'zeros':
            w = np.zeros((output_size, input_size))
        elif init == 'normal':
            w = np.random.randn(output_size, input_size)
        else:
            raise ValueError(f'Unknown initialization: {init}')

        self.w = vg.TensorNode(w)

        # weights of the bias (the translation)
        b = np.zeros((1, output_size))
        self.b = vg.TensorNode(b)
        # -- We initialize the biases to zero for simplicity. This is a common approach, but with ReLU units it's
        #    sometimes best to add a little noise to avoid dead neurons.

    def forward(self, input):

        outsize, insize = self.w.size()
        n, f = input.size()

        assert f == insize, f'Number of features in input ({f}) does not match input dimension ({insize}).'
        assert len(input.size()) == 2

        # Multiply all input vectors by the weight matrix.
        x = BatchMM.do_forward(self.w, input)

        assert x.size() == (n, outsize)

        exb = Expand.do_forward(self.b, dim=0, repeats=n)
        # -- We are broadcasting the (1, outsize) vector b over the (n, outsize) matrix x. Numpy normally does this
        #    automatically, if we just do `x + self.b`, but we wouldn't get a gradient over that operation. Expand
        #    is a minimal broadcasting op that is sufficient for our purposes.
        # -- In pytorch, full-featured broadcasting is implemented so there you would actually be able to do `x + self.b`.

        assert x.size() == exb.size()

        return x + exb

    def parameters(self):
        return [self.w, self.b]


# Create an MLP with one or two hidden layers.
class MLP(vg.Module):

    def __init__(self, input_size, output_size, hidden_mult=4,
                 init='glorot', hidden_layers=1, extra_width=128):
        super().__init__()

        hidden_size = hidden_mult * input_size
        self.layer1 = MyLinear(input_size, hidden_size, init=init)
        self.extra_layer = None

        if hidden_layers == 2:
            self.extra_layer = MyLinear(hidden_size, extra_width, init=init)
            self.layer2 = MyLinear(extra_width, output_size, init=init)
        else:
            self.layer2 = MyLinear(hidden_size, output_size, init=init)

    def forward(self, input):
        assert len(input.size()) == 2

        hidden = vg.sigmoid(self.layer1(input))

        # Use the extra hidden layer only for the depth experiment.
        if self.extra_layer is not None:
            hidden = vg.sigmoid(self.extra_layer(hidden))

        output = self.layer2(hidden)
        return vg.logsoftmax(output)

    def parameters(self):
        params = self.layer1.parameters()
        if self.extra_layer is not None:
            params += self.extra_layer.parameters()
        params += self.layer2.parameters()
        return params


## Instantiate the model
mlp = MLP(input_size=num_features, output_size=num_classes,
          init=args.init, hidden_layers=args.hidden_layers,
          extra_width=args.extra_width)

print(f' initialization: {args.init}')
print(f' hidden layers: {args.hidden_layers}')
print(f' first hidden size: {4 * num_features}')
if args.hidden_layers == 2:
    print(f' second hidden size: {args.extra_width}')
print(f' learning rate: {args.lr}, batch size: {args.batch_size}, epochs: {args.epochs}')

n, m = xtrain.shape
b = args.batch_size

print('\n## Starting training')
for epoch in range(args.epochs):

    print(f'epoch {epoch:03}')

    ## Compute validation accuracy
    o = mlp(vg.TensorNode(xval))
    oval = o.value

    predictions = np.argmax(oval, axis=1)
    num_correct = (predictions == yval).sum()
    acc = num_correct / yval.shape[0]

    o.clear() # gc the computation graph
    print(f'       accuracy: {acc:.4}')

    cl = 0.0 # running sum of the training loss

    # We loop over the data in batches of size `b`
    for fr in range(0, n, b):

        # The end index of the batch
        to = min(fr + b, n)

        # Slice out the batch and its corresponding target values
        batch, targets = xtrain[fr:to, :], ytrain[fr:to]

        # Wrap the inputs in a Node
        batch = vg.TensorNode(value=batch)

        outputs = mlp(batch)
        loss = vg.logceloss(outputs, targets)
        # -- The computation graph is now complete. It consists of the MLP, together with the computation of
        #    the scalar loss.
        # -- The variable `loss` is the TensorNode at the very top of our computation graph. This means we can call
        #    it to perform operations on the computation graph, like clearing the gradients, starting the backpropgation
        #    and clearing the graph.
        # -- Note that we set the MLP up to produce log probabilties, so we should compute the CE loss for these.

        cl += loss.value
        # -- We must be careful here to extract the _raw_ value for the running loss. What would happen if we kept
        #    a running sum using the TensorNode?

        # Start the backpropagation
        loss.backward()

        # pply gradient descent
        for parm in mlp.parameters():
            parm.value -= args.lr * parm.grad
            # -- Note that we are directly manipulating the members of the parm TensorNode. This means that for this
            #    part, we are not building up a computation graph.

        # -- In Pytorch, the gradient descent is abstracted away into an Optimizer. This allows us to build slightly more
        #    complexoptimizers than plain graident descent.

        # Finally, we need to reset the gradients to zero ...
        loss.zero_grad()
        # ... and delete the parts of the computation graph we don't need to remember.
        loss.clear()

    print(f'   running loss: {cl/n:.4}')
