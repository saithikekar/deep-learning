# Deep Learning - Assignment 2

This README contains the commands for the experiments in Questions 5 to 12. I used Windows PowerShell and Python. Questions 1 to 4 are mathematical derivations, so there are no commands for them.

## 1. Setup

Run the commands from the main `assignment2` folder. The `vugrad` folder contains the library from the assignment and the `code` folder contains the PyTorch scripts.

```powershell
cd vugrad
py -m pip install -e .
```

The `vugrad` library is from https://github.com/dlvu/vugrad. If the library is not already included, clone it into the assignment folder first using `git clone https://github.com/dlvu/vugrad` and place the modified experiment scripts in its `experiments` folder.

Install PyTorch and torchvision if they are not already installed. I used the CPU version of PyTorch (2.15.0+cpu).

```powershell
py -m pip install torch torchvision matplotlib
py -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__)"
```

## 2. Questions 5 to 8 - vugrad

From the `vugrad` folder, start Python:

```powershell
py
```

For Question 5, I used the following commands in Python:

```python
import vugrad as vg
import numpy as np

a = vg.TensorNode(np.random.randn(2, 2))
b = vg.TensorNode(np.random.randn(2, 2))
c = a + b

c.value
a.value + b.value
type(c.source)
c.source.op
c.source.inputs
c.source.inputs[0] is a
c.source.inputs[0].value
a.grad
```

The values of `a` and `b` are random. Questions 6 and 7 are based on reading `vugrad/core.py`. For Question 8, I used the `Expand` operation in `vugrad/ops.py` and derived its backward by hand.

Exit Python with `exit()` before running the training scripts.

## 3. Question 9 - Sigmoid and ReLU

Run these commands from the `vugrad` folder. Both models use the synthetic dataset, a learning rate of 0.01, batch size of 128, 20 epochs and a random seed of 42. I added the ReLU operation in `train_mlp_relu.py` and kept the rest of the network the same.

```powershell
py -u experiments/train_mlp.py -D synth -l 0.01 -b 128 -e 20 | Tee-Object -FilePath results_sigmoid.txt
py -u experiments/train_mlp_relu.py -D synth -l 0.01 -b 128 -e 20 | Tee-Object -FilePath results_relu.txt
```

| Activation | Last reported validation accuracy |
| --- | ---: |
| Sigmoid | 99.40% |
| ReLU | 99.03% |

## 4. Question 10 - Initialization and network depth

Stay in the `vugrad` folder. I used MNIST with a learning rate of 0.0001, batch size of 128, 5 epochs and seed 42. All runs use sigmoid. The additional hidden layer has 128 neurons.

```powershell
# Original MLP with Glorot initialization
py -u experiments/train_mlp.py -D mnist -l 0.0001 -b 128 -e 5 | Tee-Object -FilePath results_q10_baseline.txt

# Zero initialization
py -u experiments/train_mlp_q10.py -D mnist -l 0.0001 -b 128 -e 5 --init zeros | Tee-Object -FilePath results_q10_zeros.txt

# Standard normal initialization
py -u experiments/train_mlp_q10.py -D mnist -l 0.0001 -b 128 -e 5 --init normal | Tee-Object -FilePath results_q10_normal.txt

# One additional hidden layer
py -u experiments/train_mlp_q10.py -D mnist -l 0.0001 -b 128 -e 5 --init glorot --hidden-layers 2 --extra-width 128 | Tee-Object -FilePath results_q10_deep.txt
```

| Experiment | Last reported validation accuracy |
| --- | ---: |
| Glorot | 95.06% |
| Zeros | 47.66% |
| Standard normal | 78.58% |
| Extra hidden layer | 92.26% |

For Questions 9 and 10, the script measures validation accuracy at the beginning of each epoch. The reported values are from epoch 19 and epoch 4, respectively.

## 5. Question 11 - PyTorch classifier and hyperparameters

Go back to the main assignment folder before running the PyTorch scripts:

```powershell
cd ..
```

I used the CIFAR-10 classifier from the PyTorch 60-minute blitz tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

The classifier uses cross-entropy loss, SGD with momentum 0.9 and seed 42. The first run downloads CIFAR-10 into the `data` folder. Run all the commands from the same directory to reuse the dataset.

```powershell
# Baseline
py -u code/cifar_baseline.py -l 0.001 -b 4 -e 2 | Tee-Object -FilePath results_q11_baseline.txt

# Higher learning rate
py -u code/cifar_baseline.py -l 0.01 -b 4 -e 2 | Tee-Object -FilePath results_q11_high_lr.txt

# Lower learning rate
py -u code/cifar_baseline.py -l 0.0001 -b 4 -e 2 | Tee-Object -FilePath results_q11_low_lr.txt

# Larger batch size
py -u code/cifar_baseline.py -l 0.001 -b 32 -e 2 | Tee-Object -FilePath results_q11_batch32.txt

# More epochs
py -u code/cifar_baseline.py -l 0.001 -b 4 -e 5 | Tee-Object -FilePath results_q11_epochs5.txt
```

| Experiment | Learning rate | Batch size | Epochs | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.001 | 4 | 2 | 54.75% |
| Higher learning rate | 0.01 | 4 | 2 | 23.48% |
| Lower learning rate | 0.0001 | 4 | 2 | 38.57% |
| Larger batch size | 0.001 | 32 | 2 | 42.23% |
| More epochs | 0.001 | 4 | 5 | 61.70% |

## 6. Question 12 - Other changes to the classifier

Run these commands from the main `assignment2` folder. I used the five-epoch run from Question 11 as the baseline. All runs use a learning rate of 0.001, batch size of 4, 5 epochs and seed 42. SGD uses momentum 0.9.

```powershell
# Adam optimizer
py -u code/cifar_q12.py -l 0.001 -b 4 -e 5 --optimizer adam | Tee-Object -FilePath results_q12_adam.txt

# Batch normalization after both convolutional layers
py -u code/cifar_q12.py -l 0.001 -b 4 -e 5 --batchnorm | Tee-Object -FilePath results_q12_batchnorm.txt

# Dropout after both hidden fully connected layers
py -u code/cifar_q12.py -l 0.001 -b 4 -e 5 --dropout 0.3 | Tee-Object -FilePath results_q12_dropout.txt

# All three changes together
py -u code/cifar_q12.py -l 0.001 -b 4 -e 5 --optimizer adam --batchnorm --dropout 0.3 | Tee-Object -FilePath results_q12_combined.txt
```

| Experiment | Optimizer | Batch norm | Dropout | Final average training loss | Test accuracy |
| --- | --- | --- | ---: | ---: | ---: |
| Baseline (Q11) | SGD | No | 0.0 | 1.0147 | 61.70% |
| Adam | Adam | No | 0.0 | 1.1077 | 57.61% |
| Batch norm | SGD | Yes | 0.0 | 1.0377 | 63.82% |
| Dropout | SGD | No | 0.3 | 1.2137 | 58.76% |
| Combined | Adam | Yes | 0.3 | 1.3636 | 57.92% |

For Questions 11 and 12, accuracy is measured on the CIFAR-10 test set after training, as in the tutorial.

## 7. Notes

- The commands use `Tee-Object` to show the output in the terminal and save it in a text file. Running the same command again overwrites its result file.
- For Question 9, the reported Sigmoid result is from the run with seed 42 (99.40%), not the earlier unseeded run (99.55%).
- Each setting was run once. Results may vary with different seeds or environments.
- The full explanations and discussion of the results are in the assignment report.
