# Adapted from the PyTorch 60 Minute Blitz, Training a Classifier:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
from torchvision.transforms import v2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import ArgumentParser


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def main():
    parser = ArgumentParser()
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-e', '--epochs', type=int, default=2)
    args = parser.parse_args()

    # Use the same seed for each run.
    torch.manual_seed(42)

    # Load and normalize the CIFAR-10 images.
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)

    print('Dataset: CIFAR-10')
    print('Training images:', len(trainset))
    print('Test images:', len(testset))
    print('Learning rate:', args.learning_rate)
    print('Batch size:', args.batch_size)
    print('Epochs:', args.epochs)
    print('Optimizer: SGD')
    print('Momentum: 0.9')
    print('Random seed: 42')

    # Train the network.
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item() * labels.size(0)
            if (i + 1) % 200 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        print(f'Epoch {epoch + 1} average loss: {epoch_loss / len(trainset):.4f}')

    print('Finished Training')

    # Calculate accuracy on the test set.
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            predicted = net(images).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test accuracy: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    main()
