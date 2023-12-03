import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from NeuralNetwork import NeuralNetwork


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./dataset', train=False, transform=transform, download=True)

    # Move data loaders to the specified device
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True, num_workers=20)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, pin_memory=True, num_workers=20)

    criterion = nn.CrossEntropyLoss()
    model = NeuralNetwork().to(device)  # Move the model to the specified device
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(1000):
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print("Training is done!")
    print("Testing the model...")
    total_correct = 0
    total_images = 0
    confusion_matrix = torch.zeros(10, 10)
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total_images += labels.size(0)
            total_correct += int((predicted == labels).sum())
            for i in range(len(labels)):
                confusion_matrix[labels[i], predicted[i]] += 1

    print("Accuracy: %.2f%%" % (total_correct / total_images * 100))


if __name__ == "__main__":
    main()
