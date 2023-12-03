import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from NeuralNetwork import NeuralNetwork


def calculate_predictions(loader, model, device):
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return all_labels, all_predictions


def compute_f1_accuracy(labels, predictions):
    true_positives = sum((l == 1 and p == 1) for l, p in zip(labels, predictions))
    false_positives = sum((l == 0 and p == 1) for l, p in zip(labels, predictions))
    false_negatives = sum((l == 1 and p == 0) for l, p in zip(labels, predictions))

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1


def general_accuracy(labels, predictions):
    correct = sum((l == p) for l, p in zip(labels, predictions))
    return correct / len(labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./dataset', train=False, transform=transform, download=True)

    # Move data loaders to the specified device
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True, num_workers=20)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=True, num_workers=20)

    criterion = nn.CrossEntropyLoss()
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    for epoch in range(100):
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
    model.eval()

    print("TRAINING SET")
    all_labels, all_predictions = calculate_predictions(train_loader, model, device)

    # Calculate general accuracy
    accuracy = general_accuracy(all_labels, all_predictions)
    print("General Accuracy: %.2f%%" % (accuracy * 100))

    # Calculate F1 score
    f1 = compute_f1_accuracy(all_labels, all_predictions)
    print("F1 Accuracy: %.2f%%" % (f1 * 100))

    print("TEST SET")
    all_labels, all_predictions = calculate_predictions(test_loader, model, device)

    # Calculate general accuracy
    accuracy = general_accuracy(all_labels, all_predictions)
    print("General Accuracy: %.2f%%" % (accuracy * 100))

    # Calculate F1 score
    f1 = compute_f1_accuracy(all_labels, all_predictions)
    print("F1 Accuracy: %.2f%%" % (f1 * 100))


if __name__ == "__main__":
    main()
