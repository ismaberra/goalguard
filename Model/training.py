import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sttrans import ST_Trans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

LABELS = ['TR', 'TL', 'BR', 'BL']

def calculate_class_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    for k, v in distribution.items():
        print(f"{LABELS[k]}: {v}")
    return distribution

def balance_dataset(data, labels):
    data_np = data.numpy()
    labels_np = labels.numpy()
    min_class_count = min(np.bincount(labels_np))

    balanced_data = []
    balanced_labels = []

    for label in np.unique(labels_np):
        data_class = data_np[labels_np == label]
        labels_class = labels_np[labels_np == label]
        data_resampled, labels_resampled = resample(data_class, labels_class, n_samples=min_class_count)
        balanced_data.append(data_resampled)
        balanced_labels.append(labels_resampled)

    balanced_data = np.concatenate(balanced_data)
    balanced_labels = np.concatenate(balanced_labels)

    return torch.tensor(balanced_data), torch.tensor(balanced_labels)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, patience=4):
    best_loss = float('inf')
    patience_counter = 0
    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_predicted_labels = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_predicted_labels.extend(outputs.argmax(dim=1).cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        training_losses.append(epoch_loss)

        val_loss, val_predicted_labels = validate_model(model, val_loader, criterion)
        validation_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')
        print(f'Training Predicted labels: {train_predicted_labels}')
        print(f'Validation Predicted labels: {val_predicted_labels}')

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        print_memory_usage()

    plt.figure()
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.legend()
    plt.savefig('Training_Validation_Loss.png')

def validate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    predicted_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            predicted_labels.extend(outputs.argmax(dim=1).cpu().numpy())
    return running_loss / len(dataloader.dataset), predicted_labels

def evaluate_model(model, dataloader):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
    return true_labels, pred_labels

def print_memory_usage():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    memory_in_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_in_mb:.2f} MB")

def main():
    train_data_file = 'train_data.pt'
    val_data_file = 'val_data.pt'
    test_data_file = 'test_data.pt'

    train_data_tuple, train_folders = torch.load(train_data_file)
    val_data_tuple, val_folders = torch.load(val_data_file)
    test_data_tuple, test_folders = torch.load(test_data_file)

    train_data, train_labels = train_data_tuple
    val_data, val_labels = val_data_tuple
    test_data, test_labels = test_data_tuple

    print("Class distribution in training data:")
    calculate_class_distribution(train_labels.numpy())
    print("Class distribution in validation data:")
    calculate_class_distribution(val_labels.numpy())
    print("Class distribution in test data:")
    calculate_class_distribution(test_labels.numpy())

    train_data, train_labels = balance_dataset(train_data, train_labels)
    val_data, val_labels = balance_dataset(val_data, val_labels)
    test_data, test_labels = balance_dataset(test_data, test_labels)

    print("Class distribution in balanced training data:")
    calculate_class_distribution(train_labels.numpy())
    print("Class distribution in balanced validation data:")
    calculate_class_distribution(val_labels.numpy())
    print("Class distribution in balanced test data:")
    calculate_class_distribution(test_labels.numpy())

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ST_Trans(input_dim=36, num_classes=4).to(device)
    model.device = device

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels.numpy()), y=train_labels.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=35, patience=5)

    model.load_state_dict(torch.load('best_model.pth'))
    true_labels, pred_labels = evaluate_model(model, test_loader)

    accuracy = sum(np.array(true_labels) == np.array(pred_labels)) / len(true_labels)
    print(f'Accuracy: {accuracy * 100:.2f}% ({sum(np.array(true_labels) == np.array(pred_labels))}/{len(true_labels)})')

    for i in range(len(true_labels)):
        print(f"True label: {LABELS[true_labels[i]]}, Predicted label: {LABELS[pred_labels[i]]}")

if __name__ == "__main__":
    main()



