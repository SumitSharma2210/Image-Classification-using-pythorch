import torch
from dataset import get_data_loaders
from model import CoinClassifier

# Paths to your dataset
test_dir = '/home/sumit/python/project2/data/test'

# Parameters
img_size = 128
batch_size = 32
num_classes = 4  # Adjust based on your dataset

# Get test data loader
_, test_loader = get_data_loaders('/home/sumit/python/project2/data/test', test_dir, img_size, batch_size)

# Load the model
model_path = '../models/coin_classifier.pth'
model = CoinClassifier(num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Evaluate the model on the test data
correct = 0
total = 0
test_loss = 0.0
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
test_loss /= len(test_loader)
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
