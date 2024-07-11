import torch
from torchvision import transforms
from PIL import Image
from model import CoinClassifier

model_path = '../models/coin_classifier.pth'
img_size = 128
num_classes = 4  # Adjust based on your dataset

# Load the model
model = CoinClassifier(num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

def predict_coin(image_path):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted.item()

# Example usage
new_image_path = './image053.jpg'
predicted_class = predict_coin(new_image_path)
print(f"The predicted class is: {predicted_class}")
