import torch
from torchvision import transforms
from PIL import Image

# Load your trained model (make sure to specify the correct path to your model)
model = torch.load('path_to_your_model.pth')
model.eval()

# Define the image transformation
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_bytes)
    return my_transforms(image).unsqueeze(0)  # Add batch dimension

# Prediction Function
def predict(image_path):
    # Apply the transformations
    tensor = transform_image(image_path)
    # Perform inference
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)
        return 'Safe' if predicted.item() == 0 else 'Damaged'

# Example of usage
# prediction = predict('path_to_your_image.jpg')
# print(prediction)