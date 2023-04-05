import torchvision.transforms as transforms
import torch
import io
from PIL import Image

def transform_image(image_bytes):
    img_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize(
                    [0.485],
                    [0.229]),
        transforms.Lambda(lambda x: torch.flatten(x)),
        transforms.Lambda(lambda x: x.view(1,-1))
        ])
    image = Image.open(io.BytesIO(image_bytes))
    return img_transform(image)

def get_prediction(model, image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    return outputs