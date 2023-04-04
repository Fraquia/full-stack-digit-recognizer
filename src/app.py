from flask import Flask, jsonify, request
from training import MNISTModel
import io
import torchvision.transforms as transforms
from PIL import Image
import torch

app = Flask(__name__)


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

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    return outputs
    
model = MNISTModel.load_from_checkpoint("lightning_logs/version_4/checkpoints/epoch=2-step=2814.ckpt")
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        # predict with the model
        y_hat = get_prediction(image_bytes=img_bytes)
        print('output:')
        return {
            'output':str(torch.argmax(y_hat).item())
        }
        
    

@app.route('/', methods=['GET'])
def home():
    return 'Welcome to digit recognizer app!'

if __name__ == '__main__':

    app.run(debug=True)