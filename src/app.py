from flask import Flask, jsonify, request
from models.train_model import MNISTModel
from models.predict_model import get_prediction
import torch

app = Flask(__name__)

#define model 
model = MNISTModel.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=2-step=2814.ckpt")
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        try:
            file = request.files['file']
        except KeyError:
            return jsonify({'error': 'no text sent'})

        img_bytes = file.read()
        y_hat = get_prediction(model, image_bytes=img_bytes)
        return {
            'output':str(torch.argmax(y_hat).item())
        }
        
@app.route('/', methods=['GET'])
def home():
    return 'Welcome to digit recognizer app!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True) 