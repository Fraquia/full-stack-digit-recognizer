from flask import Flask, jsonify, request
from model import get_prediction

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        print('output:')
        print(jsonify({'class_id': class_id, 'class_name': class_name}))
        return jsonify({'class_id': class_id, 'class_name': class_name})
    

@app.route('/', methods=['GET'])
def home():
    return 'Welcome to digit recognizer app!'

if __name__ == '__main__':
    app.run()