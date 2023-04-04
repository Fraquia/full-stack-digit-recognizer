import os
import requests

if __name__ == '__main__':
    os.environ['NO_PROXY'] = '127.0.0.1'
    resp = requests.post("http://127.0.0.1:5000/predict",
                     files={"file": open('mnist_number.jpeg','rb')})
    
    resp_json = resp.json()
    print('Your image represents the number : ' , resp_json['output'])