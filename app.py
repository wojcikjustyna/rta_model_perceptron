from flask import Flask, request, jsonify
from sklearn.linear_model import Perceptron
import numpy as np

app = Flask(__name__)

# Initialize a Perceptron model
model = Perceptron()

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    X = np.array(data['X'])
    y = np.array(data['y'])
    model.fit(X, y)
    return "Model trained!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_new = np.array(data['X_new'])
    predictions = model.predict(X_new)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
