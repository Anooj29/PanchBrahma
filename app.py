import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

with open('models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.json
        onus_attr = float(data['onus-attr'])
        trans_attr = float(data['trans-attr'])
        bureau_attr = float(data['bureau-attr'])
        bureau_enq_attr = float(data['bureau-enq-attr'])

        
        feature_vector = np.array([[onus_attr, trans_attr, bureau_attr, bureau_enq_attr]])

        
        prediction = model.predict(feature_vector)[0]
        probability = model.predict_proba(feature_vector)[0][1]


        result = {
            'prediction': 'Default' if prediction == 1 else 'No Default',
            'probability': f"{probability:.2f}"
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
