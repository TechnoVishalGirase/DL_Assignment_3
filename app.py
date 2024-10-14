from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Path to your CNN model
model_path = 'D:/Assignment_3/model/cnn_model.h5'

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the model with error handling
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
    model.summary()  # Print model summary to verify the architecture
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

# Mapping of labels
label_mapping = {
    0: 'drink',
    1: 'food',
    2: 'inside',
    3: 'menu',
    4: 'outside'
}

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction."""
    # Check if the image is part of the request
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    
    image_file = request.files['image']

    # Check if a file was selected
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the image temporarily
        filepath = os.path.join('D:/Assignment_3/uploads', image_file.filename)
        image_file.save(filepath)

        # Load and preprocess the image
        img = load_img(filepath, target_size=(224, 224))  # Ensure the target size matches your model input
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = model.predict(img_array)
        label_index = np.argmax(prediction, axis=1)[0]  # Get the predicted label index

        # Map the numeric label to the corresponding string label
        predicted_label = label_mapping.get(label_index, "Unknown")  # Default to "Unknown" if not found

        print("Predicted label index:", label_index)
        print("Predicted label:", predicted_label)

        # Cleanup: Remove the uploaded file
        os.remove(filepath)

        return jsonify({'prediction': predicted_label})  # Return the string label

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    os.makedirs('D:/Assignment_3/uploads', exist_ok=True)
    app.run(host='127.0.0.1', port=5006, debug=True)