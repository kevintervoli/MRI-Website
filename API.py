import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, send_file
import tensorflow as tf
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('./VNet/model.h5')


def overlay_mask(original_image, mask):
    alpha = 0.5  # Opacity of the overlay
    overlay = np.zeros_like(original_image)
    overlay[mask == 1] = [255, 0, 0]  # Red color for the predicted region
    return cv2.addWeighted(overlay, alpha, original_image, 1 - alpha, 0)


@app.route('/MRITest', methods=['POST'])
def predict():
    # Ensure that an image file is uploaded
    if 'image' not in request.files:
        return "No image file uploaded", 400

    # Read the uploaded image file
    image_file = request.files['image']

    # Check if the file is not empty
    if image_file.filename == '':
        return "Empty file uploaded", 400

    # Read the image
    image = Image.open(image_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Predict image
    resized_image = image.resize((256, 256))  # Resize the image
    image_array = np.array(resized_image)
    image_array = image_array / 255.0
    image_array = image_array.astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    y_pred = model.predict(image_array)[0] > 0.5
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred.astype(np.int32)

    # Resize the mask back to original image dimensions
    y_pred_resized = cv2.resize(y_pred.astype(np.uint8), (image.width, image.height))

    # Overlay mask on original image
    overlayed_image = overlay_mask(np.array(image), y_pred_resized)

    # Save the result (optional)
    cv2.imwrite("../BrainTumorSegmentation/Results/result.png", overlayed_image)

    # Return the overlayed image
    return "Image processed successfully"

@app.route('/getPath', methods=['POST'])
def get_result_path():
    RESULTS_FOLDER = 'Results'
    result_path = os.path.join(RESULTS_FOLDER, 'result.png')
    if os.path.exists(result_path):
        return send_file(result_path, mimetype='image/png')
    else:
        return "Result image not found", 404

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True,host='0.0.0.0',port=5000)