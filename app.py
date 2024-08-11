import os
import numpy as np
import cv2
from flask import Flask, request, render_template
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Load the face detection model
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if facec.empty():
    print("Error loading Haar Cascade.")

# Load the facial expression model
def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r", encoding="utf-8") as file:
        loaded_model_json = file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = FacialExpressionModel("emotion_model.json", "emotion_model.weights.h5")

# List of emotions
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        try:
            # Ensure the static directory exists
            static_folder = 'static'
            if not os.path.exists(static_folder):
                os.makedirs(static_folder)

            # Save the file
            file_path = os.path.join(static_folder, 'uploaded_image.jpg')
            file.save(file_path)

            # Process the image
            image = cv2.imread(file_path)
            if image is None:
                return "Error loading image. Check file path.", 400

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = facec.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                return "No faces detected.", 400

            for (x, y, w, h) in faces:
                fc = gray_image[y:y + h, x:x + w]
                roi = cv2.resize(fc, (48, 48))
                roi = roi.astype('float32') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)

                # Predict emotion
                pred = model.predict(roi)
                emotion_index = np.argmax(pred)
                emotion = EMOTIONS_LIST[emotion_index]

                return f"Detected Emotion: {emotion}"
        except Exception as e:
            return f"Error processing the image: {str(e)}", 500
    return "Error uploading file.", 400

if __name__ == "__main__":
    app.run(debug=True)
