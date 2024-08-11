import sys
import io
import tkinter as tk
from tkinter import filedialog, Label, Button
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# Redirect stdout to support UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r", encoding="utf-8") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Load the face detection model
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if facec.empty():
    print("Error loading Haar Cascade.")

# Load the facial expression model
model = FacialExpressionModel("emotion_model.json", "emotion_model.weights.h5")

# List of emotions
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def Detect(file_path):
    try:
        image = cv2.imread(file_path)
        if image is None:
            label1.configure(foreground="#011638", text="Error loading image. Check file path.")
            return

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            label1.configure(foreground="#011638", text="No faces detected.")
            return

        for (x, y, w, h) in faces:
            try:
                fc = gray_image[y:y + h, x:x + w]
                roi = cv2.resize(fc, (48, 48))
                roi = roi.astype('float32') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)

                # Debugging output: shape and raw model output
                print(f"ROI shape: {roi.shape}")
                pred = model.predict(roi)
                print(f"Model Output: {pred}")

                # Determine the emotion
                emotion_index = np.argmax(pred)
                emotion = EMOTIONS_LIST[emotion_index]

                print(f"Predicted Emotion: {emotion}")
                label1.configure(foreground="#011638", text=emotion)
            except Exception as e:
                print(f"Error processing face at x={x}, y={y}, w={w}, h={h}. Exception: {str(e)}")
                label1.configure(foreground="#011638", text="Error during detection. See console.")
    except Exception as e:
        print(f"General detection error: {str(e)}")
        label1.configure(foreground="#011638", text="General error during detection. See console.")

def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail((top.winfo_width() // 3, top.winfo_height() // 3), Image.Resampling.LANCZOS)
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(f"Error during image upload: {str(e)}")
        label1.configure(foreground="#011638", text="Error uploading image.")

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')

heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.mainloop()
