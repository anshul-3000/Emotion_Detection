# Emotion Sense: Real-Time Emotion Detection Using Deep Learning". 


## To use our Emotion Sense--

Step-1 Just clone or download our repository and unzip it. 

Step-2 Download tensorflow and cv2 in your system using pip command. 

Step-3 Run the ipynb file to create json and weight file. 

Step-4 Then run the gui.py file ,then an interface will appear. 
                     or
            
Step-5 Make a folder named templates and move index.html into that folder then run app.py for a better interface.

Step-6 Upload any image for emotion detection.

## Overview
This project involves creating an emotion detection application using a pre-trained deep learning model. The system analyzes facial expressions to predict emotions. It includes a user-friendly graphical interface for uploading images and receiving emotion predictions.

## Project Details
Dataset: Uses pre-trained models to recognize emotions based on facial expressions.
Tools & Libraries: Python, TensorFlow/Keras, OpenCV, PIL, Tkinter
Objective: To develop an application that detects and classifies emotions from facial expressions in images.

## Features
Face Detection: Uses OpenCV to detect faces in images.
Emotion Classification: Utilizes a pre-trained deep learning model to classify emotions.
User Interface: Built with Tkinter to allow users to upload images and display emotion predictions.

## Installation
Navigate to the project directory:
cd emotion-detection

Install the required dependencies:
pip install -r requirements.txt

## Usage
Place your pre-trained model files (e.g., JSON and weights files) in the project directory.
Run the application:
python app.py

Use the GUI to upload an image. The application will detect faces, classify emotions, and display the result.

## Project Structure
app.py: Main script for the Tkinter GUI and emotion detection logic.
model.py: Contains the code to load and use the pre-trained deep learning model.
requirements.txt: List of dependencies for the project.
static/: Directory for static files (e.g., CSS, JavaScript).
templates/: Directory for HTML templates if needed.

## Results
Emotion Detection: Displays detected emotions based on facial expressions in uploaded images.
Model Accuracy: Performance metrics for emotion classification based on pre-trained model accuracy.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue if you have suggestions or find bugs.
