from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from collections import Counter
import smtplib
import os
import pygame
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
app = Flask(__name__)

pygame.init()

# Initialize pygame mixer
pygame.mixer.init()
# Route to render the HTML form
@app.route('/')
def index():
    return render_template('upload.html')

# Route to handle form submission
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded file to a temporary location
            video_path = os.path.join(app.root_path, 'static', 'uploads', file.filename)
            file.save(video_path)

            # Now you can call your backend code with the video path
            process_video(video_path)

            return redirect(url_for('index'))

# Function to process the uploaded video

# @app.route('/anomaly', methods=['GET'])
# def anomaly_detected():
#     # Assuming the frame is saved as 'last_frame.jpg' in the static directory
#     frame_path = os.path.join(app.root_path, 'static', 'last_frame.jpg')
#     #return send_from_directory(directory=os.path.dirname(frame_path), filename=os.path.basename(frame_path))
#     most_common_prediction = most_common_prediction # You need to pass this dynamically
#     return render_template('anomaly.html', most_common_prediction=most_common_prediction)

def process_video(video_path):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=r"C:\Users\anurw\Downloads\VIDEOPROCESS\VIDEOPROCESS\model_unquant.tflite")
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Define class names
    class_names = ['Abuse', 'Arrest', 'Assault', 'Fighting', 'Robbery']

    # Function to preprocess input frames.
    def preprocess_frame(frame):
        # Preprocess the frame according to your model requirements.
        # Resize, normalize, etc.
        frame_resized = cv2.resize(frame, (224, 224))
        frame_resized = frame_resized.astype(np.float32)
        frame_resized /= 255.0
        return frame_resized

    # Function to make predictions on a frame.
    def predict_frame(frame):
        input_data = np.expand_dims(preprocess_frame(frame), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data

    # Load the video file
    video_path = video_path
    cap = cv2.VideoCapture(video_path)

    # Container for predictions
    predictions = []

    # Function to send an email
    def SendMail(most_common_prediction, frame_path):
        msg = MIMEMultipart()
        msg['Subject'] = ' 🚨 Anomaly Detected: Immediate Action Required 🚨'
        msg['From'] = 'nayanakishore28@gmail.com.cc'
        msg['To'] = 'nayanakishore28@gmail.com.cc'

        # Create the email body
       
        body = f"""
            Dear Administrator,
                    Anomaly Alert!!!!
                    Anomaly Detected: {most_common_prediction}
                Immediate action is required! Our anomaly detection system has detected a potential anomaly in progress. The safety and security of the public are at risk, and urgent intervention is necessary. Please take immediate action to address this situation and ensure the safety of everyone in the vicinity. This is a critical situation, and prompt action is essential to prevent further harm and ensure public safety.

            Sincerely,
            SafeNest Security Team
                """
        text = MIMEText(body)
        msg.attach(text)

        # Attach the image
        with open(frame_path, 'rb') as file:
            img = MIMEImage(file.read())
        msg.attach(img)

        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login("nayanakishore28@gmail.com", "mghmxpqycdeozioa")
        s.sendmail("nayanakishore28@gmail.com", "nayanadece@gmail.com", msg.as_string())
        s.quit()
        print("Mail send succesfully")
        play_beep()

    # Loop through the frames and make predictions
    while cap.isOpened():
        ret, frame = cap.read()
        # Example: Capture the last frame and save it as an image
        ret, last_frame = cap.read()
        if ret:
            print("Frame captured")
            frame_path = "last_frame.jpg"
            cv2.imwrite(frame_path, last_frame)
        # Now call SendMail with the path to the saved image
        if not ret:
            break
        
        # Preprocess the frame and make predictions
        prediction = predict_frame(frame)
        
        # Get the predicted class index
        pred_class_index = np.argmax(prediction)
        
        # Get the predicted class name
        pred_class_name = class_names[pred_class_index]
        
        # Append the prediction to the list
        predictions.append(pred_class_name)
        
        # Print the prediction
        print("Prediction:", pred_class_name)
        
        ret, last_frame = cap.read()
        # Display the frame (optional)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Find the most frequent prediction
    most_common_prediction = Counter(predictions).most_common(1)[0][0]
    print("Most frequent prediction:", most_common_prediction)

    # Call the SendMail function with the last frame as the image attachment
    SendMail(most_common_prediction, frame_path)

    # Release the video capture object and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def play_beep():
    # Load the beep sound file
    beep_sound = pygame.mixer.Sound("C:\\Users\\anurw\\Downloads\\VIDEOPROCESS\\VIDEOPROCESS\\beep.mp3")
    # Play the beep sound
    beep_sound.play()

if __name__ == '__main__':
    app.run(debug=True)