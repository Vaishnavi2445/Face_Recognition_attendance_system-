import face_recognition
import cv2
import numpy as np
import pandas as pd
import csv
import os
from datetime import datetime

students = {
        "Shrashti Bhumarkar": {"roll": "001", "image": "photos/shrashti.jpg"},
        "Vaishnavi Gupta": {"roll": "002", "image": "photos/vaishnavi.jpg"},
        
}

known_face_encodings = []
known_face_names = []
student_roll_numbers = []

for name, data in students.items():
    image_path = data["image"]
    roll_number = data["roll"]
    
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    student_roll_numbers.append(roll_number)

video_capture = cv2.VideoCapture(0)

csv_file = 'attendance.csv'
if not os.path.exists(csv_file):
    data = {'Name': [], 'Roll Number': [], 'Date Time': []}
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)


df = pd.read_csv(csv_file)

while True:
    ret, frame = video_capture.read()

    face_location = face_recognition.face_locations(frame)
    face_encoding = face_recognition.face_encodings(frame, face_location)

    for face_encoding, face_location in zip(face_encoding, face_location):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "unknown"
        roll_number = " "

        if True in matches:
            matched_indices = [i for i, match in enumerate(matches) if match]
            first_match_index = matched_indices[0]
            name = known_face_names[first_match_index]
            roll_number = student_roll_numbers[first_match_index]

            if name not in df['Name'].values:
                now = datetime.now()
                date_time = now.strftime("%Y-%m-%d %H:%M:%S")
                new_row = pd.DataFrame({'Name': [name], 'Roll Number': [roll_number], 'Date Time': [date_time]})
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(csv_file, index=False)

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{name} ({roll_number})"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()