import face_recognition
import cv2
import os

# Load known face encodings
known_face_encodings = []
known_face_names = []

path = 'known_faces'
for file_name in os.listdir(path):
    image_path = os.path.join(path, file_name)
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) > 0:
        known_face_encodings.append(encodings[0])
        name = os.path.splitext(file_name)[0]
        known_face_names.append(name)
    else:
        print(f"[WARNING] No face found in '{file_name}'. Skipping this image.")

if not os.path.exists(path):
    os.makedirs(path)
    print(f"'{path}' folder created. Add some images and restart the program.")
    exit()
for file_name in os.listdir(path):
    image_path = os.path.join(path, file_name)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    
    known_face_encodings.append(encoding)
    name = os.path.splitext(file_name)[0]
    known_face_names.append(name)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect and encode faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Scale up coordinates
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
