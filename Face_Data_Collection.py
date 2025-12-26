import numpy as np
import cv2

cap = cv2.VideoCapture(0)
skip = 0
face_data = []
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# initialize face_section 
face_section = np.zeros((100,100,3), dtype=np.uint8)

name = input('enter your name : ')

while True:
    value, frame = cap.read() 
    
    if value == False:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Detecting the Largest Face if multiple faces exist
    faces = sorted(faces, key=lambda f: f[2] * f[3])
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 3)
        
        offset = 10

        # clamp the crop coordinates so they never go out of bounds
        y1 = max(0, y - offset)
        y2 = min(frame.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(frame.shape[1], x + w + offset)

        face_section = frame[y1:y2, x1:x2]

        # prevent resize crash if empty
        if face_section.size != 0:
            face_section = cv2.resize(face_section, (100,100))
        
            skip += 1
            if skip % 10 == 0:
                face_data.append(face_section)
                print(len(face_data))
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Face Section", face_section)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

face_data = np.asarray(face_data)
face_data = face_data.reshape(face_data.shape[0], -1)
print(face_data.shape)

np.save(name + ".npy", face_data)

cap.release()
cv2.destroyAllWindows()
