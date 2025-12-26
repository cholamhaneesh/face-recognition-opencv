import os
import cv2
import numpy as np

# KNN Algorithm

def KNN(X,Y,query,k=100):
    k = min(k, X.shape[0])
    distance=np.sqrt(np.sum((X-query)**2,axis=1))
    nearest_idx=np.argsort(distance)[:k]
    
    nearest_labels=Y[nearest_idx]
    
    values,count=np.unique(nearest_labels,return_counts=True)
    nearest_lable=values[np.argmax(count)]
    return nearest_lable
    

cap=cv2.VideoCapture(0)
face_data=[]
class_id=0
lables=[]
names={}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

#Data preperation
DATA_DIR = "."
for file in os.listdir(DATA_DIR):
    if file.endswith('.npy'):
        data = np.load(os.path.join(DATA_DIR, file))
        print("loaded",file)
        face_data.append(data)
        
        #creating lables
        lable=class_id*np.ones(data.shape[0],)
        lables.append(lable)
        print("created lable",class_id)
        names[class_id]=file[:-4]
        class_id+=1

face_dataset=np.concatenate(face_data,axis=0)
face_lables=np.concatenate(lables,axis=0)


#Prediction
while True:
    value,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if value==False:
        continue
    
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    
    for face in faces:
        x,y,w,h=face
        
        offset=10
        y1=max(0,y-offset)
        y2=min(frame.shape[0],y+h+offset)
        x1 = max(0, x - offset)
        x2 = min(frame.shape[1], x + w + offset)
        
        face_section = frame[y1:y2, x1:x2]
        face_section=cv2.resize(face_section,(100,100))
        
        output=KNN(face_dataset,face_lables,face_section.flatten())
        
        pred_name=names[int(output)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow("Faces",frame)
    #Quitting
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
        
        
        