import cv2 as cv
import numpy as np
import os

base_path="" #Directory of the folder of extracted faces 
family=[]

features=[]
labels=[]

haar_cascade=cv.CascadeClassifier('haar_face.xml')

for i in os.listdir(base_path):
    family.append(i)

def train():
    for memeber in family:
        folder_path=base_path+"\\"+memeber
        label=family.index(memeber)

        for img in os.listdir(folder_path):
            image_path=folder_path+"\\"+img
            
            image_mat=cv.imread(image_path)
            image_mat=cv.resize(image_mat,(200,200),interpolation=cv.INTER_CUBIC)
            gray=cv.cvtColor(image_mat,cv.COLOR_BGR2GRAY)

            face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10)

            for (x,y,w,h) in face_rect:
                crop=gray[y:y+h,x:x+w]

                features.append(crop)
                labels.append(label)

train()

features=np.array(features,dtype="object")
labels=np.array(labels)

face_recognizer=cv.face.LBPHFaceRecognizer_create() # OpenCV's in-built face recognizer
face_recognizer.train(features,labels)

face_recognizer.save('trained_faces.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)

print("Training done....")