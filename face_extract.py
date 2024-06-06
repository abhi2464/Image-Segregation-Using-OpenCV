import cv2 as cv
import os
import numpy as np

def resize(image,scale=0.75):
    width=int(image.shape[1]*scale)
    height=int(image.shape[0]*scale)

    dimension=(width,height)

    return cv.resize(image,dimension,interpolation=cv.INTER_AREA)

base="" #Directory of the folder where all the photos are present 
faces="" #Directory of the folder where you want to save all the detected faces in the photos
photos=os.listdir(base)
haar_cascade=cv.CascadeClassifier('haar_face.xml') #For face detection

c=0
f=0
face_count=0
for i in photos:
    img=cv.imread(base+"\\"+i)
    img=resize(img)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10)

    for (x,y,w,h) in face_rect:
        crop=img[y-20:y+h+20,x-20:x+w+20]
        cv.imwrite(faces+"\\"+str(face_count)+".jpg",crop)

        rect=cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
        f+=1
        face_count+=1

    c=c+1
    print(len(face_rect))


cv.waitKey(0)