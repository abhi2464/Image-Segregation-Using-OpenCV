import cv2 as cv
import numpy as np
import os

check=0
family=[] #List of all family member's name
features=np.load("features.npy",allow_pickle=True)
labels=np.load("labels.npy")
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("trained_faces.yml")

base_path="" #Directory of the folder having photos you want to segregate 

haar_cascade=cv.CascadeClassifier('haar_face.xml')


print("Enter the person name whose image you want to segregate from all the photos")
person=str(input())

for j in os.listdir(base_path):
    img=cv.imread(base_path+"\\"+j)
    copy_img=img
    # img=cv.resize(img,(int(img.shape[1]*0.60),int(img.shape[0]*0.60)))
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10)

    for (x,y,w,h) in face_rect:
        crop=gray[y:y+h,x:x+w]

        label,confidence=face_recognizer.predict(crop)

        print(f"{family[label]} is detected in the photo with a confidence of {round(confidence,2)}%")

        cv.putText(img,str(family[label]),(x-20,y-20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),thickness=2)

        cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),thickness=2)

        if label==family.index(person) and confidence>=30.00 and confidence<=100.00:
            check=1
            cv.imwrite(""+j,copy_img) #Directory where you want to save the segregated photos

    cv.imshow("detected image"+j,img)

if check == 0:
    print("Person is not found in any of the photos")
else:
    print("Segregation done !! ")
cv.waitKey(0)