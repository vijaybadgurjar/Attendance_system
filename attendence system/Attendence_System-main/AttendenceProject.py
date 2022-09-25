import cv2                       #package for face detection and image processing
import numpy as np               #used for working with arrays
import face_recognition          #package used to identify th detected face
import os                        #package for creating and removing directory
from datetime import datetime    #supplies classes for manipulating dates


#Storing names of all enteries from known ImagesAttendence folder in an array
path = 'ImagesAttendence'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#function to find the encodings of all images in the passed array
#encoding refers to taking different measurements of face.It consist of 128 measurements
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


#function to mark attendence in the .csv file by adding name and time of arrival.
#If name already present than it skips and if not then add it to the file
def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        print(myDataList)
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


#Finding the encodings of the known images from ImagesAttendence folder
encodeListKnown = findEncodings(images)
print('Encoding Complete')


#Createing a video capture object using webcam which will capture images of people in camera.
cap = cv2.VideoCapture(0)


#main working algo to detect face and recogonise it
while True:
    success, img = cap.read()                                #reading the image captured by webcam
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)              #resizing the img
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)              #converting it from BGR to RGB for face detection

    facesCurFrame = face_recognition.face_locations(imgS)    #finding coordinates location of faces in webcam current frame
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)  #checking whether the encodings match or not

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) #finding facial distance between known enteries and captured images

        matchIndex = np.argmin(faceDis)                         #stores minnimum face distance

        if matches[matchIndex]:                                 #if this matchindex is present in matches
            name = classNames[matchIndex].upper()

            #assigning coordinates of faces to variables and draw rectangle around face with name using putText on image
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)            #Calling markAttendence function on each name found


    cv2.imshow('webcam',img)           #To display the webcame image
    cv2.waitKey(1)                     #Time for which webcam image will be displayed


