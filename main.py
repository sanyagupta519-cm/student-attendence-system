import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import date

path = 'ImagesAttendence'
images = []
classNames = []
attendeList=[]
myList = os.listdir(path)
today = date.today()
print(myList)

for img in myList:
    curImg = cv2.imread(f'{path}/{img}')
    images.append(curImg)
    classNames.append(os.path.splitext(img)[0].capitalize())

print(classNames)
# print(images)


def findencodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode_img = face_recognition.face_encodings(img)[0]
        encodeList.append(encode_img)
    return encodeList



def markAttendence(name):
    with open('attendence.csv', 'r+') as f:
        nameList=f.readlines()
        # attendeList=[]
        for line in nameList:
            entry = line.split(',')
            attendeList.append(entry[0])
        if name not in attendeList:
            now = datetime.now()
            dstring = now.strftime('%H:%M:%S')
            # Month abbreviation, day and year
            thisDay = today.strftime("%b-%d-%Y")
            f.writelines(f'\n{name}, {dstring}, {thisDay}')
    print(attendeList)


encodedList = findencodings(images)

print("Encoding Completed")

cam=cv2.VideoCapture(0)
while True:
    success, img = cam.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodedList, encodeFace)
        faceDis = face_recognition.face_distance(encodedList, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        for i in attendeList:
            if name==i:
                cv2.putText(img, "Attendence Already Marked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (11, 64, 245), 2)



        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1),(x2, y2), (255, 116, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (255, 116, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (11, 64, 245), 2)
            markAttendence(name)

        else:
            print("Doesn't match")
            cv2.putText(img, "Doesn't match",(x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (11, 64, 245), 2)



    cv2.imshow('Webcam', img)
    cv2.waitKey(1)


