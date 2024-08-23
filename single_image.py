import cv2
import numpy
from predictions import predict
import matplotlib.pyplot as plt 

image = cv2.imread('D:\\Computer vision\\Image Processing\\Projects\\Face Recognition\\test_data\\girl.png',1)


model = cv2.CascadeClassifier('D:\\Computer vision\\Image Processing\\Projects\\Face Recognition\\cascade classifiers\\haarcascade_frontalface_default.xml')


cordinates , num_of_faces = model.detectMultiScale2(image)

print(cordinates)
print('-----')

print(len(num_of_faces))
pt1 = (cordinates[0][0] , cordinates[0][1])
pt2 = (cordinates[0][2]  + cordinates[0][0] ,cordinates[0][3] + cordinates[0][1])

font = cv2.FONT_HERSHEY_COMPLEX
text = 'Number of Faces detected = '+str(len(num_of_faces))
cv2.rectangle(image,pt1,pt2,(0,0,255),2,cv2.LINE_AA)
cv2.putText(image,text,(15,25) , font,0.5,(0,0,0),2,cv2.LINE_8)
cv2.imwrite('D:\\Computer vision\\Image Processing\\Projects\\Face Recognition\\face detection\\re.png',image)
predict(image)


