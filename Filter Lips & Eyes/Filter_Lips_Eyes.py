import numpy as np
import cv2
from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel


lip=[52 , 55, 56, 53, 59,58,61,68,67,71,63,64,65]
eye1=[89,90,87,91,93,96,94,95]
eye2=[39,42,40,41,35,36,33,37]

def zoom(face,landmark,points):
    for  i in points:
        landmark.append(pred[i])
    landmark=np.array(landmark,dtype=int)
    x,y,w,h=cv2.boundingRect(landmark)
    mask=np.zeros(fruit.shape,dtype=np.uint8)
    cv2.drawContours(mask, [landmark], -1, (255,255,255),-1)
    mask=mask//255
    result=face * mask
    result_big = cv2.resize(result,(0,0),fx = 1.7 , fy = 1.7)

fd = UltraLightFaceDetecion("weights\RFB-320.tflite", conf_threshold=0.88)
fa = CoordinateAlignmentModel("weights\coor_2d106.tflite")

face = cv2.imread("input/person.jpg")
fruit = cv2.imread("input/my_apple.jpg")
fruit=cv2.resize(fruit,[223,226])
boxes, scores = fd.inference(face)

for pred in fa.get_landmarks(face, boxes):
    lips_landmarks = []
    for  i in [52 , 55, 56, 53, 59,58,61,68,67,71,63,64,65]:
        lips_landmarks.append(pred[i])

    eye1_landmarks = []
    for  i in [89,90,87,91,93,96,94,95]:
        eye1_landmarks.append(pred[i])

    eye2_landmarks = []
    for  i in [39,42,40,41,35,36,33,37]:
        eye2_landmarks.append(pred[i])

    lips_landmarks=np.array(lips_landmarks,dtype=int)
    x_l,y_l,w_l,h_l=cv2.boundingRect(lips_landmarks)

    eye1_landmarks=np.array(eye1_landmarks,dtype=int)
    x_e1,y_e1,w_e1,h_e1=cv2.boundingRect(eye1_landmarks)

    eye2_landmarks=np.array(eye2_landmarks,dtype=int)
    x_e2,y_e2,w_e2,h_e2=cv2.boundingRect(eye2_landmarks)

mask=np.zeros(face.shape,dtype=np.uint8)
cv2.drawContours(mask, [lips_landmarks], -1, (255,255,255),-1)
cv2.drawContours(mask, [eye1_landmarks], -1, (255,255,255),-1)
cv2.drawContours(mask, [eye2_landmarks], -1, (255,255,255),-1)
mask=mask//255
result=face * mask
result_lip=result[y_l:y_l+h_l,x_l:x_l+w_l]
result_eye1=result[y_e1:y_e1+h_e1,x_e1:x_e1+w_e1]
result_eye2=result[y_e2:y_e2+h_e2,x_e2:x_e2+w_e2]

result_lip=cv2.resize(result_lip, [90, 80])
result_eye1=cv2.resize(result_eye1,[60, 55])
result_eye2=cv2.resize(result_eye2,[60, 55])

mask=np.zeros([fruit.shape[0],fruit.shape[1],3])
mask[70:125, 50:110  ]=result_eye2
mask[70:125, 130:190]=result_eye1
mask[135:215, 80:170]=result_lip

x, y, w, h = [0, 0, fruit.shape[0], fruit.shape[1]]
for i in range(w):
    for j in range(h):
        if mask[i][j][0] == 0 and mask[i][j][1] == 0 and mask[i][j][2] == 0:
            mask[i][j] = fruit[i,j]
fruit[ x:x+w,y:y+h] = mask

cv2.imshow("result", fruit )
cv2.imwrite("output/Fruit_Result.jpg",fruit)
cv2.waitKey()
