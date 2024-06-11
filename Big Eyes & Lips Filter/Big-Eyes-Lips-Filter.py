import cv2
import numpy as np
from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel

fd = UltraLightFaceDetecion("weights/RFB-320.tflite", conf_threshold=0.88)
fa = CoordinateAlignmentModel("weights/coor_2d106.tflite")

def facial_features(landmarks, image):
    features_landmarks=[]
    for i in landmarks:
        features_landmarks.append(pred[i])
    features_landmarks=np.array(features_landmarks, dtype=int)
    x, y, w, h=cv2.boundingRect(features_landmarks)
    mask=np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [features_landmarks], -1, (255, 255, 255), -1)
    mask=mask // 255
    result=image*mask
    result_fit=result[y:y+h, x:x+w]
    doubled_feature=cv2.resize(result_fit, (w*2, h*2))

    for i in range(h*2):
        for j in range(w*2):
            if doubled_feature[i][j][0] == 0 and doubled_feature[i][j][1] == 0 and doubled_feature[i][j][2] == 0:
                doubled_feature[i][j] = image[int(y-h//2)+i, int(x-w//2)+j]
    image[int(y-h//2):int(y-h//2)+h*2, int(x-w//2):int(x-w//2)+w*2]=doubled_feature
    return image

image=cv2.imread("input\person.jpg")
boxes, scores = fd.inference(image)

for pred in fa.get_landmarks(image, boxes):
    features_landmarks=[]
lip_landmarks=[52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64]
right_eye_landmarks=[35, 36, 33, 37, 39, 42, 40, 41]
left_eye_landmarks=[89, 90, 87, 91, 93, 96, 94, 95]

image=facial_features(lip_landmarks, image)
image=facial_features(right_eye_landmarks, image)
image=facial_features(left_eye_landmarks, image)


cv2.imshow("result", image)
cv2.imwrite("Output/My_Pic.jpg", image)
cv2.waitKey()