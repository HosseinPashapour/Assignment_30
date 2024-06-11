import cv2
import numpy as np
from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel

fd = UltraLightFaceDetecion("weights/RFB-320.tflite", conf_threshold=0.88)
fa = CoordinateAlignmentModel("weights/coor_2d106.tflite")

def recognition_of_facial_features(image, index):
    boxes, ـ = fd.inference(image)
    for pred in fa.get_landmarks(image, boxes):
        landmarks = []
        for i in index:
            landmarks.append(pred[i])
        landmarks = np.array(landmarks, dtype=int)
    return landmarks


def rotate_align_face(image, lip, eye_r, eye_l):
    left_eye_center = np.mean(eye_l, axis=0).astype(int)
    right_eye_center = np.mean(eye_r, axis=0).astype(int)

    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]

    angle = np.degrees(np.arctan2(dy, dx)) - 180
    eyes_center = (int(right_eye_center[0]), int(right_eye_center[1]))

    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)
    aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    return aligned_image

image = cv2.imread("input\mrbean.jpg")
lips_landmarks = [52, 55, 56, 53, 59, 58, 61, 68, 67, 70, 67, 71, 63, 64]
eye_right = [39, 42, 40, 41, 35, 36, 33, 37]
eye_left = [89, 90, 87, 91, 93, 96, 94, 95]

lip = recognition_of_facial_features(image=image, index=lips_landmarks)
eye_r = recognition_of_facial_features(image=image, index=eye_right)
eye_l = recognition_of_facial_features(image=image, index=eye_left)

result_image = rotate_align_face(image, lip, eye_r, eye_l)

cv2.imshow("result", result_image)
cv2.imwrite("output/MR_Bean.jpg", result_image)
cv2.waitKey(0)