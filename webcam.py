'''
① webcam으로 face detection
② 이미지 덮어씌우기
③ 얼굴 각도에 따라 덮어씌운 이미지 각도 변형하기
'''

# 라이브러리 불러오기
import cv2
import mediapipe as mp
import numpy as np
import math

# 얼굴 찾고 표시를 해주기 위한 변수 정의
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# rotate function 설정
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotate_math = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rotate_math, image.shape[1::-1], flags = cv2.INTER_LINEAR, borderValue = (255, 255, 255))
    return result

# Webcam OPEN
cap = cv2.VideoCapture(0)

# 덮어씌울 이미지 불러오기
image_right_eye = cv2.imread('D:/Computer_Vision/computer__vision/OPENCV/Face Recognition/image/panda_left_eye.png', cv2.IMREAD_UNCHANGED)
image_left_eye = cv2.imread('D:/Computer_Vision/computer__vision/OPENCV/Face Recognition/image/panda_right_eye.png', cv2.IMREAD_UNCHANGED)
image_nose = cv2.imread('D:/Computer_Vision/computer__vision/OPENCV/Face Recognition/image/fox_nose.png', cv2.IMREAD_UNCHANGED)

def overlay(image, x, y, w, h, overlay_image):
    alpha = overlay_image[:, :, 4]
    mask_image = alpha / 255

    for c in range(0, 3): # channel BGR
        image[y - h : y + h, x - w : x + w, c] = (overlay_image[:, :, c] * mask_image) + (image[y - h : y + h, x - w : x + w, c] * (1 - mask_image))

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_detection.process(image)

      # Draw the face detection annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      if results.detections:
        for detection in results.detections:
          keypoints = detection.location_data.relative_keypoints
          right_eye = keypoints[0]
          left_eye = keypoints[1]
          nose_tip = keypoints[2]

          h, w, _ = image.shape # height, width, channel : 이미지로부터 height, width 크기에 대한 정보 가져옴 (여기서 channel은 필요없으므로 _처리)

          # 이미지 내에서 실제 좌표(x, y) 설정
          # relative_keypoints에서 가져온 x, y좌표와 h, w을 곱하면 이미지에서 원하는 좌표를 얻을 수 있음
          right_eye = (int(right_eye.x * w) - 100, int(right_eye.y * h) - 150)
          left_eye = (int(left_eye.x * w) + 20, int(left_eye.y * h) - 150)
          nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))

          # 3개 포인트에 이미지 덮어씌우기
          image[right_eye[1] - 50 : right_eye[1] + 50, right_eye[0] - 50 : right_eye[0] + 50] = int(image_right_eye)
          image[left_eye[1] - 50 : left_eye[1] + 50, left_eye[0] - 50 : left_eye[0] + 50] = int(image_left_eye)
          image[nose_tip[1] - 50 : nose_tip[1] + 50, nose_tip[0] - 150 : nose_tip[0] + 150] = int(image_nose)

          # image rotate
          tan_theta = (left_eye[1] - right_eye[1]) / (right_eye[0] - left_eye[0])
          theta = np.arctan(tan_theta)
          rotate_angle = theta * 180 / math.pi
          rotate_image_right_eye = rotate_image(image_right_eye, rotate_angle)
          rotate_image_left_eye = rotate_image(image_left_eye, rotate_angle)
          rotate_image_nose = rotate_image(image_nose, rotate_angle)

          # overlay 함수 호출해서 이미지 적용
          overlay(image, *right_eye, 50, 50, rotate_image_right_eye)
          overlay(image, *left_eye, 50, 50, rotate_image_left_eye)
          overlay(image, *nose_tip, 150, 50, rotate_image_nose)

      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('Face Detection', cv2.flip(image, 1))

      if cv2.waitKey(1) == ord('q'):
        break
      
cap.release()
cv2.destroyAllWindows()