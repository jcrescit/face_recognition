'''
① webcam으로 face detection
② 이미지 덮어씌우기
③ 얼굴 각도에 따라 덮어씌운 이미지 각도 변형하기
'''

# 라이브러리 불러오기
import cv2
import mediapipe as mp
import numpy as np

# 얼굴 찾고 표시를 해주기 위한 변수 정의
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Webcam OPEN
cap = cv2.VideoCapture(0)

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

          mp_drawing.draw_detection(image, detection)

          keypoints = detection.location_data.relative_keypoints
          right_eye = keypoints[0]
          left_eye = keypoints[1]
          nose_tip = keypoints[2]

          h, w, _ = image.shape

          # 이미지 내에서 실제 좌표(x, y) 설정
          # relative_keypoints에서 가져온 x, y좌표와 h, w을 곱하면 이미지에서 원하는 좌표를 얻을 수 있음
          right_eye = (right_eye.x * w - 100, right_eye.y * h - 150)
          left_eye = (left_eye.x * w + 20, left_eye.y * h - 150)
          nose_tip = (nose_tip.x * w, nose_tip.y * h)

          # 3개 포인트에 원으로 표시
          cv2.circle(image, right_eye, 50, (255, 0, 0), 10, cv2.LINE_AA)
          cv2.circle(image, left_eye, 50, (0, 255, 0), 10, cv2.LINE_AA)
          cv2.circle(image, nose_tip, 50, (0, 255, 255), 10, cv2.LINE_AA)
        
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('Face Detection', cv2.flip(image, 1))

      if cv2.waitKey(1) == ord('q'):
        break
      
cap.release()
cv2.destroyAllWindows()