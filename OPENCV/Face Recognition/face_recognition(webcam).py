'''
① webcam으로 face detection
② 이미지 덮어씌우기
③ 얼굴 각도에 따라 덮어씌운 이미지 각도 변형하기
'''

# 라이브러리 불러오기
import cv2
import mediapipe as mp

# 얼굴 찾고 표시를 해주기 위한 변수 정의
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 덮어씌울 이미지 불러오기
image_right_eye = cv2.imread('D:/Computer_Vision/computer__vision/OPENCV/Face Recognition/image/panda_left_eye_2.png', cv2.IMREAD_UNCHANGED)
image_left_eye = cv2.imread('D:/Computer_Vision/computer__vision/OPENCV/Face Recognition/image/panda_right_eye_2.png', cv2.IMREAD_UNCHANGED)
image_nose = cv2.imread('D:/Computer_Vision/computer__vision/OPENCV/Face Recognition/image/fox_nose_2.png', cv2.IMREAD_UNCHANGED)

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
          right_eye = (int(right_eye.x * w), int(right_eye.y * h)) # x, y 좌표와 w, h의 곱을 정수형으로 변환 후 전체 튜플 형태로 변경
          left_eye = (int(left_eye.x * w), int(left_eye.y * h))
          nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))

          # 3개 포인트에 원으로 표시
          cv2.circle(image, right_eye, 50, (255, 0, 0), 10, cv2.LINE_AA)
          cv2.circle(image, left_eye, 50, (0, 255, 0), 10, cv2.LINE_AA)
          cv2.circle(image, nose_tip, 50, (0, 255, 255), 10, cv2.LINE_AA)

      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('Face Detection', cv2.flip(image, 1)) # cv2.flip(image, 1) : image flip

      if cv2.waitKey(1) == ord('q'):
        break
      
cap.release()
cv2.destroyAllWindows()