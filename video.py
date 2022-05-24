'''
① 비디오 파일 불러오기
② 이미지 덮어씌우기
③ 얼굴 각도에 따라 덮어씌운 이미지 각도 변형하기
'''

# 라이브러리 불러오기
import cv2
import mediapipe as mp
import numpy as np
import math

# 얼굴을 찾고, 찾은 얼굴에 표시를 해주기 위한 변수 정의
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# rotate function 설정
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotate_math = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rotate_math, image.shape[1::-1], flags = cv2.INTER_LINEAR, borderValue = (255, 255, 255))
    return result

# 동영상 파일 열기
cap = cv2.VideoCapture('D:/Computer_Vision/computer__vision/OPENCV/Face Recognition/video/face_video2.mp4')

# 덮어씌울 이미지 불러오기
image_right_eye = cv2.imread('D:/Computer_Vision/computer__vision/OPENCV/Face Recognition/image/panda_left_eye.png', cv2.IMREAD_UNCHANGED)
image_left_eye = cv2.imread('D:/Computer_Vision/computer__vision/OPENCV/Face Recognition/image/panda_right_eye.png', cv2.IMREAD_UNCHANGED)
image_nose = cv2.imread('D:/Computer_Vision/computer__vision/OPENCV/Face Recognition/image/fox_nose.png', cv2.IMREAD_UNCHANGED)

# 동영상(3channel)에 불러온 이미지(4channel)를 넣어주기 위한 추가연산함수
def overlay(image, x, y, w, h, overlay_image): # 대상 이미지(3channel), x, y, width, height, 덮어씌울 이미지(4channel)
    alpha = overlay_image[:, :, 4] # BGRA 중에서 A만 가져옴
    mask_image = alpha / 255
    # 0 ~ 255 -> 0 ~ 1 (1 : 불투명, 0 : 투명)
    # mask_image가 붍투명하면 overlay_image을 적용하고, 투명하면 overlay_image을 적용하지 않게됨

    for c in range(0, 4): # channel BGR
        image[y - h : y + h, x - w : x + w, c] = (overlay_image[:, :, c] * mask_image) + (image[y - h : y + h, x - w : x + w, c] * (1 - mask_image))


with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
  # model_selection = 0 : 카메라로부터 2m 이내의 근거리에 적합, = 1 : 5m 이내의 근거리에 적합
  # min_detection_confidence : 신뢰도가 특정 %이상되면, 얼굴로 인식할지 정의 (threshold개념과 유사)

    while cap.isOpened():
      success, image = cap.read() # frame을 불러와서 success되면 image에 저장
      if not success:
        break

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_detection.process(image) # image로부터 얼굴을 검출해서 results로 반환

      # Draw the face detection annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      if results.detections: # detection한 results가 있을 시 진행
        
        # 반복문 통해서 검출된 사람 얼굴만큼 draw 해줌
        # 6개 포인트(오른쪽 눈, 왼쪽 눈, 코 끝부분, 입 중심, 오른쪽 귀, 왼쪽 귀) 검출
        for detection in results.detections:
          # mp_drawing.draw_detection(image, detection) # bounding box, relative keypoints를 이미지화(이미지를 덮어씌우는 과정에서는 불필요하므로 제거)
          # print(detection) # detection 정보 확인

          # 특정 위치 가져오기(index 적용)
          keypoints = detection.location_data.relative_keypoints
          right_eye = keypoints[0]
          left_eye = keypoints[1]
          nose_tip = keypoints[2]

          h, w, _ = image.shape # height, width, channel : 이미지로부터 height, width 크기에 대한 정보 가져옴 (여기서 channel은 필요없으므로 _처리)

          # 이미지 내에서 실제 좌표(x, y) 설정
          # relative_keypoints에서 가져온 x, y좌표와 h, w을 곱하면 이미지에서 원하는 좌표를 얻을 수 있음
          right_eye = (int(right_eye.x * w) - 100, int(right_eye.y * h) - 150) # x, y 좌표와 w, h의 곱을 정수형으로 변환 후 전체 튜플 형태로 변경
          left_eye = (int(left_eye.x * w) + 20, int(left_eye.y * h) - 150)
          nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))

          image[right_eye[1] - 50 : right_eye[1] + 50, right_eye[0] - 50 : right_eye[0] + 50] = image_right_eye # right_eye x, y좌표의 ±50 영역에 image_right_eye 넣어줌
          image[left_eye[1] - 50 : left_eye[1] + 50, left_eye[0] - 50 : left_eye[0] + 50] = image_left_eye
          image[nose_tip[1] - 50 : nose_tip[1] + 50, nose_tip[0] - 150 : nose_tip[0] + 150] = image_nose

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



      # Flip the image horizontally for a selfie-view display & Video resize 0.5
      cv2.imshow('Face Detection', cv2.resize(image, None, fx = 0.5, fy =0.5))

      if cv2.waitKey(1) == ord('q'):
        break
      
cap.release()
cv2.destroyAllWindows()