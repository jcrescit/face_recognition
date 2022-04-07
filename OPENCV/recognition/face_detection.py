import cv2
import mediapipe as mp

# 얼굴을 찾고, 찾은 얼굴에 표시를 해주기 위한 변수 정의
mp_face_detection = mp.solutions.face_detection # 얼굴 검출을 위한 face_detection 모듈 사용
mp_drawing = mp.solutions.drawing_utils # 얼굴의 특징을 그리기 위한 drawing_utils 모듈 사용

# 동영상 파일 열기
cap = cv2.VideoCapture('D:/Computer_Vision/opencv_study/recognition/video/face_video.mp4')

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.7) as face_detection:
    # model_selection = 0 : 카메라로부터 2m 이내의 근거리에 적합, =1 : 5m 이내의 근거리에 적합
    # im_detection_confidence : 신뢰도가 특정 %이상되면, 얼굴로 인식할지 정의 (threshold와 유사한 개념)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            # If loading a video, use 'break' instead of 'continue'.
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
            # 6개 특징 : 오른쪽 눈, 왼쪽 눈, 코 끝부분, 입 중심, 오른쪽 귀, 왼쪽 귀
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                # print(detection)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', cv2.resize(image, None, fx=0.5, fy=0.5))

        if cv2.waitKey(25) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

'''
detection 정보

label_id: 0
score: 0.972063422203064    # 신뢰도
location_data {
  format: RELATIVE_BOUNDING_BOX
  relative_bounding_box {
    xmin: 0.34010049700737
    ymin: 0.27738773822784424
    width: 0.16470429301261902
    height: 0.2928076982498169
  }
  relative_keypoints { # 얼굴에서 찾아내는 6개 특징의 좌표
    x: 0.39748260378837585
    y: 0.3514038920402527
  }
  relative_keypoints {
    x: 0.472693532705307
    y: 0.3491246700286865
  }
  relative_keypoints {
    x: 0.4475392699241638
    y: 0.41251301765441895
  }
  relative_keypoints {
    x: 0.4445818364620209
    y: 0.4767593741416931
  }
  relative_keypoints {
    x: 0.3378288745880127
    y: 0.3956125378087036
  }
  relative_keypoints {
    x: 0.3649260103702545
    y: 0.3476579189300537
  }
}

'''