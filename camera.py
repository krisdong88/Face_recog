import cv2

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 加载人脸识别的Haar级联模型
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

try:
    while True:
        # 捕获摄像头的一帧
        ret, frame = cap.read()

        # 如果正确读取帧，ret为True
        if not ret:
            print("无法读取视频流")
            break

        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测图像中的人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # 在检测到的人脸周围绘制矩形框
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 显示帧
        cv2.imshow('摄像头 - 带人脸识别', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
