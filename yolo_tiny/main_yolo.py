try:
    from cv2 import cv2
except ImportError:
    import sys

    ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
    sys.path.remove(ros_path)
    from cv2 import cv2

    print(cv2.__version__)
    sys.path.append(ros_path)
import numpy as np
import time
import geometry_msgs.msg
from geometry_msgs.msg import Twist
import rospy


class yolo:
    def __init__(self, config):
        print('Net use', config['netname'])
        self.confThreshold = config['confThreshold']
        self.nmsThreshold = config['nmsThreshold']
        self.inpWidth = config['inpWidth']
        self.inpHeight = config['inpHeight']
        with open(config['classesFile'], 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        self.net = cv2.dnn.readNet(config['modelConfiguration'], config['modelWeights'])
        self.stop = False
        self.turn_mode = False
        self.left = Twist()
        self.left.linear.x = 0.5
        self.left.angular.z = 0.3
        self.right = Twist()
        self.right.linear.x = 0.5
        self.right.angular.z = -0.3
        self.forward = Twist()
        self.forward.linear.x = 0.5
        self.backward = Twist()
        self.backward.linear.x = -0.5

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=3)
        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]),
        # top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        classIds = []
        confidences = []
        boxes = []
        results = []
        centerxy_0 = []
        centerxy_1 = []
        centerx_0 = []
        centerx_1 = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
            center_x = left + round(width / 2)
            center_y = top + round(height / 2)
            results.append([classIds[i], center_x, center_y])
        if len(results) > 0:
            for result in results:
                if result[0] == 0:  # and result[1] * result[2] > frameWidth * frameHeight * 0.1:
                    centerxy_0.append((result[1], result[2]))
                elif result[0] == 1:  # and result[1] * result[2] > frameWidth * frameHeight * 0.1:
                    centerxy_1.append((result[1], result[2]))
            centerxy_0.sort()
            centerxy_1.sort()
            for point in centerxy_0:
                centerx_0.append(point[0])
            for point in centerxy_1:
                centerx_1.append(point[0])
            if len(centerxy_0) > 1:
                for i in range(0, len(centerxy_0) - 1):
                    cv2.line(frame, centerxy_0[i], centerxy_0[i + 1], color=(255, 0, 0), thickness=3)
            if len(centerxy_1) > 1:
                for i in range(0, len(centerxy_1) - 1):
                    cv2.line(frame, centerxy_1[i], centerxy_1[i + 1], color=(0, 255, 0), thickness=3)
            if self.stop == False and self.turn_mode == False:
                if len(centerx_0) > 0:
                    mean_0 = sum(centerx_0) / len(centerx_0)
                else:
                    mean_0 = 0
                if len(centerx_1) > 0:
                    mean_1 = min(centerx_1)
                else:
                    mean_1 = frameWidth
                self.forward.angular.z = (mean_0 / frameWidth * -0.5 + (1 - mean_1 / frameWidth) * 0.7) * 1.8
                print(frameWidth)
                vel_pub.publish(self.forward)
                if self.forward.angular.z > 0.05:
                    cv2.putText(frame, "<---turn left<---", (round(frame.shape[1] / 10), round(frame.shape[0] * 0.9)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
                elif self.forward.angular.z < -0.05:
                    cv2.putText(frame, "--->turn right--->", (round(frame.shape[1] / 10), round(frame.shape[0] * 0.9)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
                else:
                    cv2.putText(frame, "go forward", (round(frame.shape[1] / 10), round(frame.shape[0] * 0.9)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
                print(self.forward)
                print(mean_0)
                print(mean_1)
            elif self.turn_mode:
                vel_pub.publish(self.left)
                cv2.putText(frame, "<---turn left<---", (round(frame.shape[1] / 10), round(frame.shape[0] * 0.9)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
            else:
                vel_pub.publish(self.backward)
                cv2.putText(frame, "stop and backward", (round(frame.shape[1] / 10), round(frame.shape[0] * 0.9)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
                print(self.backward)
        else:
            cv2.putText(frame, "nothing found", (round(frame.shape[1] / 10), round(frame.shape[0] * 0.9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

    def detect(self, src_img):
        blob = cv2.dnn.blobFromImage(src_img, 1 / 255.0, (self.inpWidth, self.inpHeight), [0, 0, 0], swapRB=True,
                                     crop=False)
        # Sets the input to the network
        self.net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        self.postprocess(src_img, outs)
        return src_img

    def color_space(self, img):
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        low_red = np.array([0, 43, 46])
        high_red = np.array([13, 255, 255])
        mask = cv2.inRange(hsv, low_red, high_red)
        kernelSize = [(3, 3), (5, 5), (7, 7)]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize[2])
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        opening = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
        max_area = 0
        for i in range(length):
            cnt = contours[i]
            if cv2.contourArea(cnt) > max_area:
                max_area = cv2.contourArea(cnt)
            epsilon = 0.00001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(opening, approx, contourIdx=-1, color=(0, 0, 255), thickness=3)
        if max_area > 0.28 * img.shape[0] * img.shape[1]:
            self.stop = True
            self.turn_mode = False
        elif max_area > 0.12 * img.shape[0] * img.shape[1]:
            self.turn_mode = True
            self.stop = False
        else:
            self.stop = False
            self.turn_mode = False
        cv2.putText(opening, "max red area={}".format(max_area / frame.shape[1] / opening.shape[0]),
                    (round(opening.shape[1] / 10), round(opening.shape[0] * 0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        windowname = 'red'
        cv2.namedWindow(windowname, 0)
        cv2.resizeWindow(windowname, 640, 640)
        cv2.moveWindow(windowname, 10, 10)
        cv2.imshow(windowname, opening)


Net_config = [
    {'confThreshold': 0.5, 'nmsThreshold': 0.4, 'inpWidth': 320, 'inpHeight': 320, 'classesFile': 'smart.names',
     'modelConfiguration': 'yolov4-tiny/yolov4-tiny.cfg', 'modelWeights': 'yolov4-tiny/yolov4-tiny_last.weights',
     'netname': 'yolov4-tiny'}]

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    net_type = 0
    yolo_net = yolo(Net_config[net_type])
    rospy.init_node('vision_nav', anonymous=True)
    # rate = rospy.Rate(10)
    vel_pub = rospy.Publisher('cmd_vel', geometry_msgs.msg.Twist, queue_size=10)
    while not rospy.is_shutdown():
        time1 = time.time()
        ret, frame = cap.read()
        if ret == 1:
            # size = (640, 500)
            # frame = cv2.resize(frame, size)
            # frame = frame[0:round(frame.shape[0]), round(frame.shape[1] / 2):round(frame.shape[1])]
            yolo_net.color_space(frame)
            src_img = yolo_net.detect(frame)
            time2 = time.time()
            fps = 1 / (time2 - time1)
            text = 'FPS:%.2f' % fps
            winName = 'Deep learning object detection in OpenCV'
            cv2.putText(src_img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.namedWindow(winName, 0)
            cv2.resizeWindow(winName, 640, 640)
            cv2.moveWindow(winName, 720, 10)
            cv2.imshow(winName, src_img)
            cv2.waitKey(10)
        else:
            exit(0)
    # rospy.init_node('vision_nav', anonymous=True)
    # vel_pub = rospy.Publisher('cmd_vel', geometry_msgs.msg.Twist, queue_size=10)
    # yolonet = yolo(Net_config[5])
    # srcimg = cv2.imread("./test.jpg")
    # srcimg = yolonet.detect(srcimg)
    #
    # winName = 'Deep learning object detection in OpenCV'
    # cv2.namedWindow(winName, 0)
    # cv2.imshow(winName, srcimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
