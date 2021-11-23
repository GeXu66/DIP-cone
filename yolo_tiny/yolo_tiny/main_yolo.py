try:
    import cv2
except ImportError:
    import sys
    ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
    sys.path.remove(ros_path)
    import cv2
    print(cv2.__version__)
    sys.path.append(ros_path)
import numpy as np
import time

import geometry_msgs.msg
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

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
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
                if result[0] == 0:
                    centerxy_0.append((result[1], result[2]))
                else:
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
            if len(centerx_0) > 0:
                max_0 = sum(centerx_0) / len(centerx_0)
            else:
                max_0 = 0
            if len(centerx_1) > 0:
                min_1 = sum(centerx_1) / len(centerx_1)
            else:
                min_1 = frameWidth
            vel = geometry_msgs.msg.Twist()
            vel.linear.x = 0.2
            vel.linear.y = 0
            vel.linear.z = 0
            vel.angular.x = 0
            vel.angular.y = 0
            vel.angular.z = max_0 / frameWidth * -0.6 + (1 - min_1 / frameWidth) * 1.3
            vel_pub.publish(vel)
            print(vel)
            print(max_0)
            print(min_1)

    def detect(self, src_img):
        blob = cv2.dnn.blobFromImage(src_img, 1 / 255.0, (self.inpWidth, self.inpHeight), [0, 0, 0], swapRB=True,
                                     crop=False)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        self.postprocess(src_img, outs)
        return src_img


Net_config = [
    {'confThreshold': 0.5, 'nmsThreshold': 0.4, 'inpWidth': 416, 'inpHeight': 416, 'classesFile': 'coco.names',
     'modelConfiguration': 'yolov3/yolov3.cfg', 'modelWeights': 'yolov3/yolov3.weights', 'netname': 'yolov3'},
    {'confThreshold': 0.5, 'nmsThreshold': 0.4, 'inpWidth': 608, 'inpHeight': 608, 'classesFile': 'coco.names',
     'modelConfiguration': 'yolov4/yolov4.cfg', 'modelWeights': 'yolov4/yolov4.weights', 'netname': 'yolov4'},
    {'confThreshold': 0.5, 'nmsThreshold': 0.4, 'inpWidth': 320, 'inpHeight': 320, 'classesFile': 'coco.names',
     'modelConfiguration': 'yolo-fastest/yolo-fastest-xl.cfg', 'modelWeights': 'yolo-fastest/yolo-fastest-xl.weights',
     'netname': 'yolo-fastest'},
    {'confThreshold': 0.5, 'nmsThreshold': 0.4, 'inpWidth': 320, 'inpHeight': 320, 'classesFile': 'coco.names',
     'modelConfiguration': 'yolobile/csdarknet53s-panet-spp.cfg', 'modelWeights': 'yolobile/yolobile.weights',
     'netname': 'yolobile'},
    {'confThreshold': 0.5, 'nmsThreshold': 0.4, 'inpWidth': 320, 'inpHeight': 320, 'classesFile': 'coco.names',
     'modelConfiguration': 'yolov3-tiny/yolov3-tiny.cfg', 'modelWeights': 'yolov3-tiny/yolov3-tiny.weights',
     'netname': 'yolov3-tiny'},
    {'confThreshold': 0.5, 'nmsThreshold': 0.4, 'inpWidth': 320, 'inpHeight': 320, 'classesFile': 'smart.names',
     'modelConfiguration': 'yolov4-tiny/yolov4-tiny.cfg', 'modelWeights': 'yolov4-tiny/yolov4-tiny_last.weights',
     'netname': 'yolov4-tiny'}]

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    net_type = 5
    yolo_net = yolo(Net_config[net_type])
    rospy.init_node('vision_nav', anonymous=True)
    # rate = rospy.Rate(10)
    vel_pub = rospy.Publisher('cmd_vel', geometry_msgs.msg.Twist, queue_size=10)
    while not rospy.is_shutdown():
        time1 = time.time()
        ret, frame = cap.read()
        frame = frame[0:round(frame.shape[0]), round(frame.shape[1]/2):round(frame.shape[1])]
        src_img = yolo_net.detect(frame)
        time2 = time.time()
        fps = 1 / (time2 - time1)
        text = 'FPS:%.2f' % fps
        winName = 'Deep learning object detection in OpenCV'
        cv2.putText(src_img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.namedWindow(winName, 0)
        cv2.resizeWindow(winName, 640, 640)
        cv2.imshow(winName, src_img)
        cv2.waitKey(10)
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
