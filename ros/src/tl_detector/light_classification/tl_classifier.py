import rospy
import os
import shutil
import numpy as np
import math

import tensorflow as tf
import cv2

from glob import iglob
from datetime import datetime
from collections import Counter
from styx_msgs.msg import TrafficLight


# Path to frozen detection graph. This is the actual model that is used for the object detection.
# http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
PATH_TO_CHECKPOINT = 'frozen_inference_graph.pb'
TRAFFIC_LIGHT_CLASS = 10
MIN_SCORE_THRESHOLD = .75
MAX_DETECTIONS = 10
MIN_BOX_WIDTH = 15
MAX_BOX_RATIO = 0.55

TRAFFIC_LIGHT_SHAPE = (8, 24)
HSV_V_THRESHOLD = 210

class TLClassifier(object):
    def __init__(self):
        self.w, self.h = TRAFFIC_LIGHT_SHAPE

        self.h1 = self.h // 3
        self.h2 = (2 * self.h) // 3

        # if frozen_inference_graph.pb does not exist, build it from chunks
        if not os.path.exists(PATH_TO_CHECKPOINT):
            rospy.loginfo("Creating Tensorflow inference graph from chunks...")
            destination = open(PATH_TO_CHECKPOINT, 'wb')
            for filename in sorted(iglob('graph_chunks/chunk*')):
                rospy.loginfo("Adding %s to %s", filename, PATH_TO_CHECKPOINT)
                shutil.copyfileobj(open(filename, 'rb'), destination)
            destination.close()
            rospy.loginfo("%s successfully created from chunks", PATH_TO_CHECKPOINT)

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CHECKPOINT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.session = tf.Session(graph=self.detection_graph)

    def get_traffic_light_state(self, image):

        resized = cv2.resize(image, TRAFFIC_LIGHT_SHAPE)
        hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)

        lower_red = np.array([150, 0, HSV_V_THRESHOLD])
        upper_red = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv[:self.h1, :], lower_red, upper_red)

        lower_red = np.array([0, 0, HSV_V_THRESHOLD])
        upper_red = np.array([30, 255, 255])
        red_mask2 = cv2.inRange(hsv[:self.h1, :], lower_red, upper_red)

        red_mask = red_mask1 + red_mask2

        lower_yellow = np.array([0, 0, HSV_V_THRESHOLD])
        upper_yellow = np.array([60, 255, 255])
        yellow_mask = cv2.inRange(hsv[self.h1:self.h2, :], lower_yellow, upper_yellow)

        lower_green = np.array([30, 0, HSV_V_THRESHOLD])
        upper_green = np.array([100, 255, 255])
        green_mask = cv2.inRange(hsv[self.h2:, :], lower_green, upper_green)

        color_mask = np.vstack((red_mask, yellow_mask, green_mask))

        lower_white = np.array([0, 0, 245])
        upper_white = np.array([180, 10, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        tl_mask = cv2.bitwise_or(white_mask, color_mask)

        v_ch = hsv[:, :, 2]

        v_ch[tl_mask == 0] = 0

        # DEBUG BEGIN
        # hsv[tl_mask == 0] = 0
        # save_image = np.hstack((cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
        #                         cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)))
        # file_name = "debug/tl{}.jpg".format(datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3])
        # cv2.imwrite(file_name, save_image)
        # DEBUG END

        intensity = np.hstack((np.sum(np.sum(v_ch[1:self.h1 - 1, 1:-1], axis=1, dtype=np.float32)),
                               np.sum(np.sum(v_ch[self.h1 + 1:self.h2 - 1, 1:-1], axis=1, dtype=np.float32)),
                               np.sum(np.sum(v_ch[self.h2 + 1:-1, 1:-1], axis=1, dtype=np.float32))))

        if intensity[0] == intensity[1] and intensity[1] == intensity[2]:
            return TrafficLight.UNKNOWN

        return {
            0: TrafficLight.RED,
            1: TrafficLight.YELLOW,
            2: TrafficLight.GREEN,
        }[np.argmax(intensity)]

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        tl_states = []
        with self.detection_graph.as_default():
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            im_height, im_width, _ = image.shape
            image_input = np.expand_dims(image, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = self.session.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_input})
            # Visualization of the results of a detection.
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            sorted_indexes = np.argsort(scores)

            for i in reversed(sorted_indexes):
                if classes[i] == TRAFFIC_LIGHT_CLASS and scores[i] >= MIN_SCORE_THRESHOLD:
                    ymin, xmin, ymax, xmax = (
                        int(math.floor(boxes[i][0] * im_height)), int(math.floor(boxes[i][1] * im_width)),
                        int(math.ceil(boxes[i][2] * im_height)), int(math.ceil(boxes[i][3] * im_width))
                    )
                    box_h = ymax - ymin
                    box_w = xmax - xmin
                    box_ratio = box_w / box_h

                    if (box_w >= MIN_BOX_WIDTH) and (box_ratio <= MAX_BOX_RATIO):
                        tl_image = np.copy(image[ymin:ymax, xmin:xmax, :])
                        state = self.get_traffic_light_state(tl_image)
                        #rospy.loginfo("Traffic light detection (%d,%d,%d,%d)->%d", xmin, ymin, xmax, ymax, int(state))

                        tl_states.append(state)

                        if len(tl_states) >= MAX_DETECTIONS:
                            break

        if len(tl_states) > 0:
            data = Counter(tl_states)
            return data.most_common(1)[0][0]

        return TrafficLight.UNKNOWN
