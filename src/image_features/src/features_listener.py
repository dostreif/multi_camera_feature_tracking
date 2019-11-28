#!/usr/bin/python3

# Start up ROS pieces.
package = 'image_features'
import roslib;

roslib.load_manifest(package)
import rospy
import cv2
from sensor_msgs.msg import Image
from image_features.msg import Features
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os
import sys


class FeatureListener:
    def __init__(self):
        self.topic = 'features_descriptors/cam0'
        self.bridge = CvBridge()
        self.win = 'Features'

    def run(self):
        rospy.Subscriber(self.topic, Features, self.callback)
        rospy.spin()

    def callback(self, msg):
        self.height = msg.descriptors.layout.dim[0].size
        self.width = msg.descriptors.layout.dim[1].size
        self.desc_size = msg.descriptors.layout.dim[2].size

        pts = np.zeros((msg.numKeypointMeasurements, 2))
        pts[:, 0] = np.array([msg.keypointMeasurementsX])
        pts[:, 1] = np.array([msg.keypointMeasurementsY])
        cv_image = self.bridge.imgmsg_to_cv2(msg.rawImage, 'mono8')
        cv_image = cv2.resize(cv_image, (self.width, self.height))
        img = (cv_image.astype('float32') / 255.)
        self.show_image_features(img, pts)

    def show_image_features(self, img, pts):
        out = (np.dstack((img, img, img)) * 255.).astype('uint8')
        # draw interest points
        for k in range(0, pts.shape[0]):
            pt = (int(round(pts[k, 0])), int(round(pts[k, 1])))
            cv2.circle(out, pt, 1, (0, 255, 0), -1, lineType=16)
        cv2.namedWindow(self.win)
        cv2.imshow(self.win, out)
        cv2.waitKey(1)


# Main function.
if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node(package + 'listener')
    try:
        f_lis = FeatureListener()
        f_lis.run()
    except rospy.ROSInterruptException:
        pass
