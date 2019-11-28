#!/usr/bin/python3

# Start up ROS pieces.
package = 'image_features'
import roslib;

roslib.load_manifest(package)
import rospy
import cv2
from sensor_msgs.msg import Image
from image_features.msg import Features
from std_msgs.msg import MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os
import sys
import tsp_net


class FeaturePublisher:
    """
    Synchronizes multiple camera topics and publishes the features and descriptors of synchronised images.
    """
    def __init__(self):
        # define a topic for each camera image
        self.topic0 = 'features_descriptors/cam0'
        self.topic1 = 'features_descriptors/cam1'
        self.pub_cam0 = rospy.Publisher(self.topic0, Features, queue_size=10)
        self.pub_cam1 = rospy.Publisher(self.topic1, Features, queue_size=10)
        # subscribe to all camera feeds
        self.image_topic0 = "/VersaVIS/cam0/image_raw"
        self.image_topic1 = "/VersaVIS/cam1/image_raw"
        # define publish rate (Hz)
        self.rate = rospy.Rate(2)
        self.bridge = CvBridge()
        # define superpoint model
        self.width = 640
        self.height = 480
        self.desc_size = 64
        self.net = tsp_net.TSP_Net(height=self.height, width=self.width, desc_size=self.desc_size, cuda_avail=False)
        self.net.load_weights()
        self.win = 'Heatmap'
        self.myjet = np.array([[0., 0., 0.5],
                  [0., 0., 0.99910873],
                  [0., 0.37843137, 1.],
                  [0., 0.83333333, 1.],
                  [0.30044276, 1., 0.66729918],
                  [0.66729918, 1., 0.30044276],
                  [1., 0.90123457, 0.],
                  [1., 0.48002905, 0.],
                  [0.99910873, 0.07334786, 0.],
                  [0.5, 0., 0.]])
        self.processing = False
        # initialize multi camera handler
        self.image_avail = [False, False]
        self.image_timestamps = [0, 0]
        self.images = np.zeros((2, self.height, self.width))

    def callback_cam0(self, img):
        self.image_handler(img, 0)

    def callback_cam1(self, img):
        self.image_handler(img, 1)

    def image_handler(self, img, cam_id):
        """
        Called by callback functions, handles synchronisation of incoming images
        :param img: image from camera callback
        :param cam_id: which camera callback called this function
        """
        if not self.image_avail[cam_id]:
            if rospy.Time.to_sec(img.header.stamp) - self.image_timestamps[cam_id] > 1/20:
                print('set cam%u to true' % cam_id)
                self.image_timestamps[cam_id] = rospy.Time.to_sec(img.header.stamp)
                cv_image = self.bridge.imgmsg_to_cv2(img, "mono8")
                cv_image = cv2.resize(cv_image, (self.width, self.height))
                self.images[cam_id, :, :] = (cv_image.astype('float32') / 255.)
                self.image_avail[cam_id] = True
        if sum(self.image_avail) == len(self.image_avail) and not self.processing:
            self.get_features()

    def get_features(self):
        """
        Processes images in self.images to get their features and descriptors to publish them
        """
        self.processing = True
        semi, desc = self.net.run_raw(self.images)
        pts = [None, None]
        descriptors = [None, None]
        for i in range(0, len(self.image_avail)):
            pts[i], descriptors[i], heatmap = self.net.post_process(self.images[i, :, :], semi[i, :, :, :], desc[i, :, :, :], conf_thresh=0.001)

        # Show debug images
        # out1 = self.myjet[np.round(np.clip(heatmap * 10, 0, 9)).astype('int'), :]
        # out1 = (out1 * 255).astype('uint8')
        # out2 = (np.dstack((grayim, grayim, grayim)) * 255.).astype('uint8')
        # # draw interest points
        # for k in range(0, pts.shape[1]):
        #     pt = (int(round(pts[0, k])), int(round(pts[1, k])))
        #     cv2.circle(out2, pt, 1, (0, 255, 0), -1, lineType=16)
        # out = np.hstack((out1, out2))
        # cv2.namedWindow(self.win)
        # cv2.imshow(self.win, out)
        # cv2.waitKey(1)

        # publish messages
        self.publish_feature_msg(pts[0].squeeze(), descriptors[0].squeeze(), self.images[0, :, :].squeeze(), self.pub_cam0)
        self.publish_feature_msg(pts[1].squeeze(), descriptors[1].squeeze(), self.images[1, :, :].squeeze(), self.pub_cam1)
        print("published")
        self.image_avail = [False, False]
        self.processing = False
        self.rate.sleep()

    def run(self):
        rospy.Subscriber(self.image_topic0, Image, self.callback_cam0)
        rospy.Subscriber(self.image_topic1, Image, self.callback_cam1)
        rospy.spin()

    def set_ros_multi_array_dims_descriptor(self, msg):
        """
        initialize multiarray message type
        :param msg: msg in which multiarray needs to be initialized
        :return: initialized message
        """
        msg.descriptors.layout.data_offset = 0
        msg.descriptors.layout.dim.append(MultiArrayDimension())
        msg.descriptors.layout.dim.append(MultiArrayDimension())
        msg.descriptors.layout.dim.append(MultiArrayDimension())
        msg.descriptors.layout.dim[0].label = "height"
        msg.descriptors.layout.dim[0].size = self.height
        msg.descriptors.layout.dim[0].stride = self.height * self.width * self.desc_size
        msg.descriptors.layout.dim[1].label = "width"
        msg.descriptors.layout.dim[1].size = self.width
        msg.descriptors.layout.dim[1].stride = self.width * self.desc_size
        msg.descriptors.layout.dim[2].label = "desc_size"
        msg.descriptors.layout.dim[2].size = self.desc_size
        msg.descriptors.layout.dim[2].stride = self.desc_size
        return msg

    def publish_feature_msg(self, pts, descriptors, img, pub):
        """
        Set message parameters and publish message
        :param pts: keypoints to be publlished
        :param descriptors: corresponding descriptors
        :param img: corersponding image
        :param pub: publisher object in which the message is published
        """
        msg = Features()
        msg.hasKeypointMeasurements = True
        msg.hasKeypointMeasurementUncertainties = False
        msg.hasKeypointOrientations = False
        msg.hasKeypointScores = True
        msg.hasKeypointScales = False
        msg.hasFloatDescriptors = True
        msg.hasBoolDescriptors = False
        msg.hasTrackIds = False
        msg.hasRawImage = False
        msg.hasChannel = False
        msg.keypointMeasurementsX = pts[0, :].astype(int).tolist()
        msg.keypointMeasurementsY = pts[1, :].astype(int).tolist()
        msg.keypointScores = pts[2, :].tolist()
        msg.numKeypointMeasurements = pts.shape[1]
        msg.keypointMeasurementUncertainties = [-1]
        msg.keypointOrientations = [-1]
        msg.keypointScales = [-1]
        descriptors = descriptors.reshape(-1)
        msg = self.set_ros_multi_array_dims_descriptor(msg)
        msg.descriptors.data = descriptors.tolist()
        msg.trackIds = [-1]
        msg.rawImage = self.bridge.cv2_to_imgmsg((img*255).astype(np.uint8), "mono8")
        pub.publish(msg)


# Main function.
if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node(package)
    try:
        f_pub = FeaturePublisher()
        f_pub.run()
    except rospy.ROSInterruptException:
        pass
