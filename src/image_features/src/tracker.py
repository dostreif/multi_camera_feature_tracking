#!/usr/bin/python3

# Start up ROS pieces.
package = 'image_features'
import roslib;

roslib.load_manifest(package)
import rospy
import cv2
from sensor_msgs.msg import Image
from image_features.msg import Features
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os
import sys
import math


# Jet colormap for visualization.
myjet = np.array([[0., 0., 0.5],
                  [0., 0., 0.99910873],
                  [0., 0.37843137, 1.],
                  [0., 0.83333333, 1.],
                  [0.30044276, 1., 0.66729918],
                  [0.66729918, 1., 0.30044276],
                  [1., 0.90123457, 0.],
                  [1., 0.48002905, 0.],
                  [0.99910873, 0.07334786, 0.],
                  [0.5, 0., 0.]])


class PointTracker(object):
    """ Class to manage a fixed memory of points and descriptors that enables
    sparse optical flow point tracking.

    Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
    tracks with maximum length L, where each row corresponds to:
    row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
    """

    def __init__(self, max_length, nn_thresh):
        if max_length < 2:
            raise ValueError('max_length must be greater than or equal to 2.')
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.all_pts = []
        for n in range(self.maxl):
            self.all_pts.append(np.zeros((2, 0)))
        self.last_desc = None
        self.last_pts = None
        self.tracks = np.zeros((0, self.maxl + 2))
        self.track_count = 0
        self.max_score = 9999

    def nn_match_two_way(self, desc1, desc2, nn_thresh):
        """
        Performs two-way nearest neighbor matching of two sets of descriptors, such
        that the NN match from descriptor A->B must equal the NN match from B->A.

        Inputs:
          desc1 - N1xM numpy matrix of N1 corresponding M-dimensional descriptors.
          desc2 - N2xM numpy matrix of N2 corresponding M-dimensional descriptors.
          nn_thresh - Optional descriptor distance below which is a good match.

        Returns:
          matches - 3xL numpy array, of L matches, where L <= N and each column i is
                    a match of two descriptors, d_i in image 1 and d_j' in image 2:
                    [d_i index, d_j' index, match_score]^T
        """
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        if nn_thresh < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')
        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < nn_thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx
        # Populate the final 3xN match data structure.
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores
        return matches


    def eightPointRansac(self, pts1, pts2):
        """
        Computes the fundamental matrix with the cv2 library findFundamentalMat with the RANSAC method after having
        normalized the coordinates to have zero centeroid and average distance to center of sqrt(2)
        (pts1' = T1*pts1, pts2' = T2*pts2, F = T2'*F'*T1)

        Inputs:
        pts1 - 2xN array of points in image1
        pts2 - 2xN array of points in image2 in the same order as the respective match in pts1

        Outputs:
        F - Fundamental matrix according to matches of pts1 and pts2
        """
        pts1 = pts1.T
        pts2 = pts2.T
        assert pts1.shape == pts2.shape
        n = pts1.shape[0]

        pts1_ext = np.ones((n, 3))
        pts1_ext[:, :-1] = pts1

        pts2_ext = np.ones((n, 3))
        pts2_ext[:, :-1] = pts2

        def normalize(pts):
            n = pts.shape[0]

            m = np.sum(pts, 0) / n
            s = n * math.sqrt(2) / np.sum(np.linalg.norm(pts - m, axis=1))

            T = np.array([[s, 0, -m[0] * s],
                          [0, s, -m[1] * s],
                          [0, 0, 1]])

            pts_ext = np.ones((n, 3))
            pts_ext[:, :-1] = pts

            pts_normed = (pts_ext @ T.transpose())[:, 0:2]

            return pts_normed, T

        pts1_normed, T1 = normalize(pts1)
        pts2_normed, T2 = normalize(pts2)
        F_normed, mask = cv2.findFundamentalMat(pts1_normed[:, 0:2], pts2_normed[:, 0:2], cv2.FM_RANSAC, 0.01)
        F = T2.transpose() @ F_normed @ T1
        return F, mask.T


    def get_offsets(self):
        """ Iterate through list of points and accumulate an offset value. Used to
        index the global point IDs into the list of points.

        Returns
          offsets - N length array with integer offset locations.
        """
        # Compute id offsets.
        offsets = []
        offsets.append(0)
        for i in range(len(self.all_pts) - 1):  # Skip last camera size, not needed.
            offsets.append(self.all_pts[i].shape[1])
        offsets = np.array(offsets)
        offsets = np.cumsum(offsets)
        return offsets

    def update(self, pts, desc):
        """ Add a new set of point and descriptor observations to the tracker.

        Inputs
          pts - 3xN numpy array of 2D point observations.
          desc - DxN numpy array of corresponding D dimensional descriptors.
        """
        if pts is None or desc is None:
            print('PointTracker: Warning, no points were added to tracker.')
            return
        assert pts.shape[1] == desc.shape[1]
        # Initialize last_desc.
        if self.last_desc is None:
            self.last_desc = np.zeros((desc.shape[0], 0))
        # Remove oldest points, store its size to update ids later.
        remove_size = self.all_pts[0].shape[1]
        self.all_pts.pop(0)
        self.all_pts.append(pts)
        # Remove oldest point in track.
        self.tracks = np.delete(self.tracks, 2, axis=1)
        # Update track offsets.
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size
        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        offsets = self.get_offsets()
        # Add a new -1 column.
        self.tracks = np.hstack((self.tracks, -1 * np.ones((self.tracks.shape[0], 1))))
        # Try to append to existing tracks.
        matched = np.zeros((pts.shape[1])).astype(bool)
        matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)

        # remove outlier matches with RANSAC
        if self.last_pts is not None:
            if matches.shape[1] >= 8:
                _, mask = self.eightPointRansac(self.last_pts[0:2, matches[0, :].astype(int)], pts[0:2, matches[1, :].astype(int)])
                matches = matches[:, mask.astype(bool).squeeze()]

        for match in matches.T:
            # Add a new point to it's matched track.
            id1 = int(match[0]) + offsets[-2]
            id2 = int(match[1]) + offsets[-1]
            found = np.argwhere(self.tracks[:, -2] == id1)
            if found.shape[0] > 0:
                matched[int(match[1])] = True
                row = int(found)
                self.tracks[row, -1] = id2
                if self.tracks[row, 1] == self.max_score:
                    # Initialize track score.
                    self.tracks[row, 1] = match[2]
                else:
                    # Update track score with running average.
                    # NOTE(dd): this running average can contain scores from old matches
                    #           not contained in last max_length track points.
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.
                    frac = 1. / float(track_len)
                    self.tracks[row, 1] = (1. - frac) * self.tracks[row, 1] + frac * match[2]
        # Add unmatched tracks.
        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]
        new_tracks = -1 * np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids
        new_num = new_ids.shape[0]
        new_trackids = self.track_count + np.arange(new_num)
        new_tracks[:, 0] = new_trackids
        new_tracks[:, 1] = self.max_score * np.ones(new_ids.shape[0])
        self.tracks = np.vstack((self.tracks, new_tracks))
        self.track_count += new_num  # Update the track count.
        # Remove empty tracks.
        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
        self.tracks = self.tracks[keep_rows, :]
        # Store the last descriptors and points.
        self.last_desc = desc.copy()
        self.last_pts = pts.copy()
        return

    def get_tracks(self, min_length):
        """ Retrieve point tracks of a given minimum length.
        Input
          min_length - integer >= 1 with minimum track length
        Output
          returned_tracks - M x (2+L) sized matrix storing track indices, where
            M is the number of tracks and L is the maximum track length.
        """
        if min_length < 1:
            raise ValueError('\'min_length\' too small.')
        valid = np.ones((self.tracks.shape[0])).astype(bool)
        good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
        # Remove tracks which do not have an observation in most recent frame.
        not_headless = (self.tracks[:, -1] != -1)
        keepers = np.logical_and.reduce((valid, good_len, not_headless))
        returned_tracks = self.tracks[keepers, :].copy()
        return returned_tracks

    def draw_tracks(self, out, tracks):
        """ Visualize tracks all overlayed on a single image.
        Inputs
          out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
          tracks - M x (2+L) sized matrix storing track info.
        """
        # Store the number of points per camera.
        pts_mem = self.all_pts
        N = len(pts_mem)  # Number of cameras/images.
        # Get offset ids needed to reference into pts_mem.
        offsets = self.get_offsets()
        # Width of track and point circles to be drawn.
        stroke = 1
        # Iterate through each track and draw it.
        for track in tracks:
            clr = myjet[int(np.clip(np.floor(track[1] * 10), 0, 9)), :] * 255
            for i in range(N - 1):
                if track[i + 2] == -1 or track[i + 3] == -1:
                    continue
                offset1 = offsets[i]
                offset2 = offsets[i + 1]
                idx1 = int(track[i + 2] - offset1)
                idx2 = int(track[i + 3] - offset2)
                pt1 = pts_mem[i][:2, idx1]
                pt2 = pts_mem[i + 1][:2, idx2]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
                # Draw end points of each track.
                if i == N - 2:
                    clr2 = (255, 0, 0)
                    cv2.circle(out, p2, stroke, clr2, -1, lineType=16)


class FeatureListener:
    def __init__(self):
        self.topic = 'features_descriptors/cam0'
        self.bridge = CvBridge()
        # This class helps merge consecutive point matches into tracks.
        self.tracker = PointTracker(5, nn_thresh=0.7)
        self.win = 'SuperPoint Tracker'

        # Font parameters for visualizaton.
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_clr = (255, 255, 255)
        self.font_pt = (4, 12)
        self.font_sc = 0.4

    def run(self):
        features_sub = message_filters.Subscriber(self.topic, Features)
        image_sub = message_filters.Subscriber('/bas_usb_0/image_raw', Image)

        ts = message_filters.TimeSynchronizer([image_sub, features_sub], queue_size=5)
        ts.registerCallback(self.callback)
        rospy.spin()

    def callback(self, image_msg, feature_msg):
        self.height = feature_msg.descriptors.layout.dim[0].size
        self.width = feature_msg.descriptors.layout.dim[1].size
        self.desc_size = feature_msg.descriptors.layout.dim[2].size

        pts = np.zeros((feature_msg.numKeypointMeasurements, 2))
        pts[:, 0] = np.array([feature_msg.keypointMeasurementsX])
        pts[:, 1] = np.array([feature_msg.keypointMeasurementsY])
        pts = pts.T

        desc_vector = np.array(feature_msg.descriptors.data)
        desc = desc_vector.reshape(feature_msg.descriptors.layout.dim[2].size, -1)
        print(pts.shape)
        print(desc.shape)
        print()

        cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'mono8')
        cv_image = cv2.resize(cv_image, (self.width, self.height))
        img = (cv_image.astype('float32') / 255.)
        self.track(img, pts, desc)

    def track(self, img, pts, desc):
        # Add points and descriptors to the tracker.
        self.tracker.update(pts, desc)

        # Get tracks for points which were match successfully across all frames.
        tracks = self.tracker.get_tracks(min_length=2)

        # Primary output - Show point tracks overlayed on top of input image.
        out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        tracks[:, 1] /= float(self.tracker.nn_thresh)  # Normalize track scores to [0,1].
        self.tracker.draw_tracks(out1, tracks)
        cv2.putText(out1, 'Point Tracks', self.font_pt, self.font, self.font_sc, self.font_clr, lineType=16)

        # Extra output -- Show current point detections.
        out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        for pt in pts.T:
            pt1 = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
        cv2.putText(out2, 'Raw Point Detections', self.font_pt, self.font, self.font_sc, self.font_clr, lineType=16)

        out = np.hstack((out1, out2))
        out = cv2.resize(out, (2 * self.width, self.height))
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
