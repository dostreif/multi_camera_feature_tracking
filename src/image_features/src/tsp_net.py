import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.backend import set_session
import torch
import tensorflow as tf
import math


class TSP_Net:
    """ Keras definition of SuperPoint Network. """

    def __init__(self, height, width, desc_size, cuda_avail=False):
        global graph
        global sess
        graph = tf.get_default_graph()
        sess = tf.Session()

        set_session(sess)

        self.H = height
        self.W = width
        self.cuda_avail = cuda_avail

        c1, c2, c3, c4, c5 = 32, 32, 64, 64, 128
        input = Input(shape=(self.H, self.W, 1), name='input')
        conv1a = Conv2D(c1, kernel_size=(3, 3), activation='relu', padding='same', trainable=True)(input)
        conv1b = Conv2D(c1, kernel_size=(3, 3), activation='relu', padding='same', trainable=True)(conv1a)
        pool1 = MaxPooling2D((2, 2), strides=2)(conv1b)
        conv2a = Conv2D(c2, kernel_size=(3, 3), activation='relu', padding='same', trainable=True)(pool1)
        conv2b = Conv2D(c2, kernel_size=(3, 3), activation='relu', padding='same', trainable=True)(conv2a)
        pool2 = MaxPooling2D((2, 2), strides=2)(conv2b)
        conv3a = Conv2D(c3, kernel_size=(3, 3), activation='relu', padding='same', trainable=True)(pool2)
        conv3b = Conv2D(c3, kernel_size=(3, 3), activation='relu', padding='same', trainable=True)(conv3a)
        pool3 = MaxPooling2D((2, 2), strides=2)(conv3b)
        conv4a = Conv2D(c4, kernel_size=(3, 3), activation='relu', padding='same', trainable=True)(pool3)
        enc = Conv2D(c4, kernel_size=(3, 3), activation='relu', padding='same', name='enc', trainable=True)(conv4a)

        # detector head
        convH = Conv2D(c5, kernel_size=(3, 3), activation='relu', padding='same', name='convH')(enc)
        heat = Conv2D(65, kernel_size=(1, 1), activation=None, padding='valid', name='heat')(convH)

        # descriptor head
        convDa = Conv2D(c5, kernel_size=(3, 3), activation='relu', padding='same', name='convD')(enc)
        desc = Conv2D(desc_size, kernel_size=(1, 1), activation=None, padding='valid', name='desc')(convDa)

        self.model = Model(inputs=input,
                           outputs=[heat, desc],
                           name="superpoint")
        self.model.summary()

    def run_raw(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
        x: Image numpy tensor shaped N x H x W x 1.
        Output
        semi: Output point numpy tensor shaped N x 65 x H/8 x W/8.
        desc: Output descriptor numpy tensor shaped N x 256 x H/8 x W/8.
        """

        inp = x.copy()
        inp = inp.reshape(-1, self.H, self.W, 1)
        with graph.as_default():
            set_session(sess)
            heat_dense, desc_coarse = self.model.predict_on_batch(inp)
        return heat_dense.squeeze(), desc_coarse.squeeze()

    def nms_fast(self, in_corners, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((self.H, self.W)).astype(int)  # Track NMS data.
        inds = np.zeros((self.H, self.W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def post_process(self, img, heatmap_dense, coarse_desc, conf_thresh=0.015, nms_dist=4, border_remove_pixels=4):
        """ Process a numpy image to extract points and descriptors.
                Input
                  img - H x W numpy float32
                  heatmap_dense - (H/8 x W/8 x 65) numpy float32.
                  coarse_desc = (H/8 x W/8 x desc_size) numpy float32
                  conf_thresh - confidence threshold to determine which points are selected as corners from heatmap
                  nms_dist - Distance to suppress in nms algorithm, measured as an infinty norm distance
                  border_remove_pixels - how many pixels at border are ignored for corner selection
                                        (due to convolution artifacts at border)
                Output
                  corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
                  desc - desc_sizexN numpy array of corresponding unit normalized descriptors.
                  heatmap - HxW numpy heatmap in range [0,1] of point confidences.
                  """
        semi = heatmap_dense.squeeze().transpose(2, 0, 1)
        coarse_desc = coarse_desc.squeeze().transpose(2, 0, 1)

        # --- Process points.
        dense = np.exp(semi)  # Softmax.
        dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(math.ceil(self.H / 8))
        Wc = int(math.ceil(self.W / 8))
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, 8, 8])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * 8, Wc * 8])
        xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, dist_thresh=nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.

        # # Set the needed parameters to find the refined corners
        # winSize = (5, 5)
        # zeroZone = (-1, -1)
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
        #
        # # Calculate the refined corner locations
        # refined_corners = cv2.cornerSubPix(img.astype(np.float32).copy(), np.expand_dims(pts[:2, :], axis=1).astype(np.float32).transpose().copy(), winSize, zeroZone, criteria)
        # pts[:2, :] = refined_corners.astype(int).transpose().reshape(2, -1)

        # Remove points along border.
        bord = border_remove_pixels
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (self.W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (self.H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]

        # --- Process descriptor.
        D = coarse_desc.shape[0]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            coarse_desc = torch.from_numpy(np.expand_dims(coarse_desc, 0))
            samp_pts[0, :] = (samp_pts[0, :] / (float(self.W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(self.H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            if self.cuda_avail:
                coarse_desc = coarse_desc.cuda()
                samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.data.cpu().numpy().reshape(D, -1)

            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
            min_conf = 0.001
            heatmap[heatmap < min_conf] = min_conf
            heatmap = -np.log(heatmap)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
        return pts, desc, heatmap

    def load_weights(self):
        self.model.load_weights('/home/dominic/supermarionet_ros/src/image_features/src/half_encoder_half_decoder_64.h5')
