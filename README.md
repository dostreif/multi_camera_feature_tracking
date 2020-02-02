Code repo for the feature detection and description algorithm in a ROS environment. It is part of the work of my semester thesis "Multi camera deep learned feature tracking".

A message type is defined in the image_features package. It handles visual frame informations including key point locations and the descriptors. Moreover, a python ROS node using this message type implemented in the image_features package. It synchronizes multiple camera messages and extract all the keypoint lcations and descriptors using a neural network as described in my thesis.

In the net_inference package, they network inference code is implemented using C++ [WIP]. Essentially this code snippet can be used to implement the same functionality as the python ROS code. However, the C++ code needs further optimization.
