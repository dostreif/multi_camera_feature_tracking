cmake_minimum_required(VERSION 2.8)
project(net_inference)

find_package(catkin_simple REQUIRED
	     cv_bridge
	     opencv3_catkin
	     eigen_catkin)

catkin_simple()

cs_add_executable(inference src/inference.cpp)

cs_install()
cs_export()
