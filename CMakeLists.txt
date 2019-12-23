cmake_minimum_required(VERSION 3.15)
project(supermarionet_ros)

set(CMAKE_CXX_STANDARD 14)

include_directories(devel)
include_directories(devel/include)
include_directories(devel/include/image_features)
include_directories(src)

add_executable(supermarionet_ros
        devel/include/image_features/Features.h
        src/image_features/src/inference.cpp)
