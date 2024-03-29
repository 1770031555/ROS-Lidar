cmake_minimum_required(VERSION 2.8.3)
project(cam_lidar_calibration)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(catkin REQUIRED COMPONENTS nodelet
        roscpp
        std_msgs
        pcl_conversions
        pcl_ros
        image_transport
        cv_bridge
        sensor_msgs
        message_generation
        cmake_modules
        darknet_ros_msgs
        dynamic_reconfigure
        )

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen REQUIRED)
## Setup include directories
include_directories(
        ${catkin_INCLUDE_DIRS}
        include/${PROJECT_NAME}
        ${OpenCV_INCLUDE_DIRS}
        ${Eigen_INCLUDE_DIRS}
        src
)

#add_message_files(
#  FILES
#  calibration_data.msg
#  extrinsics.msg
#		BoundingBox.msg
#		BoundingBoxes.msg
#)

generate_messages(
        DEPENDENCIES
        std_msgs
        sensor_msgs
)

generate_dynamic_reconfigure_options(
        cfg/bounds.cfg
)

catkin_package(
        CATKIN_DEPENDS nodelet
        roscpp
        std_msgs
        sensor_msgs
        message_runtime
        darknet_ros_msgs
        DEPENDS system_lib Eigen
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#add_executable(input_sample src/input_sample.cpp)
#target_link_libraries(input_sample ${catkin_LIBRARIES})
#
#add_executable(optimizer src/optimizer.cpp)
#target_link_libraries(optimizer ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Eigen_LIBRARIES})
#add_dependencies(optimizer cam_lidar_calibration_generate_messages_cpp)

#add_executable(projector src/projector.cpp)
#target_include_directories(projector PRIVATE
#		include)
#target_link_libraries(projector ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Eigen_LIBRARIES})

add_executable(projector
        src/projector.cpp
        )
target_include_directories(projector PRIVATE
        include
        )
target_link_libraries(projector
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${Eigen_LIBRARIES}
        )

add_dependencies(projector
        #		cam_lidar_calibration_generate_messages_cpp
        darknet_ros_msgs_generate_messages_cpp)


## Create the nodelet tutorial library
#add_library(feature_extraction src/feature_extraction.cpp include/cam_lidar_calibration/projector.h)
#add_dependencies(feature_extraction ${PROJECT_NAME}_gencfg)
#
#target_link_libraries(feature_extraction ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Eigen_LIBRARIES})
#if(catkin_EXPORTED_LIBRARIES)
#  add_dependencies(feature_extraction ${catkin_EXPORTED_LIBRARIES})
#endif()

