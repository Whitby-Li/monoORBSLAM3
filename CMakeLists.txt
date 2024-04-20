cmake_minimum_required(VERSION 3.22)
project(mono_orb_slam3)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED)
find_package(g2o REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}/test
        ${PROJECT_SOURCE_DIR}/modules
        ${PROJECT_SOURCE_DIR}/thirdParty/DBoW2
        /usr/include/eigen3
        ${OpenCV_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS}
)

link_libraries(
        ${PROJECT_SOURCE_DIR}/thirdParty/DBoW2/lib/libDBoW2.so
        ${OpenCV_LIBRARIES}
        g2o_core g2o_types_sba
        ${Pangolin_LIBRARIES}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(mono_orb_slam3 SHARED
        modules/Utils/Converter.cpp
        modules/Utils/LieAlgeBra.cpp
        modules/Sensor/Camera.cpp
        modules/Sensor/Pinhole.cpp
        modules/Sensor/Fisheye.cpp
        modules/Sensor/Imu.cpp
        modules/ORB/ORBExtractor.cpp
        modules/ORB/ORBMatcher.cpp
        modules/ORB/ORBVocabulary.cpp
        modules/BasicObject/Pose.cpp
        modules/BasicObject/Frame.cpp
        modules/BasicObject/KeyFrame.cpp
        modules/BasicObject/MapPoint.cpp
        modules/BasicObject/Map.cpp
        modules/Frontend/TwoViewReconstruction.cpp
        modules/Frontend/Tracking.cpp
        modules/Frontend/LocalMapping.cpp
        modules/Backend/G2oTypes.cpp
        modules/Backend/Optimize.cpp
        modules/Log/Logger.cpp
        modules/View/FrameDrawer.cpp
        modules/View/MapDrawer.cpp
        modules/View/Viewer.cpp
        modules/System.cpp
        )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test/bin)
link_libraries(mono_orb_slam3)

# ORB
add_executable(extractor_test test/ORB/extractorTest.cpp)
add_executable(initialize_search_test test/ORB/initializeSearchTest.cpp)

# Camera
add_executable(img_bound_test test/Camera/imageBoundTest.cpp)

# Imu
add_executable(imu_test test/Imu/imuTest.cpp)

# Frontend
add_executable(initialize_test test/Frontend/initializeTest.cpp)
add_executable(tracking_test test/Frontend/trackingTest.cpp)

# SLAM Demo
add_executable(euroc_demo test/eurocDemo.cpp)
add_executable(kitti_demo test/kittiDemo.cpp)
add_executable(phone_demo test/phoneDemo.cpp)
add_executable(ntu_demo test/ntuDemo.cpp)
add_executable(rect_demo test/rectDemo.cpp)
add_executable(demo test/demo.cpp)
