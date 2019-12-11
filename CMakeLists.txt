cmake_minimum_required(VERSION 3.0.0)
project(kinect2grabber VERSION 0.1.0)

set(CMAKE_CXX_FLAGS "-std=c++11")

# Set cmake prefix path to enable cmake to find freenect2
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} $ENV{HOME}/drives/freenect2/lib/cmake/freenect2)
set(INCLUDE           "${CMAKE_SOURCE_DIR}/include")

# 寻找库
find_package(freenect2 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("/usr/include/libusb-1.0/")
include_directories( ${freenect2_INCLUDE_DIRS} )
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${INCLUDE})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

#生成可执行文件
add_executable(kinect2grabber test.cpp)

# 链接OpenCV库. freenect2, PCL
target_link_libraries(kinect2grabber ${freenect2_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
