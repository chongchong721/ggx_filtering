cmake_minimum_required(VERSION 3.10)
project(TEST VERSION 1.0)

set(CMAKE_PREFIX_PATH "/home/yuan/software/opencv_source/build" ${CMAKE_PREFIX_PATH})

find_package( OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test test.cpp
        GGX.cpp
        include/GGX.h
        include/util.h)
target_include_directories(test PUBLIC include/)
target_link_libraries(test ${OpenCV_LIBS})


add_executable(opencv_test opencv_test.cpp)
target_link_libraries( opencv_test ${OpenCV_LIBS} )