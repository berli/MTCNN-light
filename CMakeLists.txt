cmake_minimum_required(VERSION 2.8)
project(main)

set(CMAKE_CXX_STANDARD 11)
set(QMAKE_CXXFLAGS "-std=c++11")
set(QMAKE_CXXFLAGS "-O3")

find_package( OpenCV REQUIRED)

include_directories(src/)
set(MAIN_SRCS  pikaqiu.cpp )
AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
add_executable( main ${DIR_SRCS} ${MAIN_SRCS})
target_link_libraries( main ${OpenCV_LIBS} )
target_link_libraries( main openblas)

set(CMAKE_BUILD_TYPE Release)
