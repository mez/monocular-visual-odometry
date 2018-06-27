cmake_minimum_required(VERSION 3.1)
project(mono_vo)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 3 REQUIRED)

add_executable(mono_vo main.cpp)

#message(FATAL_ERROR ${OpenCV_LIBS})
target_link_libraries( mono_vo ${OpenCV_LIBS} )
