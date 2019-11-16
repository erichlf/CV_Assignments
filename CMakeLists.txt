cmake_minimum_required(VERSION 3.10)

project(CV_Assignments)

find_package(OpenCV 3 REQUIRED)

add_executable(distortion
               distortion.cpp
               )

set_target_properties(distortion
                      PROPERTIES CXX_STANDARD 14)

target_include_directories(distortion
                           PUBLIC
                           ${CMAKE_CURRENT_SOURCE_DIR}/include
                           )

target_link_libraries(distortion
                      PRIVATE
                      ${OpenCV_LIBRARIES}
                      )

