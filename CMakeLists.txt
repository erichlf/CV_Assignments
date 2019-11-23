cmake_minimum_required(VERSION 3.10)

project(CV_Assignments)

find_package(OpenCV 3 REQUIRED)

add_executable(distortion
               distortion.cpp
               )

set_target_properties(distortion
                      PROPERTIES CXX_STANDARD 17)

target_include_directories(distortion
                           PUBLIC
                               ${CMAKE_CURRENT_SOURCE_DIR}/include
                           )

target_link_libraries(distortion
                      PRIVATE
                          ${OpenCV_LIBRARIES}
                      )

add_executable(camera_pose
               camera_pose.cpp
               )

set_target_properties(camera_pose
                      PROPERTIES CXX_STANDARD 17)

target_include_directories(pose
                           PUBLIC
                               ${CMAKE_CURRENT_SOURCE_DIR}/include
                           )

target_link_libraries(pose
                      PRIVATE
                          ${OpenCV_LIBRARIES}
                      )
