cmake_minimum_required(VERSION 3.10)

project(CV_Assignments)

find_package(OpenCV 3 REQUIRED)
find_package(jsoncpp REQUIRED)

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

target_include_directories(camera_pose
                           PUBLIC
                               ${CMAKE_CURRENT_SOURCE_DIR}/include
                           )

target_link_libraries(camera_pose
                      PRIVATE
                          ${OpenCV_LIBRARIES}
                      )

add_executable(camera_to_camera
               camera_to_camera.cpp
               )

set_target_properties(camera_to_camera
                      PROPERTIES CXX_STANDARD 17)

target_include_directories(camera_to_camera
                           PUBLIC
                               ${CMAKE_CURRENT_SOURCE_DIR}/include
                           )

target_link_libraries(camera_to_camera
                      PRIVATE
                      ${OpenCV_LIBRARIES}
                      )
