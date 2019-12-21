cmake_minimum_required(VERSION 3.10)

project(CV_Assignments)

find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 3.3.4 REQUIRED)
find_package(Ceres REQUIRED)
find_package(jsoncpp REQUIRED)

# these are used by utilities so just include them everywhere
add_library(cv_assignments INTERFACE)

target_include_directories(cv_assignments
                           INTERFACE
                                include
                                ${EIGEN3_INCLUDE_DIRS}
                                ${CERES_INCLUDE_DIRS}
                          )

target_link_libraries(cv_assignments
                      INTERFACE
                            ${OpenCV_LIBRARIES}
                            ${EIGEN3_LIBRARIES}
                            ${CERES_LIBRARIES}
                            jsoncpp_lib
                     )

add_executable(distortion
               distortion.cpp
               )

set_target_properties(distortion
                      PROPERTIES CXX_STANDARD 17)

target_link_libraries(distortion
                      PRIVATE
                          cv_assignments
                      )

add_executable(camera_pose
               camera_pose.cpp
               )

set_target_properties(camera_pose
                      PROPERTIES CXX_STANDARD 17)

target_link_libraries(camera_pose
                      PRIVATE
                          cv_assignments
                      )

add_executable(camera_to_camera
               camera_to_camera.cpp
               )

set_target_properties(camera_to_camera
                      PROPERTIES CXX_STANDARD 17)

target_link_libraries(camera_to_camera
                      PRIVATE
                          cv_assignments
                      )

add_executable(camera_pose_ransac
               camera_pose_ransac.cpp
               )

set_target_properties(camera_pose_ransac
                      PROPERTIES CXX_STANDARD 17)

target_link_libraries(camera_pose_ransac
                      PRIVATE
                          cv_assignments
                      )
add_executable(ba_extrinsics
               ba_extrinsics.cpp
               )

set_target_properties(ba_extrinsics
                      PROPERTIES CXX_STANDARD 17)

target_link_libraries(ba_extrinsics
                      PRIVATE
                          cv_assignments
                      )
