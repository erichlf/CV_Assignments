cmake_minimum_required(VERSION 3.10)

project(CV_Assignments)

find_package(OpenCV 3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(jsoncpp REQUIRED)

# these are used by utilities so just include them everywhere
add_library(cv_assignments
            SHARED
                Calibration.cpp
           )

target_include_directories(cv_assignments
                           PUBLIC
                                include
                                ${OpenCV_INCLUDE_DIRS}
                                ${CERES_INCLUDE_DIRS}
                          )

target_link_libraries(cv_assignments
                      PUBLIC
                            ${OpenCV_LIBRARIES}
                            ${CERES_LIBRARIES}
                            jsoncpp_lib
                     )

set_target_properties(cv_assignments
                      PROPERTIES CXX_STANDARD 17)

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

add_executable(calibration
               calibration.cpp
              )

set_target_properties(calibration
                      PROPERTIES CXX_STANDARD 17)

target_link_libraries(calibration
                      PRIVATE
                          cv_assignments
                     )
