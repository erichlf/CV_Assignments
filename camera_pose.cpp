#include "utilities.hpp"

int main()
{
  cv::Matx<double, 1, 4> fisheye_model(0.1, -0.2, 0.03, 0.001);

  cv::Matx33d camera_matrix(584.4, 0, 622.8,
                            0, 584.4, 538.3,
                            0, 0, 1);

  std::vector<cv::Point3d> object_points{{0., 0., 0.},
                                         {2., 0., 0.},
                                         {2., 2., 0.},
                                         {0., 2., 0.}};

  std::vector<cv::Point2d> image_points{{752.62227857, 777.97189891},
                                        {933.85321358, 793.21655267},
                                        {871.88605416, 895.78261428},
                                        {720.14648139, 886.22059601}};

  const auto& [rvec, tvec] = assignments::fisheye_solvePnP(object_points, image_points, camera_matrix, fisheye_model);

  const auto reprojection_error = assignments::reprojection_error(object_points, image_points, rvec, tvec,
                                                                  camera_matrix, fisheye_model);

  const auto objects_from_camera = cv::Affine3f(rvec, tvec).inv();

  assignments::print_result(objects_from_camera, reprojection_error);

  cv::viz::Viz3d window = cv::viz::Viz3d("Camera Pose");
  window.showWidget("Axis", cv::viz::WCoordinateSystem());
  window.showWidget("Camera", cv::viz::WCameraPosition(camera_matrix));

  cv::Mat object_points_C3(object_points.size(), 1, CV_64FC3);
  for (int i = 0; i < object_points.size(); ++i)
  {
    const auto object_point = object_points[i];
    object_points_C3.at<cv::Vec3d>(i) = object_point;

    window.showWidget("L" + std::to_string(i), cv::viz::WLine({objects_from_camera.translation()(0),
                                                               objects_from_camera.translation()(1),
                                                               objects_from_camera.translation()(2)},
                                                              object_point));

    window.showWidget("P" + std::to_string(i), cv::viz::WText3D("P" + std::to_string(i), object_point, 0.25));
  }

  window.showWidget("Points", cv::viz::WCloud(object_points_C3, cv::viz::Color::red()));

  window.setWidgetPose("Camera", objects_from_camera);
  window.setWidgetPose("Axis", objects_from_camera);

  window.setWindowSize({400, 400});
  window.resetCamera();  // makes image proportional to the current window size

  while (!window.wasStopped())
    window.spinOnce(1, true);

  return EXIT_SUCCESS;
}
