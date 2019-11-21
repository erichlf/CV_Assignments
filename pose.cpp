#include "utilities.hpp"

int main()
{
  cv::Matx<double, 1, 4> fisheye_model(0.1, -0.2, 0.03, 0.001);
  cv::Matx<double, 1, 4> no_distortion_model(0, 0, 0, 0);

  cv::Matx33d camera_matrix(584.4, 0, 622.8,
                            0, 584.4, 538.3,
                            0, 0, 1);

  cv::Matx<double, 4, 3> object_points(0., 0., 0.,
                                       2., 0., 0.,
                                       2., 2., 0.,
                                       0., 2., 0.);

  std::vector<cv::Point2d> distorted_image_points{{752.62227857, 777.97189891},
                                                  {933.85321358, 793.21655267},
                                                  {871.88605416, 895.78261428},
                                                  {720.14648139, 886.22059601}};
  std::vector<cv::Point2d> undistorted_image_points;

  cv::Vec3d rvec;
  cv::Vec3d tvec;

  cv::fisheye::undistortPoints(distorted_image_points, undistorted_image_points, camera_matrix, fisheye_model,
                               cv::noArray(), camera_matrix);

  cv::solvePnPRansac(object_points, undistorted_image_points, camera_matrix, no_distortion_model, rvec, tvec);

  const auto reprojection_error = assignments::reprojection_error(object_points, distorted_image_points, rvec, tvec,
                                                                  camera_matrix, fisheye_model);

  cv::Affine3f objects_from_camera = cv::Affine3f(rvec, tvec).inv();

  std::cout << "Rotation:" << std::endl;
  std::cout << objects_from_camera.rvec() << std::endl;
  std::cout << "Translation:" << std::endl;
  std::cout << objects_from_camera.translation() << std::endl;
  std::cout << "Reprojection Error:" << std::endl;
  std::cout << reprojection_error << std::endl;


  cv::Mat object_points_C3(object_points.rows, 1, CV_64FC3);
  for (int i = 0; i < object_points.rows; ++i)
    object_points_C3.at<cv::Vec3d>(i) = cv::Point3d{object_points(i, 0), object_points(i, 1), object_points(i, 2)};

  cv::viz::Viz3d window = cv::viz::Viz3d("Camera Pose");
  // window.showWidget("Axis", cv::viz::WCoordinateSystem());
  window.showWidget("Camera", cv::viz::WCameraPosition(camera_matrix));
  window.showWidget("Points", cv::viz::WCloud(object_points_C3, cv::viz::Color::green()));

  window.setWidgetPose("Camera", objects_from_camera);

  while (!window.wasStopped())
    window.spinOnce(1, true);

  return EXIT_SUCCESS;
}
