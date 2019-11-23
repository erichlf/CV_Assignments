#include "utilities.hpp"

int main()
{
  cv::Matx<double, 1, 4> camera0_fisheye_model(0.1, -0.2, 0.03, 0.001);
  cv::Matx<double, 1, 4> camera1_fisheye_model(0.1, -0.2, 0.04, 0.002);
  cv::Matx<double, 1, 4> no_distortion_model(0, 0, 0, 0);

  cv::Matx33d camera0_matrix(584.4, 0, 622.8,
                             0, 584.4, 538.3,
                             0, 0, 1);
  cv::Matx33d camera1_matrix(600.4, 0, 650.8,
                             0, 600.4, 540.3,
                             0, 0, 1);

  cv::Matx<double, 4, 3> object_points(0., 0., 0.,
                                       2., 0., 0.,
                                       2., 2., 0.,
                                       0., 2., 0.);

  std::vector<cv::Point2d> camera0_projections{{752.63479088, 777.99499855},
                                               {934.18455418, 793.48809528},
                                               {872.34108972, 896.43567092},
                                               {720.19869365, 886.40720492}};
  std::vector<cv::Point2d> camera1_projections{{1034.66797488, 754.55189296},
                                               {1119.53349968, 711.67387679},
                                               {1122.39059137, 838.73588114},
                                               {1041.58631452, 903.12181786}};
  std::vector<cv::Point2d> camera0_undistorted_image_points;
  std::vector<cv::Point2d> camera1_undistorted_image_points;

  cv::Vec3d rvec0;
  cv::Vec3d tvec0;
  cv::Vec3d rvec1;
  cv::Vec3d tvec1;

  cv::fisheye::undistortPoints(camera0_projections, camera0_undistorted_image_points, camera0_matrix,
                               camera0_fisheye_model, cv::noArray(), camera0_matrix);
  cv::fisheye::undistortPoints(camera1_projections, camera1_undistorted_image_points, camera1_matrix,
                               camera1_fisheye_model, cv::noArray(), camera1_matrix);

  cv::solvePnPRansac(object_points, camera0_undistorted_image_points, camera0_matrix, no_distortion_model, rvec0,
                     tvec0);
  cv::solvePnPRansac(object_points, camera1_undistorted_image_points, camera1_matrix, no_distortion_model, rvec1,
                     tvec1);

  const auto reprojection_error0 = assignments::reprojection_error(object_points, camera0_projections, rvec0, tvec0,
                                                                   camera0_matrix, camera0_fisheye_model);
  const auto reprojection_error1 = assignments::reprojection_error(object_points, camera1_projections, rvec1, tvec1,
                                                                   camera1_matrix, camera1_fisheye_model);

  const auto camera0_from_objects = cv::Affine3f(rvec0, tvec0);
  const auto objects_from_camera0 = camera0_from_objects.inv();
  const auto objects_from_camera1 = cv::Affine3f(rvec1, tvec1).inv();

  const auto camera0_from_camera1 = camera0_from_objects * objects_from_camera1;

  std::cout << "Rotation:" << std::endl;
  std::cout << camera0_from_camera1.rvec() << std::endl;
  std::cout << "Translation:" << std::endl;
  std::cout << camera0_from_camera1.translation() << std::endl;
  std::cout << "Reprojection Errors:" << std::endl;
  std::cout << reprojection_error0 << std::endl;
  std::cout << reprojection_error1 << std::endl;


  cv::viz::Viz3d window = cv::viz::Viz3d("Camera Poses");
  window.showWidget("Axis0", cv::viz::WCoordinateSystem());
  window.showWidget("Camera0", cv::viz::WCameraPosition(camera0_matrix, 1, cv::viz::Color::blue()));
  window.showWidget("Axis1", cv::viz::WCoordinateSystem());
  window.showWidget("Camera1", cv::viz::WCameraPosition(camera1_matrix, 1, cv::viz::Color::green()));

  cv::Mat object_points_C3(object_points.rows, 1, CV_64FC3);
  for (int i = 0; i < object_points.rows; ++i)
  {
    const auto object_point = cv::Point3d{object_points(i, 0), object_points(i, 1), object_points(i, 2)};
    object_points_C3.at<cv::Vec3d>(i) = object_point;

    window.showWidget("C0 t0 L" + std::to_string(i), cv::viz::WLine({objects_from_camera0.translation()(0),
                                                                     objects_from_camera0.translation()(1),
                                                                     objects_from_camera0.translation()(2)},
                                                                    object_point));

    window.showWidget("C1 to L" + std::to_string(i), cv::viz::WLine({objects_from_camera1.translation()(0),
                                                                     objects_from_camera1.translation()(1),
                                                                     objects_from_camera1.translation()(2)},
                                                                    object_point));

    window.showWidget("P" + std::to_string(i), cv::viz::WText3D("P" + std::to_string(i), object_point, 0.25));
  }

  window.showWidget("Points", cv::viz::WCloud(object_points_C3, cv::viz::Color::red()));

  window.setWidgetPose("Camera0", objects_from_camera0);
  window.setWidgetPose("Axis0", objects_from_camera0);
  window.setWidgetPose("Camera1", objects_from_camera1);
  window.setWidgetPose("Axis1", objects_from_camera1);

  window.setWindowSize({400, 400});
  window.resetCamera();  // makes image proportional to the current window size

  while (!window.wasStopped())
    window.spinOnce(1, true);

  return EXIT_SUCCESS;
}
