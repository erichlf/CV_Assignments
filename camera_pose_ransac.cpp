#include "utilities.hpp"

void plot_points(cv::viz::Viz3d& window, const std::vector<cv::Point3d>& object_points,
                 const std::vector<cv::Point2d>& image_points, cv::Affine3d objects_from_camera,
                 std::string str, cv::viz::Color color)
{
  for (int i = 0; i < object_points.size(); ++i)
  {
    std::cout << object_points[i] << " -> " << image_points[i]<< std::endl;

    window.showWidget("L" + str[0] + std::to_string(i), cv::viz::WLine({objects_from_camera.translation()(0),
                                                                        objects_from_camera.translation()(1),
                                                                        objects_from_camera.translation()(2)},
                                                                       object_points[i], color));

    window.showWidget(str[0] + std::to_string(i), cv::viz::WText3D(str[0] + std::to_string(i), object_points[i], 0.25));
  }

  window.showWidget(str, cv::viz::WCloud(object_points, color));
}

int main()
{
  cv::Matx<double, 1, 4> fisheye_model(0.1, -0.2, 0.03, 0.001);

  cv::Matx33d camera_matrix(584.4, 0, 622.8,
                            0, 584.4, 538.3,
                            0, 0, 1);

  std::vector<cv::Point3d> object_points{{1.1, 2.2, 0.},
                                         {0., 2.2, 0.},
                                         {3.3, 2.2, 0.},
                                         {6.7, 8.5, 0.},
                                         {-1.2, 4.1, 0.},
                                         {2.4, 0.5, 0.},
                                         {0.1, -0., 0.},
                                         {4.1, -2.3, 0.}};

  std::vector<cv::Point2d> image_points{{801.88308337, 902.17992978},
                                        {717.37983112, 894.41915636},
                                        {948.24770469, 898.13947261},
                                        {930.6333786, 1002.97069515},
                                        {615.71308703, 943.09995767},
                                        {947.00931604, 823.38577582},
                                        {610.223, 1090.54},
                                        {610.346, 1093.88}};

  int ransac_iters;
  cv::Vec3f rvec, tvec;
  std::vector<int> inlier_indices;
  std::vector<int> outlier_indices;
  std::tie(rvec, tvec, inlier_indices,
           outlier_indices, ransac_iters) = assignments::fisheye_solvePnPRansac(object_points, image_points,
                                                                                camera_matrix, fisheye_model);

  std::vector<cv::Point3d> object_inliers;
  std::vector<cv::Point2d> image_inliers;
  std::vector<cv::Point3d> object_outliers;
  std::vector<cv::Point2d> image_outliers;

  std::tie(object_inliers, image_inliers) = assignments::get_liers(object_points, image_points, inlier_indices);
  std::tie(object_outliers, image_outliers) = assignments::get_liers(object_points, image_points, outlier_indices);
  const auto reprojection_error = assignments::reprojection_error(object_inliers, image_inliers, rvec, tvec,
                                                                  camera_matrix, fisheye_model);

  const auto objects_from_camera = cv::Affine3f(rvec, tvec).inv();

  std::string plural = (ransac_iters != 1) ? "s." : ".";
  std::cout << "RANSAC converged in " << ransac_iters << " iteration" << plural << std::endl;
  assignments::print_result(objects_from_camera, reprojection_error);

  cv::viz::Viz3d window = cv::viz::Viz3d("Camera Pose");
  window.showWidget("Axis", cv::viz::WCoordinateSystem());
  window.showWidget("Camera", cv::viz::WCameraPosition(camera_matrix));

  std::cout << "Inliers:" << std::endl;
  plot_points(window, object_inliers, image_inliers, objects_from_camera, "Inlier Points", cv::viz::Color::green());

  std::cout << "Outliers:" << std::endl;
  plot_points(window, object_outliers, image_outliers, objects_from_camera, "Outlier Points", cv::viz::Color::red());

  window.setWidgetPose("Camera", objects_from_camera);
  window.setWidgetPose("Axis", objects_from_camera);

  window.setWindowSize({400, 400});
  window.resetCamera();  // makes image proportional to the current window size

  while (!window.wasStopped())
    window.spinOnce(1, true);

  return EXIT_SUCCESS;
}
