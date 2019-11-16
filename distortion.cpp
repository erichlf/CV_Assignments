#include "utilities.hpp"

int main()
{
  cv::Mat_<double> fisheye_model(1, 4);
  fisheye_model << 0.1, -0.2, 0.03, 0.001;

  cv::Mat_<double> plumbbob_model(1, 6);
  plumbbob_model << 0.2, 0.1, 0.001, 0.002, -0.003;

  cv::Mat_<double> rational_poly_model(1, 8);
  rational_poly_model << 0.4, -0.3, 0.001, 0.004, 0.1, 0.2, -0.12, 0.3;

  cv::Mat_<double> camera_matrix(3, 3);
  camera_matrix << 584, 0, 622.8,
                   0, 584.4, 538.3,
                   0, 0, 1;

  cv::Mat_<double> world_points(3, 3);
  world_points << 5, 10, 15,
                  0, 0, 10,
                  1.23, 3.4, 5.67;

  cv::Mat_<double> rvec(1, 3);
  rvec << 0, 0, 0;

  cv::Mat_<double> tvec(1, 3);
  tvec << 0, 0, 0;

  std::vector<cv::Point2d> plumbbob_image;
  assignments::project_points(world_points, rvec, tvec, camera_matrix, plumbbob_model, plumbbob_image);

  std::cout << "Plumb Bob:" << std::endl;
  for (const auto& point : plumbbob_image)
    std::cout << "[" << point.x << ", " << point.y << "]" << std::endl;

  std::vector<cv::Point2d> rational_poly_image;
  assignments::project_points(world_points, rvec, tvec, camera_matrix, rational_poly_model, rational_poly_image);

  std::cout << "Rational Polynomial:" << std::endl;
  for (const auto& point : rational_poly_image)
    std::cout << "[" << point.x << ", " << point.y << "]" << std::endl;

  std::vector<cv::Point2d> fisheye_image;
  assignments::project_points(world_points, rvec, tvec, camera_matrix, fisheye_model, fisheye_image);

  std::cout << "Fisheye:" << std::endl;
  for (const auto& point : fisheye_image)
    std::cout << "[" << point.x << ", " << point.y << "]" << std::endl;

  return EXIT_SUCCESS;
}
