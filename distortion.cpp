#include "utilities.hpp"

int main()
{
  cv::Mat_<double> fisheye_model(1, 4);
  fisheye_model << 0.1, -0.2, 0.03, 0.001;

  cv::Mat_<double> plumbbob_model(1, 5);
  plumbbob_model << 0.2, 0.1, 0.001, 0.002, -0.003;

  cv::Mat_<double> rational_poly_model(1, 8);
  rational_poly_model << 0.4, -0.3, 0.001, 0.004, 0.1, 0.2, -0.12, 0.3;

  cv::Mat_<double> camera_matrix(3, 3);
  camera_matrix << 584.4, 0, 622.8,
                   0, 584.4, 538.3,
                   0, 0, 1;

  cv::Mat_<double> world_points(3, 3);
  world_points << 5, 10, 15,
                  0, 0, 10,
                  1.23, 3.4, 5.67;
  /*
  std::vector<cv::Point3d> world_points_vector{
    {world_points.at<double>(0, 0), world_points.at<double>(0, 1), world_points.at<double>(0, 2)},
    {world_points.at<double>(1, 0), world_points.at<double>(1, 1), world_points.at<double>(1, 2)},
    {world_points.at<double>(2, 0), world_points.at<double>(2, 1), world_points.at<double>(2, 2)}};
    */

  cv::Mat_<double> rvec(1, 3);
  rvec << 0, 0, 0;

  cv::Mat_<double> tvec(1, 3);
  tvec << 0, 0, 0;

  std::vector<cv::Point2d> plumbbob_image;
  assignments::project_points(world_points, rvec, tvec, camera_matrix, plumbbob_model, plumbbob_image);
  /*
  std::vector<cv::Point2d> cv_plumbbob_image;
  cv::projectPoints(world_points, rvec, tvec, camera_matrix, plumbbob_model, cv_plumbbob_image);
  */

  std::cout << "Plumb Bob:" << std::endl;
  for (int i = 0; i < plumbbob_image.size(); ++i)
  {
    std::cout << "[" << plumbbob_image[i].x << ", " << plumbbob_image[i].y << "]" << std::endl;
    // std::cout << "[" << cv_plumbbob_image[i].x << ", " << cv_plumbbob_image[i].y << "]" << std::endl;
  }

  std::vector<cv::Point2d> rational_poly_image;
  assignments::project_points(world_points, rvec, tvec, camera_matrix, rational_poly_model, rational_poly_image);
  /*
  std::vector<cv::Point2d> cv_rational_poly_image;
  cv::projectPoints(world_points, rvec, tvec, camera_matrix, rational_poly_model, cv_rational_poly_image);
  */

  std::cout << "Rational Polynomial:" << std::endl;
  for (int i = 0; i < rational_poly_image.size(); ++i)
  {
    std::cout << "[" << rational_poly_image[i].x << ", " << rational_poly_image[i].y << "]" << std::endl;
    // std::cout << "[" << cv_rational_poly_image[i].x << ", " << cv_rational_poly_image[i].y << "]" << std::endl;
  }

  std::vector<cv::Point2d> fisheye_image;
  assignments::project_points(world_points, rvec, tvec, camera_matrix, fisheye_model, fisheye_image);
  /*
  std::vector<cv::Point2d> cv_fisheye_image;
  cv::fisheye::projectPoints(world_points_vector, cv_fisheye_image, rvec, tvec, camera_matrix, fisheye_model);
   */

  std::cout << "Fisheye:" << std::endl;
  for (int i = 0; i < fisheye_image.size(); ++i)
  {
    std::cout << "[" << fisheye_image[i].x << ", " << fisheye_image[i].y << "]" << std::endl;
    // std::cout << "[" << cv_fisheye_image[i].x << ", " << cv_fisheye_image[i].y << "]" << std::endl;
  }

  return EXIT_SUCCESS;
}
