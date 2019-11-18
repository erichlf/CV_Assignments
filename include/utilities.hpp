#include <opencv2/opencv.hpp>

#include <vector>

namespace assignments
{

/*
 * \brief projects the distorted and z-normalized world point into image points
 * \param distorted_point distorted and z-normalized world point
 * \param K camera matrix
 */
cv::Point2d get_image_point_(cv::Point2d distorted_point, const cv::Matx<double, 3, 3> K)
{
  const auto fx = K(0, 0);
  const auto fy = K(1, 1);
  const auto cx = K(0, 2);
  const auto cy = K(1, 2);

  const auto u = distorted_point.x * fx + cx;
  const auto v = distorted_point.y * fy + cy;

  return {u, v};
}

/*
 * \brief projects 3D world point into image points with fisheye distortion
 * \param world_points  3D world point data
 * \param rvec  rotation vector
 * \param tvec  translation vector
 * \param K camera matrix
 * \param dist_coefficients distortion coefficients k1, k2, k3, k4
 * \param image_points  points in image after distortion
 */
void project_points(const cv::InputArray& world_points_, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                    const cv::Matx<double, 3, 3>& K, const cv::Matx<double, 1, 4>& dist_coeffs,
                    std::vector<cv::Point2d>& image_points)
{
  const auto world_points = world_points_.getMat();

  const auto k1 = dist_coeffs(0, 0);
  const auto k2 = dist_coeffs(0, 1);
  const auto k3 = dist_coeffs(0, 2);
  const auto k4 = dist_coeffs(0, 3);

  for (int i = 0; i < world_points.rows; ++i)
  {
    const auto x = world_points.at<double>(i, 0);
    const auto y = world_points.at<double>(i, 1);
    const auto z = world_points.at<double>(i, 2);

    const auto xp = x / z;
    const auto yp = y / z;

    const auto r = sqrt(xp * xp + yp * yp);
    const auto theta = atan(r);

    const double theta_r = r != 0 ?
                           theta * (1 + k1 * theta * theta + k2 * pow(theta, 4) + k3 * pow(theta, 6) + k4 * pow(theta, 8)) / r
                           : 0;

    const double xpp = theta_r * xp;
    const double ypp = theta_r * yp;

    const auto image_point = get_image_point_({xpp, ypp}, K);
    image_points.push_back(image_point);
  }
}

/*
 * \brief projects 3D world point into image points with plumb bob distortion
 * \param world_points  3D world point data
 * \param rvec  rotation vector
 * \param tvec  translation vector
 * \param K camera matrix
 * \param dist_coefficients distortion coefficients k1, k2, p1, p2, k3, k4
 * \param image_points  points in image after distortion
 */
void project_points(const cv::InputArray& world_points_, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                    const cv::Matx<double, 3, 3>& K, const cv::Matx<double , 1, 5>& dist_coeffs,
                    std::vector<cv::Point2d>& image_points)
{
  const auto world_points = world_points_.getMat();
  const auto k1 = dist_coeffs(0, 0);
  const auto k2 = dist_coeffs(0, 1);
  const auto p1 = dist_coeffs(0, 2);
  const auto p2 = dist_coeffs(0, 3);
  const auto k3 = dist_coeffs(0, 4);

  for (int i = 0; i < world_points.rows; ++i)
  {
    const auto x = world_points.at<double>(i, 0);
    const auto y = world_points.at<double>(i, 1);
    const auto z = world_points.at<double>(i, 2);

    const double xp = x / z;
    const double yp = y / z;

    const auto r = sqrt(xp * xp + yp * yp);

    const auto theta = (1. + k1 * r * r + k2 * pow(r, 4) + k3 * pow(r, 6));

    const double dx1 = 2. * p1 * xp * yp + p2 * (r * r + 2. * xp * xp);
    const double dx2 = p1 * (r * r + 2. * yp * yp) + 2. * p2 * xp * yp;


    const double xpp = theta * xp + dx1;
    const double ypp = theta * yp + dx2;

    const auto image_point = get_image_point_({xpp, ypp}, K);
    image_points.push_back(image_point);
  }
}

/*
 * \brief projects 3D world point into image points with rational polynomial distortion
 * \param world_points  3D world point data
 * \param rvec  rotation vector
 * \param tvec  translation vector
 * \param K camera matrix
 * \param dist_coefficients distortion coefficients k1, k2, p1, p2, k3, k4, k5, k6
 * \param image_points  points in image after distortion
 */
void project_points(const cv::InputArray& world_points_, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                    const cv::Matx<double, 3, 3>& K, const cv::Matx<double , 1, 8>& dist_coeffs,
                    std::vector<cv::Point2d>& image_points)
{
  const auto world_points = world_points_.getMat();

  const auto k1 = dist_coeffs(0, 0);
  const auto k2 = dist_coeffs(0, 1);
  const auto p1 = dist_coeffs(0, 2);
  const auto p2 = dist_coeffs(0, 3);
  const auto k3 = dist_coeffs(0, 4);
  const auto k4 = dist_coeffs(0, 5);
  const auto k5 = dist_coeffs(0, 6);
  const auto k6 = dist_coeffs(0, 7);

  for (int i = 0; i < world_points.rows; ++i)
  {
    const auto x = world_points.at<double>(i, 0);
    const auto y = world_points.at<double>(i, 1);
    const auto z = world_points.at<double>(i, 2);

    const double xp = x / z;
    const double yp = y / z;

    double r = sqrt(xp * xp + yp * yp);

    const double theta = (1. + k1 * r * r + k2 * pow(r, 4) + k3 * pow(r, 6)) /
                         (1. + k4 * r * r + k5 * pow(r, 4) + k6 * pow(r, 6));

    const double dx1 = 2. * p1 * xp * yp + p2 * (r * r + 2. * xp * xp);
    const double dx2 = p1 * (r * r + 2. * yp * yp) + 2. * p2 * xp * yp;


    double xpp = theta * xp + dx1;
    double ypp = theta * yp + dx2;

    const auto image_point = get_image_point_({xpp, ypp}, K);
    image_points.push_back(image_point);
  }
}

};  // namespace assignements
