#include <opencv2/opencv.hpp>

#include <vector>
#include <tuple>
#include <algorithm>
#include <limits>
#include <numeric>

namespace assignments
{

/*
 * \brief convert a rotation vector and translation vector to a transformation matrix
 */
inline
cv::Affine3d get_transform_(cv::Vec3d rvec, cv::Vec3d tvec)
{
  return cv::Affine3d(rvec, tvec);
}

/*
 * \brief transform a homogeneous point
 * \param transform transformation
 * \param point point to be transformed
 */
inline
std::tuple<double, double, double> transform_point(const cv::Affine3d& transform, cv::Vec3d point)
{
  const cv::Vec3d transformed_point = transform * point;

  return {transformed_point(0), transformed_point(1), transformed_point(2)};
}

/*
 * \brief projects the distorted and z-normalized world point into image points
 * \param distorted_point distorted and z-normalized world point
 * \param K camera matrix
 */
inline
cv::Point2d get_image_point_(cv::Point2d distorted_point, const cv::Matx33d K)
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
void project_points(const std::vector<cv::Point3d>& world_points, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                    const cv::Matx33d& K, const cv::Matx<double, 1, 4>& dist_coeffs,
                    std::vector<cv::Point2d>& image_points)
{
  const auto transform = get_transform_(rvec, tvec);

  const auto k1 = dist_coeffs(0, 0);
  const auto k2 = dist_coeffs(0, 1);
  const auto k3 = dist_coeffs(0, 2);
  const auto k4 = dist_coeffs(0, 3);

  for (int i = 0; i < world_points.size(); ++i)
  {
    const auto& [x, y, z] = transform_point(transform,
                                            {world_points[i].x,
                                             world_points[i].y,
                                             world_points[i].z});
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
void project_points(const std::vector<cv::Point3d>& world_points, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                    const cv::Matx33d& K, const cv::Matx<double , 1, 5>& dist_coeffs,
                    std::vector<cv::Point2d>& image_points)
{

  const auto transform = get_transform_(rvec, tvec);

  const auto k1 = dist_coeffs(0, 0);
  const auto k2 = dist_coeffs(0, 1);
  const auto p1 = dist_coeffs(0, 2);
  const auto p2 = dist_coeffs(0, 3);
  const auto k3 = dist_coeffs(0, 4);

  for (int i = 0; i < world_points.size(); ++i)
  {
    const auto& [x, y, z] = transform_point(transform,
                                            {world_points[i].x,
                                             world_points[i].y,
                                             world_points[i].z});

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
void project_points(const std::vector<cv::Point3d>& world_points, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                    const cv::Matx33d& K, const cv::Matx<double , 1, 8>& dist_coeffs,
                    std::vector<cv::Point2d>& image_points)
{
  const auto transform = get_transform_(rvec, tvec);

  const auto k1 = dist_coeffs(0, 0);
  const auto k2 = dist_coeffs(0, 1);
  const auto p1 = dist_coeffs(0, 2);
  const auto p2 = dist_coeffs(0, 3);
  const auto k3 = dist_coeffs(0, 4);
  const auto k4 = dist_coeffs(0, 5);
  const auto k5 = dist_coeffs(0, 6);
  const auto k6 = dist_coeffs(0, 7);

  for (int i = 0; i < world_points.size(); ++i)
  {
    const auto& [x, y, z] = transform_point(transform,
                                            {world_points[i].x,
                                             world_points[i].y,
                                             world_points[i].z});

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

/*
 * \brief determines the error between the given image points and the projection of the 3D object points
 * \param object_points 3D world points to project and compare
 * \param image_points  2D image points to compare to the projected world points
 * \param rvec  rotation vector
 * \param tvec  translation vector
 * \param camera_matrix camera matrix
 * \param dist_coeffs fisheye distortion coefficients
 */
double reprojection_error(const std::vector<cv::Point3d>& object_points,
                          const std::vector<cv::Point2d>& image_points,
                          const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                          const cv::Matx33d& camera_matrix, const cv::Matx<double, 1, 4>& dist_coeffs)
{
  std::vector<cv::Point2d> projected_image_points;

  // cv::fisheye::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, projected_image_points);
  project_points(object_points, rvec, tvec, camera_matrix, dist_coeffs, projected_image_points);

  double error = 0;
  for (int i = 0; i < object_points.size(); ++i)
    error += cv::norm(cv::Mat(image_points[i]), cv::Mat(projected_image_points[i]), CV_L2);

  error /= image_points.size();

  return error;
}

/*
 * \brief solvePnP for a fisheye model
 * \param object_points   the 3D portion of 3D to 2D correspondences
 * \param image_points   the 2D portion of 3D to 2D correspondences
 * \param camera_matrix   matrix containing camera intrinsics
 * \param fisheye_model   fisheye model to use in deprojection
 */
std::tuple<cv::Vec3d, cv::Vec3d> fisheye_solvePnP(const std::vector<cv::Point3d>& object_points,
                                                  const std::vector<cv::Point2d>& image_points,
                                                  const cv::Matx33d& camera_matrix,
                                                  const cv::Matx<double, 1, 4>& fisheye_model)
{
  cv::Matx<double, 1, 4> no_distortion_model(0, 0, 0, 0);

  // undistort points so that we can use cv::solvePnP
  std::vector<cv::Point2d> camera_undistorted_image_points;
  cv::fisheye::undistortPoints(image_points, camera_undistorted_image_points, camera_matrix, fisheye_model,
                               cv::noArray(), camera_matrix);

  cv::Vec3d rvec;
  cv::Vec3d tvec;
  cv::solvePnP(object_points, camera_undistorted_image_points, camera_matrix, no_distortion_model, rvec, tvec);

  return {rvec, tvec};
}

std::tuple<std::vector<cv::Point3d>, std::vector<cv::Point2d>>
get_random_subset_(const std::vector<cv::Point3d>& object_points, const std::vector<cv::Point2d> image_points,
            const int num_model_points=4)
{
  std::vector<int> indices(object_points.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::vector<cv::Point3d> object_points_subset;
  std::vector<cv::Point2d> image_points_subset;
  for (int i = 0; i < num_model_points; ++i)
  {
    object_points_subset.push_back(object_points[indices[i]]);
    image_points_subset.push_back(image_points[indices[i]]);
  }

  return {object_points_subset, image_points_subset};
}

std::tuple<std::vector<int>, std::vector<int>>
get_liers_(const std::vector<cv::Point3d>& object_points, const std::vector<cv::Point2d>& image_points,
           const cv::Vec3d& rvec, const cv::Vec3d& tvec,
           const cv::Matx33d& camera_matrix, const cv::Matx<double , 1, 4>& fisheye_model,
           const double threshold)
{
  std::vector<int> inliers;
  std::vector<int> outliers;
  for(int i = 0; i < object_points.size(); i++)
  {
    const auto error = reprojection_error({object_points[i]}, {image_points[i]}, rvec, tvec, camera_matrix,
                                          fisheye_model);
    if (error < threshold)
      inliers.push_back(i);
    else
      outliers.push_back(i);
  }

  return {inliers, outliers};
}

int ransac_update_num_iters_(const double confidence, const double outlier_ratio, const int max_iters)
{
  const int num_model_points = 4;
  // avoid inf's & nan's
  const auto double_min = std::numeric_limits<double>::min();
  double num = std::max(1. - confidence, double_min);
  double denom = 1. - std::pow(1. - outlier_ratio, num_model_points);
  if (denom < double_min)
    return 0;

  num = std::log(num);
  denom = std::log(denom);

  return (denom >= 0 || -num >= max_iters * (-denom)) ? max_iters : std::round(num / denom);
}

std::tuple<cv::Vec3d, cv::Vec3d, std::vector<int>, std::vector<int>>
fisheye_solvePnPRansac(const std::vector<cv::Point3d>& object_points,
                       const std::vector<cv::Point2d> image_points,
                       const cv::Matx33d& camera_matrix, const cv::Matx<double, 1, 4>& fisheye_model,
                       const double threshold=1, const double confidence=0.99, const int max_iters=100,
                       const int num_model_points=4)
{
  std::vector<int> best_inlier_index;
  std::vector<int> best_outlier_index;
  cv::Vec3d best_rvec, best_tvec;
  int num_iters = max_iters;

  int iter;
  for(iter = 0; iter < num_iters; iter++)
  {
    auto const& [object_subset, image_subset] = get_random_subset_(object_points, image_points, num_model_points);

    const auto& [rvec, tvec] = fisheye_solvePnP(object_subset, image_subset, camera_matrix, fisheye_model);

    std::vector<int> inlier_index;
    std::vector<int> outlier_index;
    std::tie(inlier_index, outlier_index) = get_liers_(object_points, image_points, rvec, tvec, camera_matrix,
                                                       fisheye_model, threshold);

    if(inlier_index.size() > best_inlier_index.size())
    {
      double outlier_ratio = (double)(inlier_index.size() - best_inlier_index.size()) / inlier_index.size();
      num_iters = ransac_update_num_iters_(confidence, outlier_ratio, num_iters);

      best_inlier_index = inlier_index;
      best_outlier_index = outlier_index;

      best_rvec = rvec;
      best_tvec = tvec;
    }
  }

  std::string plural = (iter != 1) ? "s." : ".";
  std::cout << "RANSAC converged in " << iter << " iteration" << plural << std::endl;

  return {best_rvec, best_tvec, best_inlier_index, best_outlier_index};
}

};  // namespace assignments
