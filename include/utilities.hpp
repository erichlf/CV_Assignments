#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <jsoncpp/json/json.h>

#include <vector>
#include <tuple>
#include <algorithm>
#include <limits>
#include <numeric>
#include <memory>

#include "types.hpp"

#pragma once

namespace assignments
{

std::vector<cv::Point2d> convertToPoints(std::vector<Vector2<double>> points)
{
  std::vector<cv::Point2d> cvPoints;
  for (const auto& point : points)
    cvPoints.push_back(cv::Point2d(point[0], point[1]));

  return cvPoints;
}

std::vector<cv::Point3d> convertToPoints(std::vector<Vector3<double>> points)
{
  std::vector<cv::Point3d> cvPoints;
  for (const auto& point : points)
    cvPoints.push_back(cv::Point3d(point[0], point[1], point[2]));

  return cvPoints;
}

/*
 * \brief given a json file containing correspondences build the correspondences
 * \param json_file   string containing the location of the json file to read
 * \return {object_points, image_points}  correspondences between 3D and 2D
 */
std::tuple<std::vector<cv::Point3d>, std::vector<cv::Point2d>> load_correspondence(const std::string& json_file)
{
  std::ifstream ifs(json_file);
  Json::Value correspondences_json;
  ifs >> correspondences_json;

  std::vector<cv::Point3d> world_points;
  std::vector<cv::Point2d> image_points;
  for (const auto& correspondence : correspondences_json["correspondences"])
  {
    world_points.push_back({correspondence["world_point"][0].asDouble(),
                            correspondence["world_point"][1].asDouble(),
                            correspondence["world_point"][2].asDouble()});
    image_points.push_back({correspondence["image_point"][0].asDouble(),
                            correspondence["image_point"][1].asDouble()});
  }

  return {world_points, image_points};
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const Transform<T>& transform)
{
  const auto translation = transform.t();
  const auto rotation = transform.R();

  out << "Angle Axis (" << rotation[Coords::x] << ","
                        << rotation[Coords::y] << ","
                        << rotation[Coords::z] << ")" << std::endl;
  out << "Translation (" << translation[Coords::x] << ","
                         << translation[Coords::y] << ","
                         << translation[Coords::z] << ")";
  return out;
}

template <typename T>
void fisheyeReprojectionError(const T* const objectPoint, const T* const imagePoint,
                              const T* const cameraMatrix, const T* const distortionCoeffs,
                              const T* const rvec, const T* const tvec,
                              T* reprojectionError)
{
  Transform<T> transform{rvec, tvec};
  Intrinsics<T> intrinsics(cameraMatrix, distortionCoeffs);

  auto point = transform.transform(objectPoint);
  auto projectedPoint = intrinsics.projectPoint(point.data());

  reprojectionError[Coords::x] = projectedPoint[Coords::x] - imagePoint[Coords::x];
  reprojectionError[Coords::y] = projectedPoint[Coords::y] - imagePoint[Coords::y];
}

namespace
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

std::tuple<std::vector<cv::Point3d>, std::vector<cv::Point2d>, std::vector<int>>
get_random_subset_(const std::vector<cv::Point3d>& object_points, const std::vector<cv::Point2d> image_points,
                   const int num_model_points = 4)
{
  std::vector<int> indices(object_points.size());
  std::iota(indices.begin(), indices.end(), 0);  // vector containing 0,...,indices.size() - 1

  std::random_shuffle(indices.begin(), indices.end());

  indices.erase(indices.begin() + num_model_points, indices.end()); // the points that were included in the subset

  std::vector<cv::Point3d> object_points_subset;
  std::vector<cv::Point2d> image_points_subset;
  for (const auto& index : indices)
  {
    object_points_subset.push_back(object_points[index]);
    image_points_subset.push_back(image_points[index]);
  }

  return {object_points_subset, image_points_subset, indices};
}
} // anonymous namespace

void print_result(const cv::Affine3d& objects_from_camera, const double reprojection_error)
{
  std::cout << "Rotation:" << std::endl;
  std::cout << objects_from_camera.rvec() << std::endl;
  std::cout << "Translation:" << std::endl;
  std::cout << objects_from_camera.translation() << std::endl;
  std::cout << "Reprojection Error:" << std::endl;
  std::cout << reprojection_error << std::endl;
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

  double cameraMatrix[4] = {camera_matrix(0, 0), camera_matrix(1, 1), camera_matrix(0, 2), camera_matrix(1, 2)};
  double distortionCoeffs[4] = {dist_coeffs(0), dist_coeffs(1), dist_coeffs(2), dist_coeffs(3)};
  double rVec[3] = {rvec[0], rvec[1], rvec[2]};
  double tVec[3] = {tvec[0], tvec[1], tvec[2]};

  double error = 0;
  for (int i = 0; i < object_points.size(); ++i)
  {
    double objectPoint[3] = {object_points[i].x,
                             object_points[i].y,
                             object_points[i].z};
    double imagePoint[2] = {image_points[i].x,
                            image_points[i].y};

    double residual[2];

    fisheyeReprojectionError(objectPoint, imagePoint, cameraMatrix, distortionCoeffs, rVec, tVec, residual);

    error += sqrt(residual[0] * residual[0] + residual[1] * residual[1]);
  }

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

  int iters = 100;
  cv::Vec3d rvec;
  cv::Vec3d tvec;
  cv::solvePnP(object_points, camera_undistorted_image_points, camera_matrix, no_distortion_model, rvec, tvec);
  // cv::solvePnPRansac(object_points, camera_undistorted_image_points, camera_matrix, no_distortion_model, rvec, tvec);

  return {rvec, tvec};
}

namespace
{
std::tuple<std::vector<int>, std::vector<int>>
get_liers_index_(const std::vector<cv::Point3d>& object_points, const std::vector<cv::Point2d>& image_points,
                 const std::vector<int>& inlier_index, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                 const cv::Matx33d& camera_matrix, const cv::Matx<double , 1, 4>& fisheye_model,
                 const double threshold)
{
  std::vector<int> inliers;
  std::vector<int> outliers;
  for(int i = 0; i < object_points.size(); i++)
  {
    bool in_model = std::find(inlier_index.begin(), inlier_index.end(), i) != inlier_index.end();
    const auto error = in_model ? 0 : reprojection_error({object_points[i]}, {image_points[i]}, rvec, tvec,
                                                         camera_matrix, fisheye_model);
    if (error < threshold)
      inliers.push_back(i);
    else
      outliers.push_back(i);
  }

  return {inliers, outliers};
}
}  // namespace anonymous

std::tuple<std::vector<cv::Point3d>, std::vector<cv::Point2d>>
get_liers(const std::vector<cv::Point3d>& object_points, const std::vector<cv::Point2d>& image_points,
          const std::vector<int>& lier_indices)
{
  std::vector<cv::Point2d> image_liers;
  std::vector<cv::Point3d> object_liers;

  for (const auto& lier_index : lier_indices)
  {
    object_liers.push_back(object_points[lier_index]);
    image_liers.push_back(image_points[lier_index]);
  }

  return {object_liers, image_liers};
}

int ransac_update_num_iters_(const double confidence, const double outlier_ratio, const int max_iters,
                            const int num_model_points=4)
{
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

std::tuple<cv::Vec3d, cv::Vec3d, std::vector<int>, std::vector<int>, int>
fisheye_solvePnPRansac(const std::vector<cv::Point3d>& object_points,
                       const std::vector<cv::Point2d> image_points,
                       const cv::Matx33d& camera_matrix, const cv::Matx<double, 1, 4>& fisheye_model,
                       const double threshold=8, const double confidence=0.99, const int max_iters=100,
                       const int num_model_points=4)
{
  std::vector<int> best_inlier_index;
  std::vector<int> best_outlier_index;
  cv::Vec3d best_rvec, best_tvec;
  int num_iters = ransac_update_num_iters_(confidence, 0.45, max_iters, num_model_points);
  num_iters = std::max(num_iters, 3);

  int iter;
  for(iter = 0; iter < num_iters; iter++)
  {
    auto const& [object_subset, image_subset, subset_indices] = get_random_subset_(object_points, image_points,
                                                                                   num_model_points);

    const auto& [rvec, tvec] = fisheye_solvePnP(object_subset, image_subset, camera_matrix, fisheye_model);
    // const auto& [rvec, tvec] = fisheye_solvePnP(object_points, image_points, camera_matrix, fisheye_model);

    std::vector<int> inlier_index;
    std::vector<int> outlier_index;
    std::tie(inlier_index, outlier_index) = get_liers_index_(object_points, image_points, subset_indices, rvec, tvec,
                                                             camera_matrix, fisheye_model, threshold);

    if(inlier_index.size() > std::max(static_cast<int>(best_inlier_index.size()), num_model_points - 1))
    {
      double outlier_ratio = static_cast<double>(outlier_index.size()) / static_cast<double>(object_points.size());
      num_iters = ransac_update_num_iters_(confidence, outlier_ratio, num_iters, num_model_points);

      best_inlier_index = inlier_index;
      best_outlier_index = outlier_index;

      best_rvec = rvec;
      best_tvec = tvec;
    }
    // break;
  }

  return {best_rvec, best_tvec, best_inlier_index, best_outlier_index, iter};
}

namespace
{

/*
 * \brief cost functor which only allows for all parameters to vary
 */
struct ReprojectionCost
{
 public:
  ReprojectionCost(const cv::Point3d* object_point,
                   const cv::Point2d* image_point) :
      mObjectPoint(object_point),
      mImagePoint(image_point)
  { }

  template <typename T>
  bool operator()(const T* const camera_matrix, const T* const dist_coeffs,
                  const T* const rvec, const T* const tvec,
                  T* reprojection_error) const
  {
    T old_point[3] = {static_cast<T>(mObjectPoint->x),
                      static_cast<T>(mObjectPoint->y),
                      static_cast<T>(mObjectPoint->z)};
    T point[3];
    ceres::AngleAxisRotatePoint(rvec, old_point, point);
    point[0] += tvec[Coords::x];
    point[1] += tvec[Coords::y];
    point[2] += tvec[Coords::z];

    const T k1 = dist_coeffs[DistCoeffs::k1];
    const T k2 = dist_coeffs[DistCoeffs::k2];
    const T k3 = dist_coeffs[DistCoeffs::k3];
    const T k4 = dist_coeffs[DistCoeffs::k4];

    const auto xp = point[Coords::x] / point[Coords::z];
    const auto yp = point[Coords::y] / point[Coords::z];

    const auto r = sqrt(xp * xp + yp * yp);
    const auto theta = atan(r);

    const T theta_r = (r != static_cast<T>(0)) ? theta * (static_cast<T>(1) + k1 * theta * theta
                                                          + k2 * pow(theta, 4)
                                                          + k3 * pow(theta, 6)
                                                          + k4 * pow(theta, 8)) / r
                                               : static_cast<T>(0);

    const T xpp = theta_r * xp;
    const T ypp = theta_r * yp;

    const T fx = camera_matrix[CameraIntrinsics::fx];
    const T fy = camera_matrix[CameraIntrinsics::fy];
    const T cx = camera_matrix[CameraIntrinsics::cx];
    const T cy = camera_matrix[CameraIntrinsics::cy];

    const T u = xpp * fx + cx;
    const T v = ypp * fy + cy;

    reprojection_error[Coords::x] = u - static_cast<T>(mImagePoint->x);
    reprojection_error[Coords::y] = v - static_cast<T>(mImagePoint->y);

    return true;
  }

  static ceres::CostFunction* Create(const cv::Point3d* object_point,
                                     const cv::Point2d* image_point)
  {
    // each residual block returns a single number (1),takes a rotation vector (3), and a translation vector (3)
    return (new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 4, 4, 3, 3>(new ReprojectionCost(object_point,
                                                                                                  image_point)));
  }

 private:
  const cv::Point3d* mObjectPoint;
  const cv::Point2d* mImagePoint;
};

class LoggingCallback : public ceres::IterationCallback
{
 public:
  explicit LoggingCallback(bool log_to_stdout) :
      log_to_stdout_(log_to_stdout) {}

  ~LoggingCallback() {}

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override
  {
    if (log_to_stdout_)
    {
      if (summary.iteration == 0)
        std::cout << "Iteration\tcost" << std::endl;

      std::cout << summary.iteration << "\t" << summary.cost << std::endl;
    }

    return ceres::SOLVER_CONTINUE;
  }

 private:
  bool log_to_stdout_ = false;
};

}  // namespace anonymous

std::tuple<cv::Matx33d, cv::Matx<double, 1, 4>, cv::Vec3d, cv::Vec3d>
bundle_adjust(const std::vector<cv::Point3d>& object_points, const std::vector<cv::Point2d>& image_points,
              const cv::Matx33d& camera_matrix_, const cv::Matx<double, 1, 4>& fisheye_model_,
              const cv::Vec3d& rvec_, const cv::Vec3d& tvec_,
              ceres::LossFunction* loss_function, std::array<bool, 4> vary, bool print_summary = false)
{
  double camera_matrix[] = {camera_matrix_(0, 0), camera_matrix_(1, 1), camera_matrix_(0, 2), camera_matrix_(1, 2)};
  double fisheye_model[] = {fisheye_model_(0), fisheye_model_(1), fisheye_model_(2), fisheye_model_(3)};
  double rvec[] = {rvec_(0), rvec_(1), rvec_(2)};
  double tvec[] = {tvec_(0), tvec_(1), tvec_(2)};

  ceres::Problem problem;

  for (size_t i = 0; i < object_points.size(); ++i) {
    ceres::CostFunction* cost_function = ReprojectionCost::Create(&(object_points[i]), &(image_points[i]));
    problem.AddResidualBlock(cost_function, loss_function, camera_matrix, fisheye_model, rvec, tvec);
  }

  if (!vary[0])
    problem.SetParameterBlockConstant(camera_matrix);
  if (!vary[1])
    problem.SetParameterBlockConstant(fisheye_model);
  if (!vary[2])
    problem.SetParameterBlockConstant(rvec);
  if (!vary[3])
    problem.SetParameterBlockConstant(tvec);

  ceres::Solver::Options options;
  options.use_nonmonotonic_steps = true;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.use_inner_iterations = true;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = print_summary;

  // LoggingCallback logging_callback(print_summary);
  // options.callbacks.push_back(&logging_callback);

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  if (print_summary)
    std::cout << summary.FullReport() << std::endl;

  cv::Matx33d new_camera_matrix(camera_matrix[CameraIntrinsics::fx], 0.0, camera_matrix[CameraIntrinsics::cx],
                                0.0, camera_matrix[CameraIntrinsics::fy], camera_matrix[CameraIntrinsics::cy],
                                0.0, 0.0, 1.0);
  cv::Matx<double, 1, 4> new_fisheye_model(fisheye_model[0], fisheye_model[1], fisheye_model[2], fisheye_model[3]);
  cv::Vec3d new_rvec(rvec[0], rvec[1], rvec[2]);
  cv::Vec3d new_tvec(tvec[0], tvec[1], tvec[2]);

  return {new_camera_matrix, new_fisheye_model, new_rvec, new_tvec};
}

};  // namespace assignments
