#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>

#include <jsoncpp/json/json.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <vector>
#include <tuple>
#include <limits>

#include "utilities.hpp"

namespace assignments::calibration
{

std::tuple<std::vector<std::vector<cv::Point3d>>, std::vector<std::vector<cv::Point2d>>>
load_correspondence(const std::string& json_file);

namespace
{

enum Intrinsics
{
  fx = 0,
  fy = 1,
  cx = 2,
  cy = 3
};

enum DistCoeffs
{
  k1 = 0,
  k2 = 1,
  k3 = 2,
  k4 = 3
};

enum Coords
{
  x = 0,
  y = 1,
  z = 2
};

/*
 * \brief cost functor which only allows for all parameters to vary
 */
class CalibrationCost
{
 public:
  CalibrationCost(const cv::Point3d* object_point,
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

    const T fx = camera_matrix[Intrinsics::fx];
    const T fy = camera_matrix[Intrinsics::fy];
    const T cx = camera_matrix[Intrinsics::cx];
    const T cy = camera_matrix[Intrinsics::cy];

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
    return (new ceres::AutoDiffCostFunction<CalibrationCost, 2, 4, 4, 3, 3>(new CalibrationCost(object_point,
                                                                                                image_point)));
  }

 private:
  const cv::Point3d* mObjectPoint;
  const cv::Point2d* mImagePoint;
};

}  // namespace anonymous

/*
 * \brief uses method described in "A Flexible New Technique for Camera Calibration" by Zhengyou Zhang. December 2, 1998
 */
class Calibration
{
 public:
  Calibration(const cv::Size& imageSize,
              const std::vector<std::vector<cv::Point3d>>& framedGridPoints,
              const std::vector<std::vector<cv::Point2d>>& framedImagePoints,
              const bool verbose = false);

  void calibrate();

  cv::Matx33d cameraMatrix() const;

  cv::Matx<double, 1, 4> distortionCoeffs() const;

  std::tuple<double, double, double, double> errors() const;

 private:
  inline cv::Vec6d vij(const size_t i, const size_t j, const cv::Mat& H);

  void estimateIntrinsics();

  void estimateExtrinsics();

  void optimizeIntrinsics();

  cv::Size mImageSize;
  std::vector<std::vector<cv::Point3d>> mFramedGridPoints;
  std::vector<std::vector<cv::Point2d>> mFramedImagePoints;
  cv::Matx33d mCameraMatrix;
  cv::Matx<double, 1, 4> mDistortionCoeffs;
  std::vector<cv::Vec3d> mRVecs;
  std::vector<cv::Vec3d> mTVecs;
  bool mVerbose;
};

}  // namespace assignments::calibration

