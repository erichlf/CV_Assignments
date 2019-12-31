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
  bool operator()(const T* const cameraMatrix, const T* const distortionCoeffs,
                  const T* const rvec, const T* const tvec,
                  T* residual) const
  {
    T objectPoint[3] = {static_cast<T>(mObjectPoint->x),
                        static_cast<T>(mObjectPoint->y),
                        static_cast<T>(mObjectPoint->z)};
    T imagePoint[2] = {static_cast<T>(mImagePoint->x),
                       static_cast<T>(mImagePoint->y)};

    fisheyeReprojectionError(objectPoint, imagePoint, cameraMatrix, distortionCoeffs, rvec, tvec, residual);

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

  std::tuple<double, double, double, double> errors(bool recalculate=false);

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
  double mMinError = std::numeric_limits<double>::max();
  double mMaxError = std::numeric_limits<double>::min();
  double mMeanError = 0;
  double mRMSError = 0;
};

}  // namespace assignments::calibration

