#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>

#include <jsoncpp/json/json.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <vector>
#include <tuple>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "utilities.hpp"

namespace assignments::calibration
{

Detections load_correspondence(const std::string& json_file);

namespace
{

template <typename T>
class CalibrationCost
{
 public:
  CalibrationCost(const Vector3<T>& gridPoint,
                  const Vector2<T>& imagePoint) :
    mGridPoint(gridPoint),
    mImagePoint(imagePoint)
  { }

  template <typename S>
  bool operator()(const S* const intrinsics,
                  const S* const cameraMainFromGrid,
                  const S* const cameraFromCameraMain,
                  S* residual) const
  {
    const auto cameraMainFromGrid_ = reinterpret_cast<const Transform<S>*>(cameraMainFromGrid);
    // if this is the main camera then this should be identity
    const auto cameraFromCameraMain_ = reinterpret_cast<const Transform<S>*>(cameraFromCameraMain);

    const auto gridPoint = Vector3<S>{S(mGridPoint[0]), S(mGridPoint[1]), S(mGridPoint[2])};
    auto cameraMainPoint = cameraMainFromGrid_->transform(gridPoint.data());
    auto cameraPoint = cameraFromCameraMain_->transform(cameraMainPoint.data());

    const auto intrinsics_ = reinterpret_cast<const Intrinsics<S>*>(intrinsics);

    auto projectedPoint = intrinsics_->projectPoint(cameraPoint.data());

    const auto imagePoint = Vector3<S>{S(mImagePoint[0]), S(mImagePoint[1])};
    residual[Coords::x] = projectedPoint[Coords::x] - imagePoint[Coords::x];
    residual[Coords::y] = projectedPoint[Coords::y] - imagePoint[Coords::y];

    return true;
  }

  static ceres::CostFunction* Create(const Vector3<T>& gridPoint,
                                     const Vector2<T>& imagePoint)
  {
    return (new ceres::AutoDiffCostFunction<CalibrationCost, 2, 8, 6, 6>(new CalibrationCost(gridPoint,
                                                                                             imagePoint)));
  }

 private:
  const Vector3<T> mGridPoint;
  const Vector2<T> mImagePoint;
};

}  // namespace anonymous

/*
 * \brief uses method described in "A Flexible New Technique for Camera Calibration" by Zhengyou Zhang. December 2, 1998
 */
class Calibration
{
 public:
  Calibration(const cv::Size& imageSize,
              const std::vector<Detections>& detections,
              const bool verbose=false);

  void calibrate();

  Intrinsics<double> intrinsics(const size_t camera) const noexcept;

  Transform<double> transform(const size_t camera) const noexcept;

  std::tuple<double, double, double, double> errors();

 private:
  void estimateIntrinsics();

  void estimateExtrinsics();

  void optimizeIntrinsics();

  cv::Size mImageSize;
  std::vector<Detections> mDetections;
  std::vector<Intrinsics<double>> mIntrinsics;
  std::vector<std::unordered_map<std::string, std::vector<int>>> mInlierIndices;
  std::vector<Transform<double>> mCameraMainFromGrid;
  std::vector<Transform<double>> mCameraFromCameraMain;
  size_t mNumCameras;
  bool mVerbose;
  double mMinError = std::numeric_limits<double>::max();
  double mMaxError = std::numeric_limits<double>::min();
  double mMeanError = 0;
  double mRMSError = 0;
  int mRootCamera = 0;
};

}  // namespace assignments::calibration

