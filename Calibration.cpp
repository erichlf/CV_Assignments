#include "Calibration.hpp"

namespace assignments::calibration
{

/*
 * \brief given a json file containing correspondences build the correspondences
 * \param json_file   string containing the location of the json file to read
 * \return {object_points, image_points}  correspondences between 3D and 2D
 */
std::tuple<std::vector<std::vector<cv::Point3d>>, std::vector<std::vector<cv::Point2d>>>
load_correspondence(const std::string& json_file)
{
  std::ifstream ifs(json_file);
  Json::Value correspondences_json;
  ifs >> correspondences_json;

  std::vector<std::vector<cv::Point3d>> frames_world_points;
  std::vector<std::vector<cv::Point2d>> frames_image_points;
  const auto frames = correspondences_json["correspondences"];
  int frame_number = 0;
  for (const auto& frame : frames)
  {
    std::vector<cv::Point3d> world_points;
    std::vector<cv::Point2d> image_points;
    int corr = 0;
    for (const auto& correspondence : frame[std::to_string(frame_number)])
    {
      if (corr % 10)
        continue;

      world_points.push_back({correspondence["grid_point"][0].asDouble(),
                              correspondence["grid_point"][1].asDouble(),
                              correspondence["grid_point"][2].asDouble()});
      image_points.push_back({correspondence["image_point"][0].asDouble(),
                              correspondence["image_point"][1].asDouble()});
    }
    frames_world_points.emplace_back(world_points);
    frames_image_points.emplace_back(image_points);

    ++frame_number;
  }

  return {frames_world_points, frames_image_points};
}

Calibration::Calibration(const cv::Size& imageSize,
                         const std::vector<std::vector<cv::Point3d>>& framedGridPoints,
                         const std::vector<std::vector<cv::Point2d>>& framedImagePoints,
                         const bool verbose/*=false*/) :
    mImageSize(std::move(imageSize)),
    mFramedGridPoints(std::move(framedGridPoints)),
    mFramedImagePoints(std::move(framedImagePoints)),
    mCameraMatrix(1, 0, imageSize.width / 2.0,
                  0, 1, imageSize.height / 2.0,
                  0, 0, 1),
    mDistortionCoeffs(0, 0, 0, 0),
    mVerbose(verbose)
{
  mFramedInlierIndices.resize(mFramedImagePoints.size());
  mRVecs.resize(mFramedImagePoints.size());
  mTVecs.resize(mFramedImagePoints.size());
}

inline cv::Vec6d Calibration::vij(const size_t i, const size_t j, const cv::Mat& H)
{
  return cv::Vec6d(H.at<double>(0, i) * H.at<double>(0, j),
                   H.at<double>(0, i) * H.at<double>(1, j) + H.at<double>(1, i) * H.at<double>(0, j),
                   H.at<double>(1, i) * H.at<double>(1, j),
                   H.at<double>(2, i) * H.at<double>(0, j) + H.at<double>(0, i) * H.at<double>(2, j),
                   H.at<double>(2, i) * H.at<double>(1, j) + H.at<double>(1, i) * H.at<double>(2, j),
                   H.at<double>(2, i) * H.at<double>(2, j));
}

void Calibration::estimateIntrinsics()
{
  size_t numImages = mFramedImagePoints.size();
  size_t numSubImage = 35;
  size_t stride = numImages / numSubImage; // This code is extremely slow to run, when running on the 681300 points
  cv::Mat cvDistortionCoeffs = cv::Mat::zeros(1, 4, CV_64F);  // temp place holder so that calibrate can be used
  std::vector<cv::Vec3d> cvRVecs;
  std::vector<cv::Vec3d> cvTVecs;

  std::vector<std::vector<cv::Point3d>> subGrid;
  std::vector<std::vector<cv::Point2d>> subImage;
  for (size_t i = 0; i < numImages; i += stride)
  {
    subGrid.push_back(mFramedGridPoints[i]);
    subImage.push_back(mFramedImagePoints[i]);
  }
  cv::fisheye::calibrate(subGrid, subImage, mImageSize, mCameraMatrix, cvDistortionCoeffs, cv::noArray(), cv::noArray(),
                         cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | cv::fisheye::CALIB_FIX_SKEW);

  mDistortionCoeffs(0) = cvDistortionCoeffs.at<double>(0, 0);
  mDistortionCoeffs(1) = cvDistortionCoeffs.at<double>(0, 1);
  mDistortionCoeffs(2) = cvDistortionCoeffs.at<double>(0, 2);
  mDistortionCoeffs(3) = cvDistortionCoeffs.at<double>(0, 3);

  return;
}

void Calibration::estimateExtrinsics()
{
  for (size_t frame = 0; frame < mFramedGridPoints.size(); ++frame)
  {
    std::vector<cv::Point2d> framedUndistortedPoints;
    cv::Vec4d noDistortion{0, 0, 0, 0};
    cv::fisheye::undistortPoints(mFramedImagePoints[frame], framedUndistortedPoints, mCameraMatrix, mDistortionCoeffs,
                                 cv::noArray(), mCameraMatrix);
    cv::solvePnPRansac(mFramedGridPoints[frame], framedUndistortedPoints, mCameraMatrix, noDistortion,
                       mRVecs[frame], mTVecs[frame], false, 0.8*mFramedGridPoints[frame].size(), 1.0, 0.99,
                       mFramedInlierIndices[frame]);
  }
}

void Calibration::optimizeIntrinsics()
{
  double cameraMatrix[] = {mCameraMatrix(0, 0), mCameraMatrix(1, 1), mCameraMatrix(0, 2), mCameraMatrix(1, 2)};
  double distortionCoeffs[] = {mDistortionCoeffs(0, 0), mDistortionCoeffs(0, 1),
                               mDistortionCoeffs(0, 2), mDistortionCoeffs(0, 3)};
  std::vector<double[3]> rvecs(mRVecs.size());
  std::vector<double[3]> tvecs(mTVecs.size());

  ceres::Problem problem;
  ceres::LossFunction* lossFunction = new ceres::HuberLoss(0.1);

  // create residuals for each observation
  size_t numObservations = 0;
  for (size_t frame = 0; frame < mFramedImagePoints.size(); ++frame)
  {
    rvecs[frame][0] = mRVecs[frame][0]; rvecs[frame][1] = mRVecs[frame][1]; rvecs[frame][2] = mRVecs[frame][2];
    tvecs[frame][0] = mTVecs[frame][0]; tvecs[frame][1] = mTVecs[frame][1]; tvecs[frame][2] = mTVecs[frame][2];
    for (const auto& j : mFramedInlierIndices[frame])
    {
      ceres::CostFunction* costFunction = CalibrationCost::Create(&(mFramedGridPoints[frame][j]),
                                                                  &(mFramedImagePoints[frame][j]));

      problem.AddResidualBlock(costFunction, lossFunction, cameraMatrix, distortionCoeffs,
                               rvecs[frame], tvecs[frame]);
      ++numObservations;
    }
  }

  ceres::Solver::Options options;
  options.num_threads = 10;
  options.minimizer_progress_to_stdout = mVerbose;
  options.update_state_every_iteration = true;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options.max_num_iterations = 75;
  options.use_inner_iterations = false;
  options.function_tolerance = 1e-6;
  options.gradient_tolerance = 1e-10;
  options.parameter_tolerance = 1e-8;
  options.max_trust_region_radius = 1e12;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  if (mVerbose)
  {
    std::cout << summary.FullReport() << std::endl;
  }

  // update all our data
  mDistortionCoeffs(DistCoeffs::k1) = distortionCoeffs[DistCoeffs::k1];
  mDistortionCoeffs(DistCoeffs::k2) = distortionCoeffs[DistCoeffs::k2];
  mDistortionCoeffs(DistCoeffs::k3) = distortionCoeffs[DistCoeffs::k3];
  mDistortionCoeffs(DistCoeffs::k4) = distortionCoeffs[DistCoeffs::k4];

  mCameraMatrix(0, 0) = cameraMatrix[0];
  mCameraMatrix(1, 1) = cameraMatrix[1];
  mCameraMatrix(0, 2) = cameraMatrix[2];
  mCameraMatrix(1, 2) = cameraMatrix[3];

  for (size_t frame = 0; frame < mFramedImagePoints.size(); ++frame)
  {
    mRVecs[frame](0) = rvecs[frame][0]; mRVecs[frame](1) = rvecs[frame][1]; mRVecs[frame](2) = rvecs[frame][2];
    mTVecs[frame](0) = tvecs[frame][0]; mTVecs[frame](1) = tvecs[frame][1]; mTVecs[frame](2) = tvecs[frame][2];
    for (const auto& j : mFramedInlierIndices[frame])
    {
      double residual[2];

      double objectPoint[3] = {mFramedGridPoints[frame][j].x,
                               mFramedGridPoints[frame][j].y,
                               mFramedGridPoints[frame][j].z};
      double imagePoint[2] = {mFramedImagePoints[frame][j].x,
                              mFramedImagePoints[frame][j].y};

      fisheyeReprojectionError(objectPoint, imagePoint, cameraMatrix, distortionCoeffs,
                               rvecs[frame], tvecs[frame], residual);

      double squaredError = residual[0] * residual[0] + residual[1] * residual[1];
      double dist = sqrt(squaredError);

      mRMSError += squaredError;
      mMeanError += dist;
      mMinError = std::min(dist, mMinError);
      mMaxError = std::max(dist, mMaxError);
    }
  }

  mMeanError /= numObservations;
  mRMSError = sqrt(mRMSError / numObservations);
}

void Calibration::calibrate()
{
  // find the initial estimate of the camera matrix
  estimateIntrinsics();
  // estimate the position of the camera wrt to each frame
  estimateExtrinsics();
  // run ceres optimization to get closer camera extrinsics
  optimizeIntrinsics();
}

std::tuple<double, double, double, double> Calibration::errors(bool recalculate/*=false*/)
{
  if (recalculate)
  {
    double cameraMatrix[4] = {mCameraMatrix(0, 0), mCameraMatrix(1, 1), mCameraMatrix(0, 2), mCameraMatrix(1, 2)};
    double distortionCoeffs[4] = {mDistortionCoeffs(0), mDistortionCoeffs(1), mDistortionCoeffs(2), mDistortionCoeffs(3)};

    mMinError = std::numeric_limits<double>::max();
    mMaxError = std::numeric_limits<double>::min();
    mMeanError = 0;
    mRMSError = 0;

    size_t numObservations = 0;
    for (size_t frame = 0; frame < mFramedGridPoints.size(); ++frame)
    {
      double rvec[3] = {mRVecs[frame][0], mRVecs[frame][1], mRVecs[frame][2]};
      double tvec[3] = {mTVecs[frame][0], mTVecs[frame][1], mTVecs[frame][2]};

      for (const auto& i : mFramedInlierIndices[frame])
      {
        double objectPoint[3] = {mFramedGridPoints[frame][i].x,
                                 mFramedGridPoints[frame][i].y,
                                 mFramedGridPoints[frame][i].z};
        double imagePoint[2] = {mFramedImagePoints[frame][i].x,
                                mFramedImagePoints[frame][i].y};

        double residual[2];

        fisheyeReprojectionError(objectPoint, imagePoint, cameraMatrix, distortionCoeffs, rvec, tvec, residual);

        double squaredError = residual[0] * residual[0] + residual[1] * residual[1];

        mRMSError += squaredError;
        double dist = sqrt(squaredError);
        mMeanError += dist;

        mMinError = std::min(mMinError, dist);
        mMaxError = std::max(mMaxError, dist);

        ++numObservations;
      }
    }

    mMeanError /= numObservations;
    mRMSError = sqrt(mRMSError / numObservations);
  }

  return {mRMSError, mMeanError, mMinError, mMaxError};
}

cv::Matx33d Calibration::cameraMatrix() const { return mCameraMatrix; }

cv::Matx<double, 1, 4> Calibration::distortionCoeffs() const { return mDistortionCoeffs; }

}  // namespace assignments::calibration
