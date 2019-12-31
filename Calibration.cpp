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
  // Zhang's method of intrisic estimation.

  cv::Mat V = cv::Mat::zeros(2 * numImages, 6, CV_64F);
  for (size_t frame = 0; frame < numImages; ++frame)
  {
    std::vector<cv::Point2d> reducedGridPoints;
    for (const auto& gridPoint : mFramedGridPoints[frame])
      reducedGridPoints.push_back({gridPoint.x, gridPoint.y});

    cv::Mat homographyMatrix = cv::findHomography(reducedGridPoints, mFramedImagePoints[frame]);

    const auto G1 = vij(0, 1, homographyMatrix);
    const auto G2 = vij(0, 0, homographyMatrix) - vij(1, 1, homographyMatrix);

    V.at<double>(2 * frame, 0) = G1(0);
    V.at<double>(2 * frame, 1) = G1(1);
    V.at<double>(2 * frame, 2) = G1(2);
    V.at<double>(2 * frame, 3) = G1(3);
    V.at<double>(2 * frame, 4) = G1(4);
    V.at<double>(2 * frame, 5) = G1(5);

    V.at<double>(2 * frame + 1, 0) = G2(0);
    V.at<double>(2 * frame + 1, 1) = G2(1);
    V.at<double>(2 * frame + 1, 2) = G2(2);
    V.at<double>(2 * frame + 1, 3) = G2(3);
    V.at<double>(2 * frame + 1, 4) = G2(4);
    V.at<double>(2 * frame + 1, 5) = G2(5);
  }

  cv::Mat U, S, VT;
  cv::SVD::compute(V, S, U, VT);

  double minVal, maxVal;
  int minLoc, maxLoc;

  cv::minMaxIdx(S, &minVal, &maxVal, &minLoc, &maxLoc);

  cv::Vec6d b(VT.at<double>(minLoc,0),
              VT.at<double>(minLoc,1),
              VT.at<double>(minLoc,2),
              VT.at<double>(minLoc,3),
              VT.at<double>(minLoc,4),
              VT.at<double>(minLoc,5));

  double w = b(0) * b(2) * b(5) - b(1) * b(1) * b(5) + 2 * b(1) * b(3) * b(4) - b(2) * b(3) * b(3);
  double d = b(0) * b(2) - b(1) * b(1);

  double fx = sqrt(w / (d * b(0)));
  double fy = sqrt(w / (d * d) * b(0));
  double cx = (b(1) * b(4) - b(2) * b(3)) / d;
  double cy = (b(1) * b(3) - b(0) * b(4))/ d;

  mCameraMatrix(0, 0) = fx;
  mCameraMatrix(0, 2) = cx;
  mCameraMatrix(1, 1) = fy;
  mCameraMatrix(1, 2) = cy;

  std::cout << "Initial Intrinsics Guess:" << std::endl << mCameraMatrix << std::endl;
}

void Calibration::estimateExtrinsics()
{
  for (size_t frame = 0; frame < mFramedGridPoints.size(); ++frame)
  {
    std::vector<int> inlierIdx;
    std::vector<int> outlierIdx;
    size_t iter;
    std::tie(mRVecs[frame], mTVecs[frame], inlierIdx, outlierIdx, iter) =
        fisheye_solvePnPRansac(mFramedGridPoints[frame], mFramedImagePoints[frame], mCameraMatrix, mDistortionCoeffs);

    // std::vector<cv::Point2d> framedUndistortedPoints;
    // cv::fisheye::undistortPoints(mFramedImagePoints[frame], framedUndistortedPoints, mCameraMatrix, mDistortionCoeffs,
    //                              cv::noArray(), mCameraMatrix);
    // cv::solvePnPRansac(mFramedGridPoints[frame], framedUndistortedPoints, mCameraMatrix, mDistortionCoeffs,
    //                    mRVecs[frame], mTVecs[frame]);
  }
  // const auto& [RMSE, meanError, minError, maxError] = errors(true);

  // std::cout << "Intrinsics:" << std::endl << mCameraMatrix << std::endl;
  // std::cout << "Distortion Coeffs: " << mDistortionCoeffs << std::endl;
  // std::cout << "Root Mean Squared Error: " << RMSE << std::endl;
  // std::cout << "Mean Error: " << meanError << std::endl;
  // std::cout << "Minimum Error: " << minError << std::endl;
  // std::cout << "Maximum Error: " << maxError << std::endl;

  // // result errors
  // mRMSError = 0;
  // mMeanError = 0;
  // mMaxError = std::numeric_limits<double>::min();
  // mMinError = std::numeric_limits<double>::max();

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
    for (size_t j = 0; j < mFramedImagePoints[frame].size(); ++j)
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
  // options.use_nonmonotonic_steps = true;
  // options.preconditioner_type = ceres::SCHUR_JACOBI;
  // options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  // options.use_inner_iterations = true;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = mVerbose;

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
    for (size_t j = 0; j < mFramedImagePoints[frame].size(); ++j)
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

  // std::cout << "Intrinsics:" << std::endl << mCameraMatrix << std::endl;
  // std::cout << "Distortion Coeffs: " << mDistortionCoeffs << std::endl;
  // std::cout << "Root Mean Squared Error: " << mRMSError << std::endl;
  // std::cout << "Mean Error: " << mMeanError << std::endl;
  // std::cout << "Minimum Error: " << mMinError << std::endl;
  // std::cout << "Maximum Error: " << mMaxError << std::endl;

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

      for (size_t i = 0; i < mFramedGridPoints[frame].size(); ++i)
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
