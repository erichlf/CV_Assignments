#include "Calibration.hpp"

namespace assignments::calibration
{

/*
 * \brief given a json file containing correspondences build the correspondences
 * \param json_file   string containing the location of the json file to read
 * \return {object_points, image_points}  correspondences between 3D and 2D
 */
std::tuple<std::vector<std::vector<Vector3<double>>>, std::vector<std::vector<Vector2<double>>>>
load_correspondence(const std::string& json_file)
{
  std::ifstream ifs(json_file);
  Json::Value correspondenceJson;
  ifs >> correspondenceJson;

  std::vector<std::vector<Vector3<double>>> framesWorldPoints;
  std::vector<std::vector<Vector2<double>>> framesImagePoints;
  const auto frames = correspondenceJson["correspondences"];
  std::string frameNumber;
  for (const auto& frame : frames)
  {
    frameNumber = frame.getMemberNames()[0];

    std::vector<Vector3<double>> worldPoints;
    std::vector<Vector2<double>> imagePoints;
    int corr = 0;
    for (const auto& correspondence : frame[frameNumber])
    {
      if (corr % 10)
        continue;

      worldPoints.push_back({correspondence["grid_point"][0].asDouble(),
                              correspondence["grid_point"][1].asDouble(),
                              correspondence["grid_point"][2].asDouble()});
      imagePoints.push_back({correspondence["image_point"][0].asDouble(),
                              correspondence["image_point"][1].asDouble()});
    }
    framesWorldPoints.emplace_back(worldPoints);
    framesImagePoints.emplace_back(imagePoints);
  }

  return {framesWorldPoints, framesImagePoints};
}

Calibration::Calibration(const cv::Size& imageSize,
                         const std::vector<std::vector<Vector3<double>>>& gridPoints,
                         const std::vector<std::vector<std::vector<Vector2<double>>>>& imagePoints,
                         const bool verbose/*=false*/) :
  mImageSize(std::move(imageSize)),
  mGridPoints(std::move(gridPoints)),
  mImagePoints(std::move(imagePoints)),
  mVerbose(verbose)
{
  mNumCameras = mImagePoints.size();
  mNumFrames = mGridPoints.size();
  mInlierIndices = std::vector<std::vector<std::vector<int>>>(mNumCameras, std::vector<std::vector<int>>(mNumFrames));
  mCameraFromCameraMain.resize(mNumCameras);
  mCameraMainFromGrid.resize(mNumFrames);
  mIntrinsics.resize(mNumCameras);
}

void Calibration::estimateIntrinsics()
{
  size_t numSubImage = 35;
  size_t stride = mNumFrames / numSubImage; // This code is extremely slow to run, when running on the 681300 points
  cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat distortionCoeffs = cv::Mat::zeros(1, 4, CV_64F);  // temp place holder so that calibrate can be used
  for (size_t camera = 0; camera < mNumCameras; ++camera)
  {
    std::vector<std::vector<cv::Point3d>> subGrid;
    std::vector<std::vector<cv::Point2d>> subImage;
    for (size_t frame = 0; frame < mNumFrames; frame += stride)
    {
      const auto gridPoints = convertToPoints(mGridPoints[frame]);
      const auto imagePoints = convertToPoints(mImagePoints[camera][frame]);

      subGrid.push_back(gridPoints);
      subImage.push_back(imagePoints);
    }
    cv::fisheye::calibrate(subGrid, subImage, mImageSize, cameraMatrix, distortionCoeffs, cv::noArray(), cv::noArray(),
                           cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | cv::fisheye::CALIB_FIX_SKEW);

    mIntrinsics[camera] = Intrinsics<double>(cameraMatrix, distortionCoeffs);
  }
}

void Calibration::estimateExtrinsics()
{
  cv::Vec4d noDistortion{0, 0, 0, 0};
  for (size_t camera = 0; camera < mNumCameras; ++camera)
  {
    const auto intrinsics = mIntrinsics[camera].intrinsics();
    const auto distCoeffs = mIntrinsics[camera].distCoeffs();
    cv::Matx33d cameraMatrix(intrinsics[0], 0, intrinsics[2],
                             0, intrinsics[1], intrinsics[3],
                             0, 0, 1);
    cv::Matx<double, 1, 4> distortionCoeffs(distCoeffs[0], distCoeffs[1], distCoeffs[2], distCoeffs[3]);

    for (size_t frame = 0; frame < mNumFrames; ++frame)
    {
      const auto gridPoints = convertToPoints(mGridPoints[frame]);
      const auto imagePoints = convertToPoints(mImagePoints[camera][frame]);
      std::vector<cv::Point2d> undistortedPoints;
      cv::fisheye::undistortPoints(imagePoints, undistortedPoints, cameraMatrix, distortionCoeffs, cv::noArray(),
                                   cameraMatrix);

      cv::Vec3d rvec;
      cv::Vec3d tvec;

      cv::solvePnPRansac(gridPoints, undistortedPoints, cameraMatrix, noDistortion, rvec, tvec, false,
                         0.8*mGridPoints[frame].size(), 1.0, 0.99, mInlierIndices[camera][frame]);

      if (camera == 0)
      {
        mCameraFromCameraMain[camera] = Transform<double>(0, 0, 0, 0, 0, 0);
        mCameraMainFromGrid[frame] = Transform<double>(rvec, tvec);
      }
      else
      {
        const auto r = mCameraMainFromGrid[frame].R();
        const auto t = mCameraMainFromGrid[frame].t();

        const auto cameraFromGrid = cv::Affine3f(rvec, tvec);
        const auto gridFromCameraMain = cv::Affine3f(cv::Vec3d(r[0], r[1], r[2]), cv::Vec3d(t[0], t[1], t[2])).inv();

        const auto cameraFromCameraMain = cameraFromGrid * gridFromCameraMain;

        rvec = cameraFromCameraMain.rvec();
        tvec = cameraFromCameraMain.translation();

        mCameraFromCameraMain[camera] = Transform<double>(rvec, tvec);
      }
    }
  }
}

void Calibration::optimizeIntrinsics()
{
  ceres::Problem problem;
  ceres::LossFunction* lossFunction = new ceres::HuberLoss(0.1);

  // create residuals for each observation
  size_t numObservations = 0;
  size_t camera = 0;
  for (size_t camera = 0; camera < mNumCameras; ++camera)
  {
    auto cameraFromCameraMain = reinterpret_cast<double*>(mCameraFromCameraMain.data() + camera);
    auto intrinsics = reinterpret_cast<double*>(mIntrinsics.data() + camera);
    for (size_t frame = 0; frame < mNumFrames; ++frame)
    {
      auto cameraMainFromGrid = reinterpret_cast<double*>(mCameraMainFromGrid.data() + frame);
      for (const auto& point : mInlierIndices[camera][frame])
      {
        ceres::CostFunction* costFunction = CalibrationCost<double>::Create(mGridPoints[frame][point],
                                                                            mImagePoints[camera][frame][point]);

        problem.AddResidualBlock(costFunction, lossFunction, intrinsics,
                                 cameraMainFromGrid, cameraFromCameraMain);
        ++numObservations;
      }
    }
    if (camera == 0)  // no need to calculate camera from camera extrinsics when we are on the main camera
      problem.SetParameterBlockConstant(cameraFromCameraMain);
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

  for (size_t camera = 0; camera < mNumCameras; ++camera)
  {
    for (size_t frame = 0; frame < mNumFrames; ++frame)
    {
      for (const auto& point : mInlierIndices[camera][frame])
      {
        double residual[2];

        auto cameraMainPoint = mCameraMainFromGrid[frame].transform(mGridPoints[frame][point].data());
        auto cameraPoint = mCameraFromCameraMain[camera].transform(cameraMainPoint.data());

        auto projectedPoint = mIntrinsics[camera].projectPoint(cameraPoint.data());

        residual[Coords::x] = projectedPoint[Coords::x] - mImagePoints[camera][frame][point][Coords::x];
        residual[Coords::y] = projectedPoint[Coords::y] - mImagePoints[camera][frame][point][Coords::y];

        double squaredError = residual[Coords::x] * residual[Coords::x] + residual[Coords::y] * residual[Coords::y];
        double dist = sqrt(squaredError);

        mRMSError += squaredError;
        mMeanError += dist;
        mMinError = std::min(dist, mMinError);
        mMaxError = std::max(dist, mMaxError);
      }
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
    mMinError = std::numeric_limits<double>::max();
    mMaxError = std::numeric_limits<double>::min();
    mMeanError = 0;
    mRMSError = 0;

    size_t numObservations = 0;

    for (size_t camera = 0; camera < mNumCameras; ++camera)
    {
      // if this is the main camera then this should be identity
      const auto cameraFromCameraMain = mCameraFromCameraMain[camera];
      const auto intrinsics = mIntrinsics[camera];

      for (size_t frame = 0; frame < mNumFrames; ++frame)
      {
        for (const auto& j : mInlierIndices[camera][frame])
        {
          double residual[2];

          const auto cameraMainFromGrid = mCameraMainFromGrid[frame];

          const auto gridPoint = Vector3<double>{mGridPoints[frame][j][0],
                                                 mGridPoints[frame][j][1],
                                                 mGridPoints[frame][j][2]};

          auto cameraMainPoint = cameraMainFromGrid.transform(gridPoint.data());
          auto cameraPoint = cameraFromCameraMain.transform(cameraMainPoint.data());

          auto projectedPoint = intrinsics.projectPoint(cameraPoint.data());

          const auto imagePoint = Vector3<double>{mImagePoints[camera][frame][j][0], mImagePoints[camera][frame][j][1]};
          residual[Coords::x] = projectedPoint[Coords::x] - imagePoint[Coords::x];
          residual[Coords::y] = projectedPoint[Coords::y] - imagePoint[Coords::y];

          double squaredError = residual[Coords::x] * residual[Coords::x] + residual[Coords::y] * residual[Coords::y];
          double dist = sqrt(squaredError);

          mRMSError += squaredError;
          mMeanError += dist;
          mMinError = std::min(dist, mMinError);
          mMaxError = std::max(dist, mMaxError);
        }
      }
    }

    mMeanError /= numObservations;
    mRMSError = sqrt(mRMSError / numObservations);
  }

  return {mRMSError, mMeanError, mMinError, mMaxError};
}

std::array<double, 4> Calibration::cameraMatrix(const size_t camera) const noexcept
{
  return mIntrinsics[camera].intrinsics();
}

std::array<double, 4> Calibration::distortionCoeffs(const size_t camera) const noexcept
{
  return mIntrinsics[camera].distCoeffs();
}

}  // namespace assignments::calibration
