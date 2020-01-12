#include "Calibration.hpp"

namespace assignments::calibration
{

/*
 * \brief given a json file containing correspondences build the correspondences
 * \param json_file   string containing the location of the json file to read
 * \return Detections{object_points, image_points, frames}  correspondences between 3D and 2D
 */
Detections load_correspondence(const std::string& json_file)
{
  std::ifstream ifs(json_file);
  Json::Value correspondenceJson;
  ifs >> correspondenceJson;

  Detections detections;
  detections.maxFrame = std::numeric_limits<int>::min();

  std::vector<std::vector<Vector3<double>>> framesWorldPoints;
  std::vector<std::vector<Vector2<double>>> framesImagePoints;
  const auto frames = correspondenceJson["correspondences"];

  size_t frameIndex = 0;
  for (const auto& frame : frames)
  {
    std::string frameNumber = frame.getMemberNames()[0];

    detections.gridPoints.emplace_back(std::vector<Vector3<double>>());
    detections.imagePoints.emplace_back(std::vector<Vector2<double>>());
    int corr = 0;
    for (const auto& correspondence : frame[frameNumber])
    {
      detections.gridPoints[frameIndex].push_back({correspondence["grid_point"][0].asDouble(),
                                                   correspondence["grid_point"][1].asDouble(),
                                                   correspondence["grid_point"][2].asDouble()});
      detections.imagePoints[frameIndex].push_back({correspondence["image_point"][0].asDouble(),
                                                    correspondence["image_point"][1].asDouble()});
    }
    detections.frames[frameNumber] = frameIndex++;
    detections.maxFrame = std::max(std::stoi(frameNumber), detections.maxFrame);
  }

  return detections;
}

Calibration::Calibration(const cv::Size& imageSize,
                         const std::vector<Detections>& detections,
                         const bool verbose/*=false*/) :
  mImageSize(std::move(imageSize)),
  mDetections(std::move(detections)),
  mVerbose(verbose)
{
  mNumCameras = mDetections.size();
  mInlierIndices = std::vector<std::unordered_map<std::string, std::vector<int>>>(mNumCameras);

  // determine the total number of frames so that we can allocate the memory for our cameraMainFromGrid
  int maxFrame = std::numeric_limits<int>::min();
  for (size_t camera = 0; camera < mDetections.size(); ++camera)
    maxFrame = std::max(mDetections[camera].maxFrame, maxFrame);

  mCameraMainFromGrid.resize(maxFrame + 1);  // assume index starts at 0

  mCameraFromCameraMain.resize(mNumCameras);
  mIntrinsics.resize(mNumCameras);
}

void Calibration::estimateIntrinsics()
{
  cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat distortionCoeffs = cv::Mat::zeros(1, 4, CV_64F);  // temp place holder so that calibrate can be used
  size_t numSubImage = 35;
  for (size_t camera = 0; camera < mNumCameras; ++camera)
  {
    size_t stride = mDetections[camera].frames.size() / numSubImage;
    std::vector<std::vector<cv::Point3d>> subGrid;
    std::vector<std::vector<cv::Point2d>> subImage;
    for (int frameIdx = 0; frameIdx < mDetections[camera].frames.size(); frameIdx += stride)
    {
      const auto frame = mDetections[camera].frames.find(std::to_string(frameIdx));
      if (frame == mDetections[camera].frames.end())
        continue;

      const auto gridPoints = convertToPoints(mDetections[camera].gridPoints[frame->second]);
      const auto imagePoints = convertToPoints(mDetections[camera].imagePoints[frame->second]);

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

    for (const auto& frame : mDetections[camera].frames)
    {
      int frameIDX = std::stoi(frame.first);
      const auto gridPoints = convertToPoints(mDetections[camera].gridPoints[frame.second]);
      const auto imagePoints = convertToPoints(mDetections[camera].imagePoints[frame.second]);
      std::vector<cv::Point2d> undistortedPoints;
      cv::fisheye::undistortPoints(imagePoints, undistortedPoints, cameraMatrix, distortionCoeffs, cv::noArray(),
                                   cameraMatrix);

      cv::Vec3d rvec;
      cv::Vec3d tvec;

      std::vector<int> inlierIndices;
      cv::solvePnPRansac(gridPoints, undistortedPoints, cameraMatrix, noDistortion, rvec, tvec, false,
                         0.8*mDetections[camera].gridPoints[frame.second].size(), 1.0, 0.99, inlierIndices);

      mInlierIndices[camera][frame.first] = inlierIndices;

      if (camera == mRootCamera)
      {
        mCameraFromCameraMain[camera] = Transform<double>(0, 0, 0, 0, 0, 0);
        mCameraMainFromGrid[frameIDX] = Transform<double>(rvec, tvec);
      }
      else if (mDetections[mRootCamera].frames.count(frame.first))  // frame exists in the main camera
      {
        Vector3<double> r;
        Vector3<double> t;

        r = mCameraMainFromGrid[frameIDX].R();
        t = mCameraMainFromGrid[frameIDX].t();

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
    for (const auto& frame : mDetections[camera].frames)
    {
      if (!mCameraMainFromGrid[std::stoi(frame.first)].initialized())  // skip missing frames
        continue;
      auto cameraMainFromGrid = reinterpret_cast<double*>(mCameraMainFromGrid.data() + std::stoi(frame.first));
      for (const auto& point : mInlierIndices[camera][frame.first])
      {
        const auto gridPoints = mDetections[camera].gridPoints[frame.second];
        const auto imagePoints = mDetections[camera].imagePoints[frame.second];
        ceres::CostFunction* costFunction = CalibrationCost<double>::Create(gridPoints[point],
                                                                            imagePoints[point]);

        problem.AddResidualBlock(costFunction, lossFunction, intrinsics,
                                 cameraMainFromGrid, cameraFromCameraMain);
        ++numObservations;
      }
    }
    if (camera == mRootCamera)  // no need to calculate camera from camera extrinsics when we are on the main camera
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

std::tuple<double, double, double, double> Calibration::errors()
{
  mMinError = std::numeric_limits<double>::max();
  mMaxError = std::numeric_limits<double>::min();
  mMeanError = 0;
  mRMSError = 0;

  size_t numObservations = 0;

  for (size_t camera = 0; camera < mNumCameras; ++camera)
  {
    for (const auto& frame : mDetections[camera].frames)
    {
      const auto gridPoints = mDetections[camera].gridPoints[frame.second];
      const auto imagePoints = mDetections[camera].imagePoints[frame.second];
      for (const auto& point : mInlierIndices[camera][frame.first])
      {
        if (!mCameraMainFromGrid[std::stoi(frame.first)].initialized())  // skip missing frames
          continue;
        double residual[2];
        const auto cameraMainPoint = mCameraMainFromGrid[std::stoi(frame.first)].transform(gridPoints[point].data());
        const auto cameraPoint = mCameraFromCameraMain[camera].transform(cameraMainPoint.data());

        auto projectedPoint = mIntrinsics[camera].projectPoint(cameraPoint.data());

        residual[Coords::x] = projectedPoint[Coords::x] - imagePoints[point][Coords::x];
        residual[Coords::y] = projectedPoint[Coords::y] - imagePoints[point][Coords::y];

        double squaredError = residual[Coords::x] * residual[Coords::x] + residual[Coords::y] * residual[Coords::y];
        double dist = sqrt(squaredError);

        mRMSError += squaredError;
        mMeanError += dist;
        mMinError = std::min(dist, mMinError);
        mMaxError = std::max(dist, mMaxError);

        ++numObservations;
      }
    }
  }

  mMeanError /= numObservations;
  mRMSError = sqrt(mRMSError / numObservations);

  return {mRMSError, mMeanError, mMinError, mMaxError};
}

Intrinsics<double> Calibration::intrinsics(const size_t camera) const noexcept
{
  return mIntrinsics[camera];
}

Transform<double> Calibration::transform(const size_t camera) const noexcept
{
  return mCameraFromCameraMain[camera];
}

}  // namespace assignments::calibration
