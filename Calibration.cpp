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
    for (const auto& correspondence : frame[std::to_string(frame_number)])
    {
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

void Calibration::estimateIntrinsics()
{
  double cx = mCameraMatrix(0, 2);
  double cy = mCameraMatrix(1, 2);

  const auto numImages = mFramedImagePoints.size();

  cv::Mat A(numImages * 2, 2, CV_64F);
  cv::Mat b(numImages * 2, 1, CV_64F);

  for (size_t frame = 0; frame < numImages; ++frame)
  {
    std::vector<cv::Point2d> reducedGridPoints;
    for (const auto& gridPoint : mFramedGridPoints[frame])
      reducedGridPoints.push_back({gridPoint.x, gridPoint.y});

    cv::Mat homographyMatrix = cv::findHomography(reducedGridPoints, mFramedImagePoints[frame]);

    homographyMatrix.at<double>(0, 0) -= homographyMatrix.at<double>(2, 0) * cx;
    homographyMatrix.at<double>(0, 1) -= homographyMatrix.at<double>(2, 1) * cx;
    homographyMatrix.at<double>(0, 2) -= homographyMatrix.at<double>(2, 2) * cx;
    homographyMatrix.at<double>(1, 0) -= homographyMatrix.at<double>(2, 0) * cy;
    homographyMatrix.at<double>(1, 1) -= homographyMatrix.at<double>(2, 1) * cy;
    homographyMatrix.at<double>(1, 2) -= homographyMatrix.at<double>(2, 2) * cy;

    double h[3], v[3], d1[3], d2[3];
    double n[4] = {0, 0, 0, 0};

    for (int j = 0; j < 3; ++j)
    {
      double t0 = homographyMatrix.at<double>(j, 0);
      double t1 = homographyMatrix.at<double>(j, 1);
      h[j] = t0;
      v[j] = t1;
      d1[j] = (t0 + t1) * 0.5;
      d2[j] = (t0 - t1) * 0.5;
      n[0] += t0 * t0;
      n[1] += t1 * t1;
      n[2] += d1[j] * d1[j];
      n[3] += d2[j] * d2[j];
    }

    n[0] = 1.0 / sqrt(n[0]); n[1] = 1.0 / sqrt(n[1]); n[2] = 1.0 / sqrt(n[2]); n[3] = 1.0 / sqrt(n[3]);

    h[0] *= n[0]; h[1] *= n[0]; h[2] *= n[0];
    v[0] *= n[1]; v[1] *= n[1]; v[2] *= n[1];
    d1[0] *= n[2]; d1[1] *= n[2]; d1[2] *= n[2];
    d2[0] *= n[3]; d2[1] *= n[3]; d2[2] *= n[3];

    A.at<double>(frame * 2, 0) = h[0] * v[0];
    A.at<double>(frame * 2, 1) = h[1] * v[1];
    A.at<double>(frame * 2 + 1, 0) = d1[0] * d2[0];
    A.at<double>(frame * 2 + 1, 1) = d1[1] * d2[1];
    b.at<double>(frame * 2, 0) = -h[2] * v[2];
    b.at<double>(frame * 2 + 1, 0) = -d1[2] * d2[2];
  }

  cv::Vec2d f;
  cv::solve(A, b, f, cv::DECOMP_NORMAL | cv::DECOMP_LU);

  mCameraMatrix(0, 0) = sqrt(fabs(1.0 / f(0)));  // fx
  mCameraMatrix(1, 1) = sqrt(fabs(1.0 / f(1)));  // fy
}

void Calibration::estimateExtrinsics()
{
  for (size_t frame = 0; frame < mFramedGridPoints.size(); ++frame)
  {
    cv::solvePnPRansac(mFramedGridPoints[frame], mFramedImagePoints[frame], mCameraMatrix, cv::noArray(),
                       mRVecs[frame], mTVecs[frame]);
  }
}

void Calibration::optimizeIntrinsics()
{
  double cameraMatrix[] = {mCameraMatrix(0, 0), mCameraMatrix(1, 1), mCameraMatrix(0, 2), mCameraMatrix(1, 2)};
  double distortionCoeffs[] = {0, 0, 0, 0};
  std::vector<double[3]> rvecs(mRVecs.size());
  std::vector<double[3]> tvecs(mTVecs.size());
  for (size_t frame = 0; frame < mRVecs.size(); ++frame)
  {
    rvecs[frame][0] = mRVecs[frame][0]; rvecs[frame][1] = mRVecs[frame][1]; rvecs[frame][2] = mRVecs[frame][2];
    tvecs[frame][0] = mTVecs[frame][0]; tvecs[frame][1] = mTVecs[frame][1]; tvecs[frame][2] = mTVecs[frame][2];
  }

  ceres::Problem problem;
  ceres::LossFunction* lossFunction = new ceres::HuberLoss(1);

  // create residuals for each observation
  for (size_t frame = 0; frame < mFramedImagePoints.size(); ++frame)
  {
    for (size_t j = 0; j < mFramedImagePoints[frame].size(); ++j)
    {
      ceres::CostFunction* costFunction = CalibrationCost::Create(&(mFramedGridPoints[frame][j]),
                                                                  &(mFramedImagePoints[frame][j]));
      problem.AddResidualBlock(costFunction, lossFunction, cameraMatrix, distortionCoeffs,
                               rvecs[frame], tvecs[frame]);
    }
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = mVerbose;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  if (mVerbose)
  {
    std::cout << summary.FullReport() << std::endl;
  }

  // update all our data
  mDistortionCoeffs(0) = distortionCoeffs[0];
  mDistortionCoeffs(1) = distortionCoeffs[1];
  mDistortionCoeffs(2) = distortionCoeffs[2];
  mDistortionCoeffs(3) = distortionCoeffs[3];

  mCameraMatrix(0, 0) = cameraMatrix[0];
  mCameraMatrix(1, 1) = cameraMatrix[1];
  mCameraMatrix(0, 2) = cameraMatrix[2];
  mCameraMatrix(1, 2) = cameraMatrix[3];

  for (size_t frame = 0; frame < mFramedImagePoints.size(); ++frame)
  {
    mRVecs[frame](0) = rvecs[frame][0];
    mRVecs[frame](1) = rvecs[frame][1];
    mRVecs[frame](2) = rvecs[frame][2];
    mTVecs[frame](0) = tvecs[frame][0];
    mTVecs[frame](1) = tvecs[frame][1];
    mTVecs[frame](2) = tvecs[frame][2];
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

std::tuple<double, double, double, double> Calibration::errors() const
{
  double minError = std::numeric_limits<double>::max();
  double maxError = std::numeric_limits<double>::min();
  double meanError = 0;
  double rmsError = 0;

  size_t numObservations = 0;
  for (size_t frame = 0; frame < mFramedImagePoints.size(); ++frame)
  {
    std::vector<cv::Point2d> estimatedPoints;
    project_points(mFramedGridPoints[frame], mRVecs[frame], mTVecs[frame], mCameraMatrix, mDistortionCoeffs,
                   estimatedPoints);
    for (size_t i = 0; i < estimatedPoints.size(); ++i)
    {
      const auto diffX = estimatedPoints[i].x - mFramedImagePoints[frame][i].x;
      const auto diffY = estimatedPoints[i].y - mFramedImagePoints[frame][i].y;

      double diff = diffX * diffX + diffY * diffY;
      rmsError += diff;
      double dist = sqrt(diff);
      meanError += dist;
      ++numObservations;

      minError = std::min(minError, dist);
      maxError = std::max(maxError, dist);
    }
  }

  meanError /= numObservations;
  rmsError = sqrt(rmsError / numObservations);

  return {rmsError, meanError, minError, maxError};
}

cv::Matx33d Calibration::cameraMatrix() const { return mCameraMatrix; }

cv::Matx<double, 1, 4> Calibration::distortionCoeffs() const { return mDistortionCoeffs; }

}  // namespace assignments::calibration
