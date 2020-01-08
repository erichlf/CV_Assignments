#include <tuple>
#include <string>
#include <getopt.h>

#include "Calibration.hpp"
#include "utilities.hpp"

struct Args
{
  std::string input_file;
  bool verbose;
};

Args process_args(int argc, char** argv)
{
  int opt{0};
  Args args{"", false};

  auto usage = [argv]()
  {
    printf("Usage: %s [OPTION...]\n", argv[0]);
    printf("Examples:\n");
    printf("  bundle_adjustment -i correspondences.json -v\n");
    printf("Options:\n");
    printf("   -i, --input_file   json file containing the correspondences\n");
    printf("   -v, --vertbose     print bundle adjustment iteration information\n");
    exit(EXIT_SUCCESS);
  };

  static struct option options[] = {
      {"input_file", required_argument, nullptr, 'i'},
      {"verbose", optional_argument, nullptr, 'v'}
  };

  int option_index{0};
  while ((opt = getopt_long(argc, argv, "i:v", options, &option_index)) != -1)
  {
    switch (opt)
    {
      case 'i':
        args.input_file = std::string(optarg);  // convert from char* to string
        break;
      case 'v':
        args.verbose = true;  // convert from char* to string
        break;
      default:
        usage();
        break;
    }
  }

  if (args.input_file.empty())
  {
    usage();
  }

  return args;
}

int main(int argc, char** argv)
{
  const auto options = process_args(argc, argv);

  std::vector<std::vector<cv::Point3d>> framedGridPoints;
  std::vector<std::vector<cv::Point2d>> framedImagePoints;
  cv::Size imageSize(1920, 1080);

  std::tie(framedGridPoints, framedImagePoints) = assignments::calibration::load_correspondence(options.input_file);

  assignments::calibration::Calibration calibration(imageSize, framedGridPoints, framedImagePoints, options.verbose);
  calibration.calibrate();

  const auto cameraMatrix = calibration.cameraMatrix();
  const auto distortionCoeffs = calibration.distortionCoeffs();
  const auto& [RMSE, meanError, minError, maxError] = calibration.errors();

  std::cout << "Intrinsics:" << std::endl << cameraMatrix << std::endl;
  std::cout << "Distortion Coeffs: " << distortionCoeffs << std::endl;
  std::cout << "Root Mean Squared Error: " << RMSE << std::endl;
  std::cout << "Mean Error: " << meanError << std::endl;
  std::cout << "Minimum Error: " << minError << std::endl;
  std::cout << "Maximum Error: " << maxError << std::endl;

  // if (options.verbose)
  // {
  //   size_t numImages = framedImagePoints.size();
  //   size_t numSubImage = 35;
  //   size_t stride = numImages / numSubImage; // This code is extremely slow to run, when running on the 681300 points
  //   cv::Matx33d cvCameraMatrix;
  //   cv::Mat cvDistortionCoeffs = cv::Mat::zeros(1, 4, CV_64F);  // temp place holder so that calibrate can be used
  //   std::vector<cv::Vec3d> cvRVecs;
  //   std::vector<cv::Vec3d> cvTVecs;

  //   std::vector<std::vector<cv::Point3d>> subGrid;
  //   std::vector<std::vector<cv::Point2d>> subImage;
  //   for (size_t i = 0; i < numImages; i += stride)
  //   {
  //     subGrid.push_back(framedGridPoints[i]);
  //     subImage.push_back(framedImagePoints[i]);
  //   }
  //   cv::fisheye::calibrate(subGrid, subImage, imageSize, cvCameraMatrix, cvDistortionCoeffs, cvRVecs, cvTVecs,
  //                          cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | cv::fisheye::CALIB_FIX_SKEW);
  //   std::cout << "OpenCV Estimates:" << std::endl;
  //   std::cout << "Intrinsics:" << std::endl << cvCameraMatrix << std::endl;
  //   std::cout << "Distortion Coeffs: " << cvDistortionCoeffs << std::endl;
  // }

  return EXIT_SUCCESS;
}