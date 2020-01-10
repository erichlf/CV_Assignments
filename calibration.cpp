#include <tuple>
#include <string>
#include <getopt.h>

#include "Calibration.hpp"
#include "utilities.hpp"
#include "types.hpp"

struct Args
{
  std::vector<std::string> input_files;
  bool verbose;
};

Args process_args(int argc, char** argv)
{
  int opt{0};
  Args args{std::vector<std::string>(), false};

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
        args.input_files.push_back(optarg);  // convert from char* to string
        break;
      case 'v':
        args.verbose = true;  // convert from char* to string
        break;
      default:
        usage();
        break;
    }
  }

  for (; optind < argc; ++optind)
    args.input_files.push_back(argv[optind]);

  if (!args.input_files.size())
  {
    usage();
  }

  return args;
}

int main(int argc, char** argv)
{
  const auto options = process_args(argc, argv);
  using assignments::Vector3;
  using assignments::Vector2;

  const auto input_files = options.input_files;
  int numCameras = input_files.size();
  std::vector<assignments::Detections> detections(numCameras);
  cv::Size imageSize(1920, 1080);

  for (int camera = 0; camera < numCameras; ++camera)
  {
    detections[camera] = assignments::calibration::load_correspondence(options.input_files[camera]);
  }

    assignments::calibration::Calibration calibration(imageSize, detections, options.verbose);
    calibration.calibrate();

  for (int camera = 0; camera < numCameras; ++camera)
  {
    std::cout << "************************************** Camera "<< camera
              << " *********************************************" << std::endl;
    std::cout << calibration.intrinsics(camera) << std::endl;
    if (numCameras > 1)
      std::cout << calibration.transform(camera) << std::endl;
  }

  const auto& [RMSE, meanError, minError, maxError] = calibration.errors();

  std::cout << "Root Mean Squared Error: " << RMSE << std::endl;
  std::cout << "Mean Error: " << meanError << std::endl;
  std::cout << "Minimum Error: " << minError << std::endl;
  std::cout << "Maximum Error: " << maxError << std::endl;

  return EXIT_SUCCESS;
}