#include <tuple>
#include <string>
#include <getopt.h>

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
  cv::Matx<double, 1, 4> fisheye_model(0.1, -0.2, 0.03, 0.001);

  cv::Matx33d camera_matrix(584.4, 0, 622.8,
                            0, 584.4, 538.3,
                            0, 0, 1);

  std::vector<cv::Point3d> object_points;
  std::vector<cv::Point2d> image_points;

  const auto options = process_args(argc, argv);
  std::tie(object_points, image_points) = assignments::load_correspondence(options.input_file);

  std::cout <<
  "******************************************** solvePnP **************************************************************"
            << std::endl;
  cv::Vec3f pnp_rvec, pnp_tvec;
  std::tie(pnp_rvec, pnp_tvec) = assignments::fisheye_solvePnP(object_points, image_points, camera_matrix,
                                                               fisheye_model);

  const auto pnp_reprojection_error = assignments::reprojection_error(object_points, image_points, pnp_rvec, pnp_tvec,
                                                                      camera_matrix, fisheye_model);

  const auto pnp_objects_from_camera = cv::Affine3d(pnp_rvec, pnp_tvec).inv();
  assignments::print_result(pnp_objects_from_camera, pnp_reprojection_error);

  std::cout <<
  "******************************************* solvePnP RANSAC ********************************************************"
            << std::endl;
  int ransac_iters;
  cv::Vec3f ransac_rvec, ransac_tvec;
  std::vector<int> inlier_indices;
  std::vector<int> outlier_indices;
  std::tie(ransac_rvec, ransac_tvec,
           inlier_indices, outlier_indices,
           ransac_iters) = assignments::fisheye_solvePnPRansac(object_points, image_points, camera_matrix,
                                                               fisheye_model);

  std::vector<cv::Point3d> object_ransac_inliers;
  std::vector<cv::Point2d> image_ransac_inliers;
  std::vector<cv::Point3d> object_ransac_outliers;
  std::vector<cv::Point2d> image_ransac_outliers;

  std::tie(object_ransac_inliers, image_ransac_inliers) = assignments::get_liers(object_points, image_points,
                                                                                 inlier_indices);
  std::tie(object_ransac_outliers, image_ransac_outliers) = assignments::get_liers(object_points, image_points,
                                                                                   outlier_indices);
  const auto ransac_reprojection_error = assignments::reprojection_error(object_ransac_inliers, image_ransac_inliers,
                                                                         ransac_rvec, ransac_tvec, camera_matrix,
                                                                         fisheye_model);

  const auto ransac_objects_from_camera = cv::Affine3d(ransac_rvec, ransac_tvec).inv();
  assignments::print_result(ransac_objects_from_camera, ransac_reprojection_error);

  /********************************************* Bundle Adjustment ****************************************************/

  std::cout <<
  "*********************************** Bundle Adjustment with Inliers and Outliers ************************************"
            << std::endl;

  ceres::LossFunction* loss_function = NULL;

  cv::Matx33d camera_matrix_full_no_loss;
  cv::Matx<double, 1, 4> fisheye_model_full_no_loss;
  cv::Vec3d rvec_full_no_loss;
  cv::Vec3d tvec_full_no_loss;

  std::tie(camera_matrix_full_no_loss, fisheye_model_full_no_loss,
           rvec_full_no_loss, tvec_full_no_loss) = assignments::bundle_adjust(object_points, image_points,
                                                                              camera_matrix, fisheye_model,
                                                                              ransac_rvec, ransac_tvec,
                                                                              loss_function,
                                                                              {false, false, true, true},
                                                                              options.verbose);

  const auto BAFNL_reprojection_error = assignments::reprojection_error(object_points, image_points,
                                                                        rvec_full_no_loss, tvec_full_no_loss,
                                                                        camera_matrix_full_no_loss,
                                                                        fisheye_model_full_no_loss);

  const auto BAFNL_objects_from_camera = cv::Affine3d(rvec_full_no_loss, tvec_full_no_loss).inv();
  assignments::print_result(BAFNL_objects_from_camera, BAFNL_reprojection_error);

  std::cout <<
  "*********************************** Bundle Adjustment with Inliers Only ********************************************"
            << std::endl;

  cv::Matx33d camera_matrix_inliers_no_loss;
  cv::Matx<double, 1, 4> fisheye_model_inliers_no_loss;
  cv::Vec3d rvec_inliers_no_loss;
  cv::Vec3d tvec_inliers_no_loss;

  std::tie(camera_matrix_inliers_no_loss, fisheye_model_inliers_no_loss,
           rvec_inliers_no_loss, tvec_inliers_no_loss) = assignments::bundle_adjust(object_ransac_inliers,
                                                                                    image_ransac_inliers,
                                                                                    camera_matrix,
                                                                                    fisheye_model,
                                                                                    ransac_rvec, ransac_tvec,
                                                                                    loss_function,
                                                                                    {false, false, true, true},
                                                                                    options.verbose);

  const auto BAINL_reprojection_error = assignments::reprojection_error(object_ransac_inliers, image_ransac_inliers,
                                                                        rvec_inliers_no_loss, tvec_inliers_no_loss,
                                                                        camera_matrix_inliers_no_loss,
                                                                        fisheye_model_inliers_no_loss);

  const auto BAINL_objects_from_camera = cv::Affine3d(rvec_inliers_no_loss, tvec_inliers_no_loss).inv();
  assignments::print_result(BAINL_objects_from_camera, BAINL_reprojection_error);

  std::cout <<
  "***************************** Bundle Adjustment with Inliers and Outliers and LossFunction *************************"
            << std::endl;

  loss_function = new ceres::HuberLoss(0.1);  // set our loss function

  cv::Matx33d camera_matrix_full_with_loss;
  cv::Matx<double, 1, 4> fisheye_model_full_with_loss;
  cv::Vec3d rvec_full_with_loss;
  cv::Vec3d tvec_full_with_loss;

  std::tie(camera_matrix_full_with_loss, fisheye_model_full_with_loss,
           rvec_full_with_loss, tvec_full_with_loss) = assignments::bundle_adjust(object_points,
                                                                                  image_points,
                                                                                  camera_matrix,
                                                                                  fisheye_model,
                                                                                  ransac_rvec, ransac_tvec,
                                                                                  loss_function,
                                                                                  {false, false, true, true},
                                                                                  options.verbose);

  const auto BAFWL_reprojection_error = assignments::reprojection_error(object_points, image_points,
                                                                        rvec_full_with_loss, tvec_full_with_loss,
                                                                        camera_matrix_full_with_loss,
                                                                        fisheye_model_full_with_loss);

  const auto BAFWL_objects_from_camera = cv::Affine3d(rvec_full_with_loss, tvec_full_with_loss).inv();
  assignments::print_result(BAFWL_objects_from_camera, BAFWL_reprojection_error);

  std::cout <<
  "***************************** Bundle Adjustment with Inliers Only and LossFunction *********************************"
            << std::endl;

  loss_function = new ceres::HuberLoss(0.1);  // reset loss function
  cv::Matx33d camera_matrix_inliers_with_loss;
  cv::Matx<double, 1, 4> fisheye_model_inliers_with_loss;
  cv::Vec3d rvec_inliers_with_loss;
  cv::Vec3d tvec_inliers_with_loss;

  std::tie(camera_matrix_inliers_with_loss, fisheye_model_inliers_with_loss,
           rvec_inliers_with_loss, tvec_inliers_with_loss) = assignments::bundle_adjust(object_ransac_inliers,
                                                                                        image_ransac_inliers,
                                                                                        camera_matrix, fisheye_model,
                                                                                        ransac_rvec, ransac_tvec,
                                                                                        loss_function,
                                                                                        {false, false, true, true},
                                                                                        options.verbose);

  const auto BAIWL_reprojection_error = assignments::reprojection_error(object_ransac_inliers, image_ransac_inliers,
                                                                        rvec_inliers_with_loss, tvec_inliers_with_loss,
                                                                        camera_matrix_inliers_with_loss,
                                                                        fisheye_model_inliers_with_loss);

  const auto BAIWL_objects_from_camera = cv::Affine3d(rvec_inliers_with_loss, tvec_inliers_with_loss).inv();
  assignments::print_result(BAIWL_objects_from_camera, BAIWL_reprojection_error);

  return EXIT_SUCCESS;
}