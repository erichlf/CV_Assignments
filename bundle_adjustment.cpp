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

  const auto pnp_objects_from_camera = cv::Affine3f(pnp_rvec, pnp_tvec).inv();

  std::cout << "Rotation:" << std::endl;
  std::cout << pnp_objects_from_camera.rvec() << std::endl;
  std::cout << "Translation:" << std::endl;
  std::cout << pnp_objects_from_camera.translation() << std::endl;
  std::cout << "Reprojection Error:" << std::endl;
  std::cout << pnp_reprojection_error << std::endl;

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

  const auto ransac_objects_from_camera = cv::Affine3f(ransac_rvec, ransac_tvec).inv();

  std::cout << "Rotation:" << std::endl;
  std::cout << ransac_objects_from_camera.rvec() << std::endl;
  std::cout << "Translation:" << std::endl;
  std::cout << ransac_objects_from_camera.translation() << std::endl;
  std::cout << "Reprojection Error:" << std::endl;
  std::cout << ransac_reprojection_error << std::endl;

  /********************************************* Bundle Adjustment ****************************************************/

  std::cout <<
  "*********************************** Bundle Adjustment with Inliers and Outliers ************************************"
            << std::endl;

  ceres::LossFunction* loss_function = NULL;

  cv::Vec3d rvec_full_no_loss;
  cv::Vec3d tvec_full_no_loss;

  std::tie(rvec_full_no_loss, tvec_full_no_loss) = assignments::bundle_adjust(object_points, image_points,
                                                                              camera_matrix, fisheye_model,
                                                                              ransac_rvec, ransac_tvec,
                                                                              loss_function, options.verbose);

  const auto BAFNL_reprojection_error = assignments::reprojection_error(object_points, image_points,
                                                                        rvec_full_no_loss, tvec_full_no_loss,
                                                                        camera_matrix, fisheye_model);
  const auto BAFNL_objects_from_camera = cv::Affine3f(rvec_full_no_loss, tvec_full_no_loss).inv();

  std::cout << "Rotation:" << std::endl;
  std::cout << BAFNL_objects_from_camera.rvec() << std::endl;
  std::cout << "Translation:" << std::endl;
  std::cout << BAFNL_objects_from_camera.translation() << std::endl;
  std::cout << "Reprojection Error:" << std::endl;
  std::cout << BAFNL_reprojection_error << std::endl;

  std::cout <<
  "*********************************** Bundle Adjustment with Inliers Only ********************************************"
            << std::endl;

  cv::Vec3d rvec_inliers_no_loss;
  cv::Vec3d tvec_inliers_no_loss;

  std::tie(rvec_inliers_no_loss, tvec_inliers_no_loss) = assignments::bundle_adjust(object_ransac_inliers,
                                                                                    image_ransac_inliers,
                                                                                    camera_matrix, fisheye_model,
                                                                                    ransac_rvec, ransac_tvec,
                                                                                    loss_function, options.verbose);

  const auto BAINL_reprojection_error = assignments::reprojection_error(object_ransac_inliers, image_ransac_inliers,
                                                                        rvec_inliers_no_loss, tvec_inliers_no_loss,
                                                                        camera_matrix, fisheye_model);

  const auto BAINL_objects_from_camera = cv::Affine3f(rvec_inliers_no_loss, tvec_inliers_no_loss).inv();

  std::cout << "Rotation:" << std::endl;
  std::cout << BAINL_objects_from_camera.rvec() << std::endl;
  std::cout << "Translation:" << std::endl;
  std::cout << BAINL_objects_from_camera.translation() << std::endl;
  std::cout << "Reprojection Error:" << std::endl;
  std::cout << BAINL_reprojection_error << std::endl;

  std::cout <<
  "***************************** Bundle Adjustment with Inliers and Outliers and LossFunction *************************"
            << std::endl;

  loss_function = new ceres::CauchyLoss(0.01);  // set our loss function

  cv::Vec3d rvec_full_with_loss;
  cv::Vec3d tvec_full_with_loss;

  std::tie(rvec_full_with_loss, tvec_full_with_loss) = assignments::bundle_adjust(object_points, image_points,
                                                                                  camera_matrix, fisheye_model,
                                                                                  ransac_rvec, ransac_tvec,
                                                                                  loss_function, options.verbose);

  const auto BAFWL_reprojection_error = assignments::reprojection_error(object_points, image_points,
                                                                        rvec_full_with_loss, tvec_full_with_loss,
                                                                        camera_matrix, fisheye_model);
  const auto BAFWL_objects_from_camera = cv::Affine3f(rvec_full_with_loss, tvec_full_with_loss).inv();

  std::cout << "Rotation:" << std::endl;
  std::cout << BAFWL_objects_from_camera.rvec() << std::endl;
  std::cout << "Translation:" << std::endl;
  std::cout << BAFWL_objects_from_camera.translation() << std::endl;
  std::cout << "Reprojection Error:" << std::endl;
  std::cout << BAFWL_reprojection_error << std::endl;

  std::cout <<
  "***************************** Bundle Adjustment with Inliers Only and LossFunction *********************************"
            << std::endl;

  loss_function = new ceres::HuberLoss(0.1);  // reset loss function
  cv::Vec3d rvec_inliers_with_loss;
  cv::Vec3d tvec_inliers_with_loss;

  std::tie(rvec_inliers_with_loss, tvec_inliers_with_loss) = assignments::bundle_adjust(object_ransac_inliers,
                                                                                    image_ransac_inliers,
                                                                                    camera_matrix, fisheye_model,
                                                                                    ransac_rvec, ransac_tvec,
                                                                                    loss_function, options.verbose);

  const auto BAIWL_reprojection_error = assignments::reprojection_error(object_ransac_inliers, image_ransac_inliers,
                                                                        rvec_inliers_with_loss, tvec_inliers_with_loss,
                                                                        camera_matrix, fisheye_model);
  const auto BAIWL_objects_from_camera = cv::Affine3f(rvec_inliers_with_loss, tvec_inliers_with_loss).inv();

  std::cout << "Rotation:" << std::endl;
  std::cout << BAIWL_objects_from_camera.rvec() << std::endl;
  std::cout << "Translation:" << std::endl;
  std::cout << BAIWL_objects_from_camera.translation() << std::endl;
  std::cout << "Reprojection Error:" << std::endl;
  std::cout << BAIWL_reprojection_error << std::endl;

  return EXIT_SUCCESS;
}