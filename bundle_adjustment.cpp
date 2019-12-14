#include <ceres/ceres.h>
#include <jsoncpp/json/json.h>

#include <tuple>
#include <string>

#include "utilities.hpp"

std::tuple<std::vector<cv::Point3d>, std::vector<cv::Point2d>> load_correspondence(const std::string& json_file)
{
  std::ifstream ifs(json_file);
  Json::Value correspondences_json;
  ifs >> correspondences_json;

  std::vector<cv::Point3d> world_points;
  std::vector<cv::Point2d> image_points;
  for (const auto& correspondence : correspondences_json["correspondences"])
  {
    world_points.push_back({correspondence["world_point"][0].asDouble(),
                            correspondence["world_point"][1].asDouble(),
                            correspondence["world_point"][2].asDouble()});
    image_points.push_back({correspondence["image_point"][0].asDouble(),
                            correspondence["image_point"][1].asDouble()});
  }

  return {world_points, image_points};
}

int main(int argc, char** argv)
{
  cv::Matx<double, 1, 4> fisheye_model(0.1, -0.2, 0.03, 0.001);

  cv::Matx33d camera_matrix(584.4, 0, 622.8,
                            0, 584.4, 538.3,
                            0, 0, 1);

  std::vector<cv::Point3d> object_points;
  std::vector<cv::Point2d> image_points;

  std::tie(object_points, image_points) = load_correspondence(argv[1]);

  /******************************************** solvePnP **************************************************************/
  cv::Vec3f pnp_rvec, pnp_tvec;
  std::tie(pnp_rvec, pnp_tvec) = assignments::fisheye_solvePnP(object_points, image_points, camera_matrix,
                                                               fisheye_model);

  const auto pnp_reprojection_error = assignments::reprojection_error(object_points, image_points, pnp_rvec, pnp_tvec,
                                                                      camera_matrix, fisheye_model);

  const auto pnp_objects_from_camera = cv::Affine3f(pnp_rvec, pnp_tvec).inv();

  /******************************************* solvePnP RANSAC ********************************************************/
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

  std::cout << "Reprojection Errors:" << std::endl;
  std::cout << "           solvePnP: " << pnp_reprojection_error << std::endl;
  std::cout << "     solvePnPRansac: " << ransac_reprojection_error << std::endl;

  return EXIT_SUCCESS;
}