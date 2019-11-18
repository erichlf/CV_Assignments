#include "utilities.hpp"

int main()
{
  cv::Matx<double, 1, 4> fisheye_model(0.1, -0.2, 0.03, 0.001);

  cv::Matx33d camera_matrix(584.4, 0, 622.8,
                            0, 584.4, 538.3,
                            0, 0, 1);

  cv::Matx<double, 4, 3> object_points(0., 0., 0.,
                                       2., 0., 0.,
                                       2., 2., 0.,
                                       0., 2., 0.);

  std::vector<cv::Point2d> image_points{{752.62227857, 777.97189891},
                                        {933.85321358, 793.21655267},
                                        {871.88605416, 895.78261428},
                                        {720.14648139, 886.22059601}};

  cv::Vec3d rvec;
  cv::Vec3d tvec;

  cv::solvePnP(object_points, image_points, camera_matrix, fisheye_model, rvec, tvec, false, CV_P3P);

  const auto reprojection_error = assignments::reprojection_error(object_points, image_points, rvec, tvec,
                                                                  camera_matrix, fisheye_model);

  cv::Affine3f objects_from_camera = cv::Affine3f(rvec, tvec).inv();

  std::cout << "Rotation:" << std::endl;
  std::cout << objects_from_camera.rvec() << std::endl;
  std::cout << "Translation:" << std::endl;
  std::cout << objects_from_camera.translation() << std::endl;
  std::cout << "Reprojection Error:" << std::endl;
  std::cout << reprojection_error << std::endl;
}
