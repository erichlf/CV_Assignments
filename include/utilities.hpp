#include <opencv2/opencv.hpp>

#include <vector>

namespace assignments
{

void project_points(const cv::Mat& world_points, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& K,
                    const cv::Mat& dist_coeffs, std::vector<cv::Point2d>& image_points)
{
  double k1 = 0., k2 = 0., k3 = 0., p1 = 0., p2 = 0., k4 = 0., k5 = 0., k6 = 0.;
  if (dist_coeffs.cols >= 4)  // fisheye model
  {
    k1 = dist_coeffs.at<double>(0, 0);
    k2 = dist_coeffs.at<double>(0, 1);
    k3 = dist_coeffs.at<double>(0, 2);
    k4 = dist_coeffs.at<double>(0, 3);
  }

  if (dist_coeffs.cols >= 6)  // plumb bob model
  {
    p1 = dist_coeffs.at<double>(0, 2);
    p2 = dist_coeffs.at<double>(0, 3);
    k3 = dist_coeffs.at<double>(0, 4);
    k4 = dist_coeffs.at<double>(0, 5);
  }

  if (dist_coeffs.cols == 8) // rational polynomial
  {
    k5 = dist_coeffs.at<double>(0, 6);
    k6 = dist_coeffs.at<double>(0, 7);
  }

  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  for (int i = 0; i < world_points.rows; ++i)
  {
    double x = world_points.at<double>(i, 0);
    double y = world_points.at<double>(i, 1);
    double z = world_points.at<double>(i, 2);

    double xp = x / z;
    double yp = y / z;

    double r = sqrt(xp * xp + yp * yp);
    double theta;

    if (dist_coeffs.cols == 4)
    {
      theta = atan(r);
      theta = r != 0 ?
              theta * (1 + k1 * theta * theta + k2 * pow(theta, 4) + k3 * pow(theta, 6) + k4 * pow(theta, 8)) / r :
              0;
    }
    else
    {
      theta = (1. + k1 * r * r + k2 * pow(r, 4) + k3 * pow(r, 6)) /
              (1. + k4 * r * r + k5 * pow(r, 4) + k6 * pow(r, 6));
    }
    double dx1 = 2. * p1 * xp * yp + p2 * (r * r + 2. * x * x);  // zero for fisheye
    double dx2 = p1 * (r * r + 2. * y * y) + 2. * p2 * xp * yp;  // zero for fisheye


    double xpp = theta * xp + dx1;
    double ypp = theta * yp + dx2;

    double u = fx * xpp + cx;
    double v = fy * ypp + cy;

    image_points.push_back({u, v});
  }
}

};  // namespace assignements
