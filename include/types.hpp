#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <array>

#pragma once

namespace assignments
{

enum CameraIntrinsics
{
  fx = 0,
  fy = 1,
  cx = 2,
  cy = 3
};

enum DistCoeffs
{
  k1 = 0,
  k2 = 1,
  k3 = 2,
  k4 = 3
};

enum Coords
{
  x = 0,
  y = 1,
  z = 2
};

template <typename T> using Vector4 = std::array<T, 4>;
template <typename T> using Vector3 = std::array<T, 3>;
template <typename T> using Vector2 = std::array<T, 2>;

template <typename T>
class Transform
{
 public:
  Transform() = default;

  Transform(const T angleAxisX, const T angleAxisY, const T angleAxisZ,
            const T x, const T y, const T z) :
      mAngleAxisX(angleAxisX), mAngleAxisY(angleAxisY), mAngleAxisZ(angleAxisZ),
      mTranslationX(x), mTranslationY(y), mTranslationZ(z)
  { }

  Transform(const T* angleAxis, const T* translation) :
      mAngleAxisX(angleAxis[Coords::x]), mAngleAxisY(angleAxis[Coords::y]), mAngleAxisZ(angleAxis[Coords::z]),
      mTranslationX(translation[Coords::x]), mTranslationY(translation[Coords::y]), mTranslationZ(translation[Coords::z])
  { }

  Transform(const cv::Vec<T, 3>& angleAxis, const cv::Vec<T, 3>& translation) :
      mAngleAxisX(angleAxis[Coords::x]), mAngleAxisY(angleAxis[Coords::y]), mAngleAxisZ(angleAxis[Coords::z]),
      mTranslationX(translation[Coords::x]), mTranslationY(translation[Coords::y]), mTranslationZ(translation[Coords::z])
  { }

  auto transform(const T* point) const noexcept
  {
    auto result = Vector3<T>{};
    T angleAxis[] = {mAngleAxisX, mAngleAxisY, mAngleAxisZ};
    ceres::AngleAxisRotatePoint(angleAxis, point, result.data());

    result[Coords::x] += mTranslationX;
    result[Coords::y] += mTranslationY;
    result[Coords::z] += mTranslationZ;

    return result;
  }

  auto invert() const noexcept
  {
    // use ceres::AngleAxisRotationPoint to create an inverse
    auto invertedTransform = Transform<T>{0, 0, 0,
                                          -mAngleAxisX, -mAngleAxisY, -mAngleAxisZ};

    auto rotatedTranslation = invertedTransform.transform({mTranslationX, mTranslationY, mTranslationZ});

    invertedTransform.mX = -rotatedTranslation[Coords::x];
    invertedTransform.mY = -rotatedTranslation[Coords::y];
    invertedTransform.mZ = -rotatedTranslation[Coords::z];

    return invertedTransform;
  }

  Vector3<T> t() const noexcept
  {
    return {mTranslationX, mTranslationY, mTranslationZ};
  }

  Vector3<T> R() const noexcept
  {
    return {mAngleAxisX, mAngleAxisY, mAngleAxisZ};
  }

  std::ostream& operator<<(std::ostream& out)
  {
    out << "Rotation: " << R() << std::endl;
    out << "Translation: " << t();

    return out;
  }

 private:
  T mAngleAxisX, mAngleAxisY, mAngleAxisZ;
  T mTranslationX, mTranslationY, mTranslationZ;
};

template <typename T>
class Intrinsics
{
 public:
  Intrinsics() = default;

  Intrinsics(const T fx, const T fy, const T cx, const T cy, const T k1, const T k2, const T k3, const T k4) :
    mFx(fx), mFy(fy), mCx(cx), mCy(cy),
    mK1(k1), mK2(k2), mK3(k3), mK4(k4) { }

  Intrinsics(const T* const cameraMatrix, const T* const distortionCoeffs) :
      mFx(cameraMatrix[CameraIntrinsics::fx]), mFy(cameraMatrix[CameraIntrinsics::fy]),
      mCx(cameraMatrix[CameraIntrinsics::cx]), mCy(cameraMatrix[CameraIntrinsics::cy]),
      mK1(distortionCoeffs[DistCoeffs::k1]), mK2(distortionCoeffs[DistCoeffs::k2]),
      mK3(distortionCoeffs[DistCoeffs::k3]), mK4(distortionCoeffs[DistCoeffs::k4]) { }

  Intrinsics(const cv::Mat& cameraMatrix, const cv::Mat& distortionCoeffs) :
    mFx(cameraMatrix.at<T>(0, 0)), mFy(cameraMatrix.at<T>(1, 1)),
    mCx(cameraMatrix.at<T>(0, 2)), mCy(cameraMatrix.at<T>(1, 2)),
    mK1(distortionCoeffs.at<T>(DistCoeffs::k1)), mK2(distortionCoeffs.at<T>(DistCoeffs::k2)),
    mK3(distortionCoeffs.at<T>(DistCoeffs::k3)), mK4(distortionCoeffs.at<T>(DistCoeffs::k4)) { }

  Vector2<T> projectPoint(const T* point) const
  {
    const auto xp = point[Coords::x] / point[Coords::z];
    const auto yp = point[Coords::y] / point[Coords::z];

    const auto r = sqrt(xp * xp + yp * yp);
    const auto theta = atan(r);

    const T theta_r = (r != static_cast<T>(0)) ? theta * (static_cast<T>(1) + mK1 * theta * theta
                                                          + mK2 * pow(theta, 4)
                                                          + mK3 * pow(theta, 6)
                                                          + mK4 * pow(theta, 8)) / r
                                               : static_cast<T>(0);

    const T xpp = theta_r * xp;
    const T ypp = theta_r * yp;

    const T u = xpp * mFx + mCx;
    const T v = ypp * mFy + mCy;

    return {u, v};
  }

  Vector4<T> intrinsics() const noexcept
  {
    return {mFx, mFy, mCx, mCy};
  }

  Vector4<T> distCoeffs() const noexcept
  {
    return {mK1, mK2, mK3, mK4};
  }

 private:
  T mFx;
  T mFy;
  T mCx;
  T mCy;
  T mK1;
  T mK2;
  T mK3;
  T mK4;
};

template <typename T>
std::ostream& operator<<(std::ostream& out, Intrinsics<T> intrinsics)
{
  out << "Intrinsics: " << intrinsics.intrinsics() << std::endl;
  out << "Distortion Coeffs: " << intrinsics.distCoeffs();

  return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const Vector2<T> v)
{
  out << "(" << v[0] << ", " << v[1] << ")";

  return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const Vector3<T> v)
{
  out << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";

  return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const Vector4<T> v)
{
  out << "(" << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << ")";

  return out;
}

struct Detections
{
  std::vector<std::vector<Vector3<double>>> gridPoints;
  std::vector<std::vector<Vector2<double>>> imagePoints;
  std::unordered_map<std::string, size_t> frames;
  int maxFrame;
};

}  // namespace assignments
