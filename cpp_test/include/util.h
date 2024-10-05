//
// Created by yuan on 9/20/24.
//

#ifndef UTIL_H
#define UTIL_H


#include <opencv2/opencv.hpp>


#define PI 3.1415926535897932384626433832795



cv::Vec3f cross(cv::Vec3f a, cv::Vec3f b) {
    return cv::Vec3f(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    );
}

/**
 * Jacobian from cubemap to sphere
 * @param v must be a point on the unit cube()
 * @return
 */
float jacobian(cv::Point3f v) {
    float tmp = (v.x * v.x) + (v.y * v.y) + (v.z * v.z);
    return 1 / (tmp * sqrt(tmp));
}

float jacobian(cv::Vec3f v) {
    float tmp = (v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]);
    return 1 / (tmp * sqrt(tmp));
}




#endif //UTIL_H
