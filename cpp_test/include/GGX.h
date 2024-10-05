//
// Created by yuan on 9/20/24.
//

#ifndef GGX_H
#define GGX_H

#include <opencv2/opencv.hpp>


class GGX {
    float alpha;
    float alpha2;
    cv::Vec3f normal_dir;
public:
    explicit GGX(float alpha, cv::Vec3f normal);
    float sampleGGX(cv::Vec2f uv);
    float ndf_relative(cv::Vec3f world_dir);
    float ndf_absolute(cv::Vec3f local_dir);
private:
    float ndf_cos(float cosine);
};


float GGX_gloss_to_alpha(float gloss);



#endif //GGX_H
