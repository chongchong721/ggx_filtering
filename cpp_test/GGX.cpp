//
// Created by yuan on 9/20/24.
//

#include "GGX.h"

#include "util.h"


float GGX_gloss_to_alpha(float gloss) {
    const float GGX_MAX_SPEC_POWER = 18;
    float exponent = powf( 2.0f, gloss * GGX_MAX_SPEC_POWER );
    // returns roughness == sqrt(alpha) for GGX physically-based shader
    return powf( 2.0f / ( 1 + exponent ), .25f );
}

GGX::GGX(float alpha, cv::Vec3f normal) {
    this->alpha = alpha;
    this->normal_dir = normal;
    this->alpha2 = alpha * alpha;
}

float GGX::ndf_relative(cv::Vec3f world_dir) {
    float cos = world_dir.dot(this->normal_dir);
    return this->ndf_cos(cos);
}

/**
 *
 * @param local_dir local_dir must be normalized
 * @return the ndf
 */
float GGX::ndf_absolute(cv::Vec3f local_dir) {
    float cos = local_dir[2];
    return this->ndf_cos(cos);
}


float GGX::ndf_cos(float cosine) {
    if(cosine < 0) {
        return 0;
    }
    else {
        float tmp = cosine * cosine * (this->alpha2 - 1) + 1;
        float ndf = this->alpha2 / (PI * tmp * tmp);
        return ndf;
    }
}



