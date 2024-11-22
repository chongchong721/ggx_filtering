"""
The second pass filtering
There are in total 7 levels of mipmap: 2^7 -> 2^1
"""
from cmath import polar
from datetime import datetime
from random import sample

import numpy as np
import map_util
from map_util import gen_frame_weight,gen_frame_xyz,gen_theta_phi,gen_anisotropic_frame_xyz, gen_theta_phi_no_frame
import mat_util
import coefficient
import interpolation
import image_read
from tqdm import tqdm

import reference
from torch_util import QuadModel, ConstantModel, create_view_model_dict, random_dir_sphere, random_dir_hemisphere,torch_normalized,texel_dir_128_torch,texel_dir_128_torch_map,get_reflected_vector_torch_vectorized, sample_view_dependent_location
import os
import torch


def gen_tex_levels(n_level,n_high_res):
    texture_list = []
    cur_res = n_high_res
    for i in range(n_level):
        tex = np.zeros((6,cur_res,cur_res,3))
        texture_list.append(tex)
        cur_res = n_high_res / 2




def fetch_sample_view_dependent_python_table(tex_input,output_level, coeff_table, n_sample_per_frame ,follow_code = False, constant = True, j_adjust = True, allow_neg_weight = False, view_option_str="None",view_direction = np.array([1.0,0.0,0.0]), reflection_parameterization = False, clip_below_horizon = False):
    coefficient_table = coeff_table
    interpolator = interpolation.trilinear_mipmap_interpolator(tex_input)

    n_res = 128 >> output_level
    faces_xyz = map_util.texel_directions(n_res)
    faces_xyz_normalized = faces_xyz / np.linalg.norm(faces_xyz, axis=-1, keepdims=True)
    view_direction = view_direction / np.linalg.norm(view_direction, axis=-1, keepdims=True)
    cosine_view = np.dot(faces_xyz_normalized,view_direction)

    #mask_above_horizon = cosine_view > 0.0

    view_theta = np.arccos(cosine_view)
    view_theta2 = view_theta ** 2

    color = np.zeros((6, n_res, n_res, 3))
    weight = np.zeros((6, n_res, n_res))

    if view_option_str[:14] == "relative_frame":
        frame_count = 1
    else:
        frame_count = 3

    reflected_direction = map_util.get_reflected_vector_vectorized(faces_xyz_normalized,view_direction)


    for frame_idx in range(frame_count):


        if view_option_str[:14] == "relative_frame":
            X,Y,Z = gen_anisotropic_frame_xyz(faces_xyz_normalized,view_direction)
            theta_normal,phi_normal,theta_normal2,phi_normal2 = gen_theta_phi_no_frame(faces_xyz)
            theta_reflection,phi_reflection,theta_reflection2,phi_reflection2 = gen_theta_phi_no_frame(reflected_direction)
            frame_weight = np.ones(faces_xyz.shape[:-1])
        else:
            X, Y, Z = gen_frame_xyz(faces_xyz, frame_idx)
            theta_normal, phi_normal, theta_normal2, phi_normal2 = gen_theta_phi(faces_xyz, frame_idx=frame_idx, follow_code=follow_code)
            theta_normal, phi_reflection, theta_reflection2, phi_reflection2 = gen_theta_phi(reflected_direction, frame_idx=frame_idx,
                                                                                 follow_code=follow_code)
            frame_weight = gen_frame_weight(faces_xyz, frame_idx, follow_code)
        coeff_start = frame_idx * n_sample_per_frame
        coeff_end = coeff_start + n_sample_per_frame

        if reflection_parameterization:
            theta2 = theta_reflection2
            phi2 = phi_reflection2
        else:
            theta2 = theta_normal2
            phi2 = phi_normal2


        for sample_idx in range(n_sample_per_frame):

            coeff_x_table = coefficient_table[0, :, coeff_start + sample_idx]
            coeff_y_table = coefficient_table[1, :, coeff_start + sample_idx]
            coeff_z_table = coefficient_table[2, :, coeff_start + sample_idx]
            coeff_level_table = coefficient_table[3, :, coeff_start + sample_idx]
            coeff_weight_table = coefficient_table[4, :, coeff_start + sample_idx]

            if view_option_str == "even_only":
                coeff_x = coeff_x_table[0] + coeff_x_table[1] * theta2 + coeff_x_table[2] * phi2 + coeff_x_table[3] * view_theta2 + coeff_x_table[4] * view_theta
                coeff_y = coeff_y_table[0] + coeff_y_table[1] * theta2 + coeff_y_table[2] * phi2 + coeff_y_table[3] * view_theta2 + coeff_y_table[4] * view_theta
                coeff_z = coeff_z_table[0] + coeff_z_table[1] * theta2 + coeff_z_table[2] * phi2 + coeff_z_table[3] * view_theta2 + coeff_z_table[4] * view_theta
                sample_level = coeff_level_table[0] + coeff_level_table[1] * theta2 + coeff_level_table[2] * phi2 + coeff_level_table[3] * view_theta2 + coeff_level_table[4] * view_theta
                sample_weight = coeff_weight_table[0] + coeff_weight_table[1] * theta2 + coeff_weight_table[2] * phi2 + coeff_weight_table[3] * view_theta2 + coeff_weight_table[4] * view_theta
            elif view_option_str == "view_only":
                coeff_x = coeff_x_table[0] + coeff_x_table[1] * view_theta2 + coeff_x_table[2] * view_theta
                coeff_y = coeff_y_table[0] + coeff_y_table[1] * view_theta2 + coeff_y_table[2] * view_theta
                coeff_z = coeff_z_table[0] + coeff_z_table[1] * view_theta2 + coeff_z_table[2] * view_theta
                sample_level = coeff_level_table[0] + coeff_level_table[1] * view_theta2 + coeff_level_table[2] * view_theta
                sample_weight = coeff_weight_table[0] + coeff_weight_table[1] * view_theta2 + coeff_weight_table[2] * view_theta
            elif view_option_str == "odd":
                coeff_x = coeff_x_table[0] + coeff_x_table[1] * theta2 + coeff_x_table[2] * phi2 + coeff_x_table[
                    3] * view_theta2 + coeff_x_table[4] * view_theta + coeff_x_table[5] * view_theta * theta2 + \
                          coeff_x_table[6] * view_theta * phi2
                coeff_y = coeff_y_table[0] + coeff_y_table[1] * theta2 + coeff_y_table[2] * phi2 + coeff_y_table[
                    3] * view_theta2 + coeff_y_table[4] * view_theta + coeff_y_table[5] * view_theta * theta2 + \
                          coeff_y_table[6] * view_theta * phi2
                coeff_z = coeff_z_table[0] + coeff_z_table[1] * theta2 + coeff_z_table[2] * phi2 + coeff_z_table[
                    3] * view_theta2 + coeff_z_table[4] * view_theta + coeff_z_table[5] * view_theta * theta2 + \
                          coeff_z_table[6] * view_theta * phi2
                sample_level = coeff_level_table[0] + coeff_level_table[1] * theta2 + coeff_level_table[2] * phi2 + \
                        coeff_level_table[3] * view_theta2 + coeff_level_table[4] * view_theta + coeff_level_table[
                            5] * view_theta * theta2 + coeff_level_table[6] * view_theta * phi2
                sample_weight = coeff_weight_table[0] + coeff_weight_table[1] * theta2 + coeff_weight_table[2] * phi2 + \
                         coeff_weight_table[3] * view_theta2 + coeff_weight_table[4] * view_theta + coeff_weight_table[
                             5] * view_theta * theta2 + coeff_weight_table[6] * view_theta * phi2
            elif view_option_str == "relative_frame":
                coeff_x = coeff_x_table[0] + coeff_x_table[1] * theta2 + coeff_x_table[2] * phi2 + coeff_x_table[
                    3] * view_theta2 + coeff_x_table[4] * view_theta
                coeff_y = coeff_y_table[0] + coeff_y_table[1] * theta2 + coeff_y_table[2] * phi2 + coeff_y_table[
                    3] * view_theta2 + coeff_y_table[4] * view_theta
                coeff_z = coeff_z_table[0] + coeff_z_table[1] * theta2 + coeff_z_table[2] * phi2 + coeff_z_table[
                    3] * view_theta2 + coeff_z_table[4] * view_theta
                sample_level = coeff_level_table[0] + coeff_level_table[1] * theta2 + coeff_level_table[2] * phi2 + \
                               coeff_level_table[3] * view_theta2 + coeff_level_table[4] * view_theta
                sample_weight = coeff_weight_table[0] + coeff_weight_table[1] * theta2 + coeff_weight_table[2] * phi2 + \
                                coeff_weight_table[3] * view_theta2 + coeff_weight_table[4] * view_theta
            elif view_option_str == "relative_frame_full":
                coeff_x = (coeff_x_table[0] + coeff_x_table[1] * theta_normal2 + coeff_x_table[2] * phi_normal2 +
                           coeff_x_table[3] * theta_reflection2 + coeff_x_table[4] + phi_reflection2 + coeff_x_table[
                    5] * view_theta2 + coeff_x_table[6] * view_theta)
                coeff_y = (coeff_y_table[0] + coeff_y_table[1] * theta_normal2 + coeff_y_table[2] * phi_normal2 +
                           coeff_y_table[3] * theta_reflection2 + coeff_y_table[4] + phi_reflection2 + coeff_y_table[
                    5] * view_theta2 + coeff_y_table[6] * view_theta)
                coeff_z = (coeff_z_table[0] + coeff_z_table[1] * theta_normal2 + coeff_z_table[2] * phi_normal2 +
                           coeff_z_table[3] * theta_reflection2 + coeff_z_table[4] + phi_reflection2 + coeff_z_table[
                    5] * view_theta2 + coeff_z_table[6] * view_theta)
                sample_level = (coeff_level_table[0] + coeff_level_table[1] * theta_normal2 + coeff_level_table[2] * phi_normal2+
                               coeff_level_table[3] * theta_reflection2 + coeff_level_table[4] + phi_reflection2
                                +coeff_level_table[5] * view_theta2 + coeff_level_table[6] * view_theta)
                sample_weight = (coeff_weight_table[0] + coeff_weight_table[1] * theta_normal2 + coeff_weight_table[2] * phi_normal2 +
                                coeff_weight_table[3] * theta_reflection2 + coeff_weight_table[4] + phi_reflection2 +
                                coeff_weight_table[5] * view_theta2 + coeff_weight_table[6] * view_theta)
            elif view_option_str == "relative_frame_full_interaction":
                coeff_x = (coeff_x_table[0] + coeff_x_table[1] * theta_normal2 + coeff_x_table[2] * phi_normal2 +
                           coeff_x_table[3] * theta_reflection2 + coeff_x_table[4] + phi_reflection2 + coeff_x_table[
                               5] * view_theta2 + coeff_x_table[6] * view_theta + coeff_x_table[
                               7] * theta_normal2 * view_theta + coeff_x_table[8] * phi_normal2 * view_theta +
                           coeff_x_table[9] * theta_reflection2 * view_theta + coeff_x_table[
                               10] * phi_reflection2 * view_theta)
                coeff_y = (coeff_y_table[0] + coeff_y_table[1] * theta_normal2 + coeff_y_table[2] * phi_normal2 +
                           coeff_y_table[3] * theta_reflection2 + coeff_y_table[4] + phi_reflection2 + coeff_y_table[
                               5] * view_theta2 + coeff_y_table[6] * view_theta + coeff_y_table[
                               7] * theta_normal2 * view_theta + coeff_y_table[8] * phi_normal2 * view_theta +
                           coeff_y_table[9] * theta_reflection2 * view_theta + coeff_y_table[
                               10] * phi_reflection2 * view_theta)
                coeff_z = (coeff_z_table[0] + coeff_z_table[1] * theta_normal2 + coeff_z_table[2] * phi_normal2 +
                           coeff_z_table[3] * theta_reflection2 + coeff_z_table[4] + phi_reflection2 + coeff_z_table[
                               5] * view_theta2 + coeff_z_table[6] * view_theta + coeff_z_table[
                               7] * theta_normal2 * view_theta + coeff_z_table[8] * phi_normal2 * view_theta +
                           coeff_z_table[9] * theta_reflection2 * view_theta + coeff_z_table[
                               10] * phi_reflection2 * view_theta)
                sample_level = (coeff_level_table[0] + coeff_level_table[1] * theta_normal2 + coeff_level_table[
                    2] * phi_normal2 + coeff_level_table[3] * theta_reflection2 + coeff_level_table[
                                    4] + phi_reflection2 + coeff_level_table[5] * view_theta2 + coeff_level_table[
                                    6] * view_theta + coeff_level_table[7] * theta_normal2 * view_theta +
                                coeff_level_table[8] * phi_normal2 * view_theta + coeff_level_table[
                                    9] * theta_reflection2 * view_theta + coeff_level_table[
                                    10] * phi_reflection2 * view_theta)
                sample_weight = (coeff_weight_table[0] + coeff_weight_table[1] * theta_normal2 + coeff_weight_table[
                    2] * phi_normal2 + coeff_weight_table[3] * theta_reflection2 + coeff_weight_table[
                                     4] + phi_reflection2 + coeff_weight_table[5] * view_theta2 + coeff_weight_table[
                                     6] * view_theta + coeff_weight_table[7] * theta_normal2 * view_theta +
                                 coeff_weight_table[8] * phi_normal2 * view_theta + coeff_weight_table[
                                     9] * theta_reflection2 * view_theta + coeff_weight_table[
                                     10] * phi_reflection2 * view_theta)
            else:
                raise NotImplementedError

            # min_weight = sample_weight.min()

            coeff_x = np.stack((coeff_x, coeff_x, coeff_x), axis=-1)
            coeff_y = np.stack((coeff_y, coeff_y, coeff_y), axis=-1)
            coeff_z = np.stack((coeff_z, coeff_z, coeff_z), axis=-1)


            if not allow_neg_weight:
                sample_weight = np.clip(sample_weight, 0, a_max=None)
            sample_weight = sample_weight * frame_weight

            sample_direction = coeff_x * X + coeff_y * Y + coeff_z * Z
            abs_direction = np.abs(sample_direction)
            max_dir = np.max(abs_direction, axis=-1)
            sample_direction_map = sample_direction / np.stack([max_dir, max_dir, max_dir], axis=-1)





            if j_adjust:
                # adjust level
                j = 3 / 4 * np.log2(map_util.dot_vectorized_4D(sample_direction_map, sample_direction_map))
                sample_level += j

            sample_level = np.clip(sample_level, 0, 6)

            color_tmp = interpolator.interpolate_all(sample_direction_map, sample_level)
            #color_tmp = np.where(cosine_view > 0.0, color_tmp, 0.0)


            #If this direction is below the hemisphere, we should not count it
            #By setting the weight to zero, we also make the color zero(because we have color * weight)
            sample_direction_normalized = sample_direction / np.linalg.norm(sample_direction, axis=-1,keepdims=True)
            NdotL = np.sum(sample_direction_normalized * faces_xyz_normalized, axis=-1)
            if clip_below_horizon:
                sample_above_horizon = (NdotL > 0.0)
                sample_weight = np.where(sample_above_horizon, sample_weight, 0.0)


            color += color_tmp * np.stack((sample_weight, sample_weight, sample_weight), axis=-1)
            weight += sample_weight

    # devide by weight
    weight_stack = np.stack((weight, weight, weight), axis=-1)
    color = color / weight_stack


    #This is clipping the part where view direction is below the hemisphere(
    # This is different from not counting the part where the sample direction is below the hemisphere)
    cosine_view = np.stack((cosine_view, cosine_view, cosine_view), axis=-1)
    color_final = np.where(cosine_view > 0.0, color, 0.0)

    return color_final




def fetch_samples_python_table(tex_input,output_level, coeff_table, n_sample_per_frame ,follow_code = False, constant = True, j_adjust = True, allow_neg_weight = False):
    """
    Use the table generated by our python code
    :param tex_input:
    :param output_level:
    :param coeff_table:
    :param follow_code:
    :param constant: whether this is a constant table
    :param j_adjust: whether to adjust Jacobian
    :return:
    """
    coefficient_table = coeff_table
    interpolator = interpolation.trilinear_mipmap_interpolator(tex_input)



    n_res = 128 >> output_level
    faces_xyz = map_util.texel_directions(n_res)

    color = np.zeros((6,n_res,n_res,3))
    weight = np.zeros((6,n_res,n_res))

    for frame_idx in range(3):
        X, Y, Z = gen_frame_xyz(faces_xyz, frame_idx)
        frame_weight = gen_frame_weight(faces_xyz, frame_idx, follow_code)
        theta, phi, theta2, phi2 = gen_theta_phi(faces_xyz, frame_idx=frame_idx, follow_code=follow_code)
        coeff_start = frame_idx * n_sample_per_frame
        coeff_end = coeff_start + n_sample_per_frame



        for sample_idx in range(n_sample_per_frame):
            if not constant:
                coeff_x_table = coefficient_table[0, :, coeff_start + sample_idx]
                coeff_y_table = coefficient_table[1, :, coeff_start + sample_idx]
                coeff_z_table = coefficient_table[2, :, coeff_start + sample_idx]
                coeff_level_table = coefficient_table[3, :, coeff_start + sample_idx]
                coeff_weight_table = coefficient_table[4, :, coeff_start + sample_idx]

                coeff_x = coeff_x_table[0] + coeff_x_table[1] * theta2 + coeff_x_table[2] * phi2
                coeff_y = coeff_y_table[0] + coeff_y_table[1] * theta2 + coeff_y_table[2] * phi2
                coeff_z = coeff_z_table[0] + coeff_z_table[1] * theta2 + coeff_z_table[2] * phi2

                sample_level = coeff_level_table[0] + coeff_level_table[1] * theta2 + coeff_level_table[2] * phi2
                sample_weight = coeff_weight_table[0] + coeff_weight_table[1] * theta2 + coeff_weight_table[2] * phi2

                #min_weight = sample_weight.min()

                coeff_x = np.stack((coeff_x, coeff_x, coeff_x), axis=-1)
                coeff_y = np.stack((coeff_y, coeff_y, coeff_y), axis=-1)
                coeff_z = np.stack((coeff_z, coeff_z, coeff_z), axis=-1)
            else:
                coeff_x = coefficient_table[0, coeff_start + sample_idx]
                coeff_y = coefficient_table[1, coeff_start + sample_idx]
                coeff_z = coefficient_table[2, coeff_start + sample_idx]
                sample_level = coefficient_table[3, coeff_start + sample_idx]
                sample_weight = coefficient_table[4, coeff_start + sample_idx]

                coeff_x = np.stack((coeff_x, coeff_x, coeff_x), axis=-1)
                coeff_y = np.stack((coeff_y, coeff_y, coeff_y), axis=-1)
                coeff_z = np.stack((coeff_z, coeff_z, coeff_z), axis=-1)


            if not allow_neg_weight:
                sample_weight = np.clip(sample_weight, 0, a_max=None)
            sample_weight = sample_weight * frame_weight

            sample_direction = coeff_x * X + coeff_y * Y + coeff_z * Z
            abs_direction = np.abs(sample_direction)
            max_dir = np.max(abs_direction, axis=-1)
            sample_direction_map = sample_direction / np.stack([max_dir, max_dir, max_dir], axis=-1)

            if j_adjust:
                # adjust level
                j = 3 / 4 * np.log2(map_util.dot_vectorized_4D(sample_direction_map, sample_direction_map))
                sample_level += j

            sample_level = np.clip(sample_level, 0, 6)


            color_tmp = interpolator.interpolate_all(sample_direction_map, sample_level)

            color += color_tmp * np.stack((sample_weight, sample_weight, sample_weight), axis=-1)
            weight += sample_weight


    #devide by weight
    weight_stack = np.stack((weight,weight,weight),axis=-1)
    color = color / weight_stack

    return color

def fetch_samples(tex_input,output_level, follow_code = False):
    """

    :param tex_input: the all level cubemap input
    :param output_level: the output level(0-6) with 128 res for level 0 and 2 res for level 6
    :return:
    """
    interpolator = interpolation.trilinear_mipmap_interpolator(tex_input)


    n_tap = 8
    n_subtap = 4
    n_res = 128 >> output_level
    faces_xyz = map_util.texel_directions(n_res)

    coefficient_dirx = coefficient.fetch_coefficient("quad",output_level,0)
    coefficient_diry = coefficient.fetch_coefficient("quad",output_level,1)
    coefficient_dirz = coefficient.fetch_coefficient("quad",output_level,2)
    coefficient_level = coefficient.fetch_coefficient("quad",output_level,3)
    coefficient_weight = coefficient.fetch_coefficient("quad",output_level,4)


    color = np.zeros((6,n_res,n_res,3))
    weight = np.zeros((6,n_res,n_res))


    max_level = -1.0
    min_level = 100


    # frame_idx 0 means up vector is x
    # The coefficient for this frame lies in [frame_idx * 8, (frame_idx + 1) * 8) for the last dimension
    for frame_idx in range(3):
        X, Y, Z = gen_frame_xyz(faces_xyz, frame_idx)
        frame_weight = gen_frame_weight(faces_xyz, frame_idx, follow_code)
        theta,phi,theta2,phi2 = gen_theta_phi(faces_xyz,frame_idx=frame_idx,follow_code=follow_code)

        # For each frame, there are 8(n_tap) * 1/2/4(n_subtap) samples to get
        for sample_group_idx in range(n_tap):
            index = frame_idx * n_tap + sample_group_idx
            coeff_sample_dirx = coefficient_dirx[:,index]
            coeff_sample_diry = coefficient_diry[:,index]
            coeff_sample_dirz = coefficient_dirz[:,index]
            coeff_sample_level = coefficient_level[:,index]
            coeff_sample_weight = coefficient_weight[:,index]
            for sample_idx in range(n_subtap):

                coeff_x = coeff_sample_dirx[0][sample_idx] + coeff_sample_dirx[1][sample_idx] * theta2 + coeff_sample_dirx[2][sample_idx] * phi2
                coeff_y = coeff_sample_diry[0][sample_idx] + coeff_sample_diry[1][sample_idx] * theta2 + coeff_sample_diry[2][sample_idx] * phi2
                coeff_z = coeff_sample_dirz[0][sample_idx] + coeff_sample_dirz[1][sample_idx] * theta2 + coeff_sample_dirz[2][sample_idx] * phi2

                coeff_x = np.stack((coeff_x,coeff_x,coeff_x),axis=-1)
                coeff_y = np.stack((coeff_y,coeff_y,coeff_y),axis=-1)
                coeff_z = np.stack((coeff_z,coeff_z,coeff_z),axis=-1)

                sample_x = X * coeff_x
                sample_y = Y * coeff_y
                sample_z = Z * coeff_z
                sample_level = coeff_sample_level[0][sample_idx] + coeff_sample_level[1][sample_idx] * theta2 + coeff_sample_level[2][sample_idx] * phi2
                sample_weight = coeff_sample_weight[0][sample_idx] + coeff_sample_weight[1][sample_idx] * theta2 + coeff_sample_weight[2][sample_idx] * phi2
                sample_weight = sample_weight * frame_weight

                #compute sample directioin
                sample_direction = sample_x + sample_y + sample_z
                #project it to cube
                max_dir = np.max(np.abs(sample_direction),axis=-1)
                sample_direction /= np.stack((max_dir,max_dir,max_dir),axis=-1)
                #adjust level
                j = 3/4 * np.log2( map_util.dot_vectorized_4D(sample_direction,sample_direction) )
                sample_level += j

                if sample_level.max() >= max_level:
                    max_level = sample_level.max()

                if sample_level.min() <= min_level:
                    min_level = sample_level.min()

                # print("Sample" , sample_idx + sample_group_idx * n_subtap ,", max is ",sample_level.max())
                # print("Sample" , sample_idx + sample_group_idx * n_subtap,", min is ",sample_level.min())


                color_tmp = interpolator.interpolate_all(sample_direction,sample_level)

                color += color_tmp * np.stack((sample_weight,sample_weight,sample_weight),axis=-1)
                weight += sample_weight

    print("min:",min_level,"/max:",max_level)


    #devide by weight
    weight_stack = np.stack((weight,weight,weight),axis=-1)
    color = color / weight_stack

    return color





def torch_model_to_coeff_table(constant:bool,ggx_alpha,n_sample_per_frame,n_multi_loc = None):
    import optmization_torch
    import os
    import torch

    if n_multi_loc is not None:
        if constant:
            model_name = "constant_ggx_multi_" + "{:.3f}".format(ggx_alpha) + "_" + str(n_multi_loc)
        else:
            model_name = "quad_ggx_multi_" + "{:.3f}".format(ggx_alpha) + "_" + str(n_multi_loc)
    else:
        if constant:
            model_name = "constant_ggx_" + "{:.3f}".format(ggx_alpha)
        else:
            model_name = "quad_ggx_" + "{:.3f}".format(ggx_alpha)

    if not constant:
        model = optmization_torch.SimpleModel(n_sample_per_frame)
    else:
        model = optmization_torch.ConstantModel(n_sample_per_frame)

    if os.path.exists("./model/" + model_name):
        model.load_state_dict(torch.load("./model/" + model_name))
    params = model()
    table = params.detach().cpu().numpy()
    return table



def synthetic_filter_showcase_view_dependent(params,constant,adjust_level,ggx_alpha,n_sample_per_frame,level_to_test,
                                             n_multi_loc,optimize_str,random_shuffle,allow_neg_weight, ggx_ref_jac_weight,
                                             mipmaps, view_option_str, view_direction,name_post_fix = None):
    name = map_util.model_filename(ggx_alpha, constant, n_sample_per_frame, n_multi_loc, adjust_level, optimize_str,
                                   random_shuffle, allow_neg_weight, ggx_ref_jac_weight, view_dependent=True,
                                   view_option_str=view_option_str)

    # test if plots directory already exist
    if not os.path.exists("./plots/" + name):
        os.mkdir("./plots/" + name)

    if name_post_fix is not None:
        img_save_name = "filter_" + name + "_" + name_post_fix + ".exr"
    else:
        img_save_name = "filter_" + name + ".exr"

    assert view_direction.ndim == 1 and view_direction.shape[0] == 3
    view_direction /= np.linalg.norm(view_direction, axis=-1, keepdims=True)

    result = fetch_sample_view_dependent_python_table(mipmaps,level_to_test,params,n_sample_per_frame,
                                                      constant=constant, j_adjust=adjust_level,
                                                      allow_neg_weight=allow_neg_weight,
                                                      view_option_str=view_option_str,view_direction=view_direction)

    result *= 1000
    ref_res = 128 >> level_to_test
    image_read.gen_cubemap_preview_image(result, ref_res, None, "./plots/" + name + "/" + img_save_name)




def synthetic_filter_showcase(params, constant:bool, adjust_level:bool, ggx_alpha, n_sample_per_frame, level_to_test, n_multi_loc, optimize_str, random_shuffle, allow_neg_weight, ggx_ref_jac_weight ,mipmaps, view_dependent, view_option_str  ,name_post_fix = None):
    """

    :param params: the polynomial/constant param that is already given
    :param constant:
    :param l_adjust:
    :param ggx_alpha:
    :param n_sample_per_frame:
    :param level_to_test:
    :param n_multo_loc:
    :param optimize_str:
    :param mipmaps: precomputed mipmaps(downsampled) , this will be called multiple times(should be a synthetic one point mipmap
    :return:
    """
    name = map_util.model_filename(ggx_alpha, constant, n_sample_per_frame ,n_multi_loc, adjust_level, optimize_str,random_shuffle, allow_neg_weight, ggx_ref_jac_weight, view_dependent,view_option_str)

    #test if plots directory already exist
    if not os.path.exists("./plots/" + name):
        os.mkdir("./plots/" + name)

    if name_post_fix is not None:
        img_save_name = "filter_" + name + "_" + name_post_fix  + ".exr"
    else:
        img_save_name = "filter_" + name + ".exr"
    ref_res = 128 >> level_to_test
    if view_dependent:
        view_direction = np.array([0.8,0.8,1])
        view_direction /= np.linalg.norm(view_direction)
        result = fetch_sample_view_dependent_python_table(mipmaps,level_to_test,params,n_sample_per_frame,constant=constant,j_adjust=adjust_level,allow_neg_weight=allow_neg_weight,view_option_str=view_option_str,view_direction=view_direction)
    else:
        result = fetch_samples_python_table(mipmaps, level_to_test, params, n_sample_per_frame, constant=constant, j_adjust=adjust_level, allow_neg_weight=allow_neg_weight)
    result *= 1000
    image_read.gen_cubemap_preview_image(result, ref_res, None, "./plots/" + name + "/" + img_save_name)

def test_coef(constant:bool,ggx_alpha,n_sample_per_frame,level_to_test,n_multi_loc = None, adjust_level = False, optimize_str = 'adam', random_shuffle = False, allow_neg_weight = False, view_dependent = False, view_option_str = "None"):

    name = map_util.model_filename(ggx_alpha,constant, n_sample_per_frame ,n_multi_loc, adjust_level, optimize_str, random_shuffle, allow_neg_weight)

    if not constant:
        model = QuadModel(n_sample_per_frame)
    else:
        model = ConstantModel(n_sample_per_frame)

    if os.path.exists("./model/" + name):
        model.load_state_dict(torch.load("./model/" + name, map_location=torch.device('cpu')))
    if os.path.exists("../ssh_dir/" + name):
        model.load_state_dict(torch.load("../ssh_dir/" + name, map_location=torch.device('cpu')))

    params = model()
    table = params.detach().cpu().numpy()


    u,v = 0.5,0.5
    face = 4
    direction = map_util.uv_to_xyz((u,v),face)

    #mipmap_l0 = reference.synthetic_onepoint_input(direction, high_res)
    mipmap_l0 = image_read.envmap_to_cubemap('exr_files/08-21_Swiss_A.hdr',high_res)
    mipmaps = interpolation.downsample_full(mipmap_l0,n_mipmap_level)

    ref_res = high_res >> level_to_test

    #ref = reference.compute_reference(mipmap_l0,high_res,ref_res,ggx_alpha)

    #result = fetch_samples_python_table(mipmaps,level_to_test,table,8,constant=constant, j_adjust=False)
    if not view_dependent:
        result_level_adjust = fetch_samples_python_table(mipmaps,level_to_test,table,8,constant=constant, j_adjust=adjust_level, allow_neg_weight=allow_neg_weight)
    else:
        result_level_adjust = fetch_sample_view_dependent_python_table(mipmaps,level_to_test,table,256,constant=constant,
                                                                       j_adjust=adjust_level, allow_neg_weight=allow_neg_weight
                                                                       ,view_option_str=view_option_str,view_direction=np.array([1.0,1.0,1.0]))

    #result_level_adjust *= 1000

    image_read.gen_cubemap_preview_image(result_level_adjust,ref_res,None, "filter_" + name+'.exr')


    #diff_level_adjust = ref - result
    #diff = ref -result_level_adjust


    #image_read.gen_cubemap_preview_image(result,16,filename="filter_ggx0.1_16.exr")

    face = 4
    u,v = 0.8,0.2
    location_global = map_util.uv_to_xyz((u, v), face)
    print(location_global)





def test_coef_synthetic(constant:bool,ggx_alpha,n_sample_per_frame,n_multi_loc = None, adjust_level = False):
    import reference
    import matplotlib.pyplot as plt
    table = torch_model_to_coeff_table(constant,ggx_alpha,n_sample_per_frame,n_multi_loc)

    u,v = 0.8,0.2
    face = 4
    direction = map_util.uv_to_xyz((u,v),face)

    mipmap_l0 = reference.synthetic_onepoint_input(direction,high_res)
    mipmaps = interpolation.downsample_full(mipmap_l0,n_mipmap_level)


    #ref = reference.compute_reference(mipmap_l0,high_res,16,ggx_alpha)
    ref = reference.compute_ggx_ndf_reference(16, 0.1, direction)

    result = fetch_samples_python_table(mipmaps,3,table,8,constant=constant, j_adjust=False)

    result_level_adjust = fetch_samples_python_table(mipmaps,3,table,8,constant=constant, j_adjust=True)

    #normalize
    ref = ref / np.sum(ref)
    result = result / np.sum(result)
    result_level_adjust = result_level_adjust / np.sum(result_level_adjust)

    tmpmax = np.max(np.maximum(np.maximum(result,result_level_adjust),ref))

    plt.imshow(ref[4,...,0],vmin=0,vmax=tmpmax)
    plt.show()
    plt.imshow(result[4,...,0],vmin=0,vmax=tmpmax)
    plt.show()
    plt.imshow(result_level_adjust[4,...,0],vmin=0,vmax=tmpmax)
    plt.show()

    diff_level_adjust = ref - result
    diff = ref -result_level_adjust


    #image_read.gen_cubemap_preview_image(result,16,filename="filter_ggx0.1_16.exr")

    face = 4
    u,v = 0.8,0.2
    location_global = map_util.uv_to_xyz((u, v), face)
    print(location_global)


def test_ref_coef_const():
    mipmap_l0 = image_read.envmap_to_cubemap('exr_files/08-21_Swiss_A.hdr', high_res)
    mipmaps = interpolation.downsample_full(mipmap_l0, n_mipmap_level)

    table = coefficient.expand_float4(coefficient.coefficient_const8)
    output_level = 3

    result = fetch_samples_python_table(mipmaps,3,table[output_level],8,constant=True, j_adjust=False)
    image_read.gen_cubemap_preview_image(result,16,filename="filter_ggx_level3_const_codetable.exr")



def test_ref_coef(constant:bool, n_sample:int):
    direction = map_util.uv_to_xyz((0.5,0.5),4)

    param_table = coefficient.get_coeff_table(constant,n_sample)
    mipmap_l0 = reference.synthetic_onepoint_input(direction, high_res)
    #mipmap_l0 = image_read.envmap_to_cubemap('exr_files/08-21_Swiss_A.hdr',high_res)
    mipmaps = interpolation.downsample_full(mipmap_l0,n_mipmap_level)

    level_to_test = 1

    ref_res = high_res >> level_to_test

    #ref = reference.compute_reference(mipmap_l0,high_res,ref_res,ggx_alpha)

    #result = fetch_samples_python_table(mipmaps,level_to_test,table,8,constant=constant, j_adjust=False)

    result_level_adjust = fetch_samples_python_table(mipmaps,level_to_test,param_table[level_to_test],n_sample,constant=constant, j_adjust=True)

    print("Done")



def fetch_synthetic(direction,res):
    from reference import synthetic_onepoint_input
    mipmap_l0 = synthetic_onepoint_input(direction,res)
    mipmaps = interpolation.downsample_full(mipmap_l0,7,j_inv=False)
    return mipmaps



def compare_view_independent_ggx_kernel(constant,adjust_level,ggx_alpha,n_sample_per_frame,level_to_test,
                                             n_multi_loc,optimize_str,random_shuffle,allow_neg_weight, ggx_ref_jac_weight,
                                             view_option_str, view_reflection_parameterization, clip_ndf ,synthetic = True):
    model_name = map_util.model_filename(ggx_alpha, constant, n_sample_per_frame, n_multi_loc, adjust_level,
                                         optimize_str, random_shuffle,
                                         allow_neg_weight, ggx_ref_jac_weight, False, view_option_str,
                                         view_reflection_parameterization, clip_ndf)

    view_model_dict = create_view_model_dict()

    if constant:
        model = ConstantModel(n_sample_per_frame)
    else:
        model = QuadModel(n_sample_per_frame)

    if os.path.exists("./model/" + model_name):
        model.load_state_dict(torch.load("./model/" + model_name, map_location=torch.device('cpu')))
    else:
        raise NotImplementedError

    params = model().cpu().detach().numpy()
    res = 128 >> level_to_test


    texel_dir_torch_map = torch.from_numpy(map_util.texel_directions(res).astype(np.float32))
    texel_dir_torch = texel_dir_torch_map / torch.linalg.norm(texel_dir_torch_map, dim=-1, keepdim=True)

    if synthetic:
        test_direction = map_util.random_dir_sphere(n_dir=1)

        # mipmap = interpolation.downsample_full(mipmap_l0, 7, j_inv=False)

        mipmap = fetch_synthetic(test_direction, 128)
        result = reference.compute_reference(mipmap[0], 128, res, ggx_alpha)
        save_name = "./view_independent_test/" + "integral_synthetic_{:.3f}".format(ggx_alpha) + ".exr"
        image_read.gen_cubemap_preview_image(result, res, filename=save_name)

        fetched_filtered_result = fetch_samples_python_table(mipmap,level_to_test,params,n_sample_per_frame, constant=constant,
                                                                           j_adjust=adjust_level,
                                                                           allow_neg_weight=allow_neg_weight)


        save_name = "./view_independent_test/" + "filtered_synthetic_{:.3f}".format(ggx_alpha,) + ".exr"
        image_read.gen_cubemap_preview_image(fetched_filtered_result, res, filename=save_name)
    else:
        file_name = "08-21_Swiss_A.hdr"
        mipmap_l0 = image_read.envmap_to_cubemap('exr_files/' + file_name, 128)

        result = reference.compute_reference(mipmap_l0, 128, res, ggx_alpha)
        save_name = "./view_independent_test/" + "integral_{:.3f}".format(ggx_alpha) + ".exr"
        image_read.gen_cubemap_preview_image(result, res, filename=save_name)
        mipmap = interpolation.downsample_full(mipmap_l0, 7, j_inv=False)
        fetched_filtered_result = fetch_samples_python_table(mipmap, level_to_test, params, n_sample_per_frame,
                                                             constant=constant,
                                                             j_adjust=adjust_level,
                                                             allow_neg_weight=allow_neg_weight)
        save_name = "./view_independent_test/" + "filtered_{:.3f}".format(ggx_alpha) + ".exr"
        image_read.gen_cubemap_preview_image(fetched_filtered_result, res, filename=save_name)



def compare_view_dependent_ggx_kernel(constant,adjust_level,ggx_alpha,n_sample_per_frame,level_to_test,
                                             n_multi_loc,optimize_str,random_shuffle,allow_neg_weight, ggx_ref_jac_weight,
                                             view_option_str, view_reflection_parameterization, clip_ndf ,synthetic = True, clip_when_fetching = False):
    model_name = map_util.model_filename(ggx_alpha,constant,n_sample_per_frame,n_multi_loc,adjust_level,optimize_str,random_shuffle,
                                         allow_neg_weight, ggx_ref_jac_weight,True,view_option_str, view_reflection_parameterization, clip_ndf)


    view_model_dict = create_view_model_dict()

    model = (view_model_dict[view_option_str])(n_sample_per_frame)

    if os.path.exists("./model/" + model_name):
        model.load_state_dict(torch.load("./model/" + model_name, map_location=torch.device('cpu')))
    else:
        raise NotImplementedError

    params = model().cpu().detach().numpy()
    res = 128 >> level_to_test



    normal_direction = random_dir_sphere(None,1)
    view_direction_tmp = random_dir_hemisphere(None,1)

    up_vector = torch.Tensor([1.0,0.0,0.0])
    if torch.sum(up_vector * normal_direction) > 0.9:
        up_vector = torch.Tensor([0.0,1.0,0.0])

    X = torch_normalized(torch.linalg.cross(up_vector.reshape((1,3)), normal_direction))
    Y = torch.linalg.cross(normal_direction,X)

    view_direction = X * view_direction_tmp[0][0] + Y * view_direction_tmp[0][1] + normal_direction * view_direction_tmp[0][2]


    # #test
    # normal_direction = torch.Tensor([[1.0,0.2,0.1]])
    # a = torch.Tensor([[0,0,1]])
    # normal_direction = normal_direction / torch.linalg.norm(normal_direction)
    # view_direction = get_reflected_vector_torch_vectorized(normal_direction,a)
    # #test-end


    texel_dir_torch_map = torch.from_numpy(map_util.texel_directions(res).astype(np.float32))
    texel_dir_torch = texel_dir_torch_map / torch.linalg.norm(texel_dir_torch_map, dim=-1, keepdim=True)

    if synthetic:
        reflected_direction = get_reflected_vector_torch_vectorized(normal_direction.reshape((1, 3)),
                                                                    view_direction.reshape((1, 3)))

        reflected_direction = reflected_direction.cpu().detach().numpy().flatten()
        view_direction = view_direction.cpu().detach().numpy()
        # mipmap = interpolation.downsample_full(mipmap_l0, 7, j_inv=False)

        mipmap = fetch_synthetic(reflected_direction, 128)
        result = reference.compute_reference_view_dependent(mipmap[0], 128, res, info[3].roughness, view_direction)
        save_name = "./view_dependent_test/" + "integral_synthetic_{:.3f}_{:.4f}_{:.4f}_{:.4f}".format(ggx_alpha,
                                                                                             view_direction[0, 0],
                                                                                             view_direction[0, 1],
                                                                                             view_direction[
                                                                                                 0, 2]) + ".exr"
        image_read.gen_cubemap_preview_image(result, res, filename=save_name)

        view_direction = view_direction.flatten()
        fetched_filtered_result = fetch_sample_view_dependent_python_table(mipmap, level_to_test, params,
                                                                           n_sample_per_frame, constant=constant,
                                                                           j_adjust=adjust_level,
                                                                           allow_neg_weight=allow_neg_weight,
                                                                           view_option_str=view_option_str,
                                                                           view_direction=view_direction,
                                                                           clip_below_horizon=clip_when_fetching)

        save_name = "./view_dependent_test/" + "filtered_synthetic_{:.3f}_{:.4f}_{:.4f}_{:.4f}".format(ggx_alpha,
                                                                                             view_direction[0],
                                                                                             view_direction[1],
                                                                                             view_direction[2]) + ".exr"
        image_read.gen_cubemap_preview_image(fetched_filtered_result, res, filename=save_name)
    else:
        reflected_direction = get_reflected_vector_torch_vectorized(normal_direction.reshape((1, 3)),
                                                                    view_direction.reshape((1, 3)))


        print("Test view_theta:{:.4f}pi".format(np.arccos(view_direction_tmp[0,2].item()) / np.pi ))

        reflected_direction = reflected_direction.cpu().detach().numpy().flatten()
        view_direction = view_direction.cpu().detach().numpy()
        file_name = "08-21_Swiss_A.hdr"
        mipmap_l0 = image_read.envmap_to_cubemap('exr_files/' + file_name, 128)


        result = reference.compute_reference_view_dependent(mipmap_l0,128, res, info[3].roughness, view_direction)
        save_name = "./view_dependent_test/" + "integral_{:.3f}_{:.4f}_{:.4f}_{:.4f}".format(ggx_alpha,view_direction[0, 0], view_direction[0, 1],
                                                                            view_direction[0, 2]) + ".exr"
        image_read.gen_cubemap_preview_image(result, res, filename=save_name)

        view_direction = view_direction.flatten()
        mipmap = interpolation.downsample_full(mipmap_l0, 7, j_inv=False)
        fetched_filtered_result = fetch_sample_view_dependent_python_table(mipmap,level_to_test,params,
                                                                  n_sample_per_frame,constant=constant,
                                                                  j_adjust=adjust_level,allow_neg_weight=allow_neg_weight,
                                                                  view_option_str=view_option_str,view_direction=view_direction,
                                                                           clip_below_horizon=clip_when_fetching)

        save_name = "./view_dependent_test/" + "filtered_{:.3f}_{:.4f}_{:.4f}_{:.4f}".format(ggx_alpha,view_direction[0], view_direction[1],
                                                                            view_direction[2]) + ".exr"
        image_read.gen_cubemap_preview_image(fetched_filtered_result, res, filename=save_name)







def visualize_view_dependent_filter_sample_directions(constant,adjust_level,ggx_alpha,n_sample_per_frame,level_to_test,
                                             n_multi_loc,optimize_str,random_shuffle,allow_neg_weight, ggx_ref_jac_weight,
                                             view_option_str, view_reflection_parameterization, clip_ndf):
    """
    Only works for relative frame for now
    :param constant:
    :param adjust_level:
    :param ggx_alpha:
    :param n_sample_per_frame:
    :param level_to_test:
    :param n_multi_loc:
    :param optimize_str:
    :param random_shuffle:
    :param allow_neg_weight:
    :param ggx_ref_jac_weight:
    :param view_option_str:
    :param view_reflection_parameterization:
    :return:
    """
    model_name = map_util.model_filename(ggx_alpha, constant, n_sample_per_frame, n_multi_loc, adjust_level,
                                         optimize_str, random_shuffle,
                                         allow_neg_weight, ggx_ref_jac_weight, True, view_option_str,
                                         view_reflection_parameterization, clip_ndf)

    view_model_dict = create_view_model_dict()

    model = (view_model_dict[view_option_str])(n_sample_per_frame)

    if os.path.exists("./model/" + model_name):
        model.load_state_dict(torch.load("./model/" + model_name, map_location=torch.device('cpu')))
    else:
        raise NotImplementedError





    coefficient_table = model().cpu().detach().numpy()
    res = 128 >> level_to_test
    rng = np.random.default_rng(int(datetime.now().timestamp()))

    random_number = rng.integers(low = 1, high = 124451251,size = 1)
    g = torch.manual_seed(random_number[0])


    view_direction, view_theta ,xyz_cube, xyz = sample_view_dependent_location(1,g,True)

    view_direction = view_direction.cpu().detach().numpy().reshape((1,3))
    normal_direction = xyz.cpu().detach().numpy().reshape((1,3))


    X,Y,Z = gen_anisotropic_frame_xyz(normal_direction,view_direction)
    theta,phi,theta2,phi2 = gen_theta_phi_no_frame(normal_direction)
    cosine_view = np.dot(normal_direction.flatten(),view_direction.flatten())
    reflected_direction = map_util.get_reflected_vector_vectorized(normal_direction,view_direction)

    X_frame = X.flatten()

    #mask_above_horizon = cosine_view > 0.0

    view_theta = np.arccos(cosine_view)
    view_theta2 = view_theta ** 2

    all_directions = []
    all_weights= []

    theta_normal, phi_normal, theta_normal2, phi_normal2 = gen_theta_phi_no_frame(normal_direction)
    theta_reflection, phi_reflection, theta_reflection2, phi_reflection2 = gen_theta_phi_no_frame(reflected_direction)


    for sample_idx in range(n_sample_per_frame):


        coeff_x_table = coefficient_table[0, :, sample_idx]
        coeff_y_table = coefficient_table[1, :, sample_idx]
        coeff_z_table = coefficient_table[2, :, sample_idx]
        coeff_level_table = coefficient_table[3, :, sample_idx]
        coeff_weight_table = coefficient_table[4, :, sample_idx]

        if view_option_str == "relative_frame":
            coeff_x = coeff_x_table[0] + coeff_x_table[1] * theta2 + coeff_x_table[2] * phi2 + coeff_x_table[
                3] * view_theta2 + coeff_x_table[4] * view_theta
            coeff_y = coeff_y_table[0] + coeff_y_table[1] * theta2 + coeff_y_table[2] * phi2 + coeff_y_table[
                3] * view_theta2 + coeff_y_table[4] * view_theta
            coeff_z = coeff_z_table[0] + coeff_z_table[1] * theta2 + coeff_z_table[2] * phi2 + coeff_z_table[
                3] * view_theta2 + coeff_z_table[4] * view_theta
            sample_weight = coeff_weight_table[0] + coeff_weight_table[1] * theta2 + coeff_weight_table[2] * phi2 + \
                            coeff_weight_table[3] * view_theta2 + coeff_weight_table[4] * view_theta
        elif view_option_str == "relative_frame_full":
            coeff_x = (coeff_x_table[0] + coeff_x_table[1] * theta_normal2 + coeff_x_table[2] * phi_normal2 +
                       + coeff_x_table[3] * theta_reflection2 + coeff_x_table[4] + phi_reflection2 + coeff_x_table[
                           5] * view_theta2 + coeff_x_table[6] * view_theta)
            coeff_y = (coeff_y_table[0] + coeff_y_table[1] * theta_normal2 + coeff_y_table[2] * phi_normal2 +
                       coeff_y_table[3] * theta_reflection2 + coeff_y_table[4] + phi_reflection2 + coeff_y_table[
                           5] * view_theta2 + coeff_y_table[6] * view_theta)
            coeff_z = (coeff_z_table[0] + coeff_z_table[1] * theta_normal2 + coeff_z_table[2] * phi_normal2 +
                       coeff_z_table[3] * theta_reflection2 + coeff_z_table[4] + phi_reflection2 + coeff_z_table[
                           5] * view_theta2 + coeff_z_table[6] * view_theta)
            sample_weight = (
                        coeff_weight_table[0] + coeff_weight_table[1] * theta_normal2 + coeff_weight_table[2] * phi_normal2 +
                        coeff_weight_table[3] * theta_reflection2 + coeff_weight_table[4] + phi_reflection2 +
                        coeff_weight_table[5] * view_theta2 + coeff_weight_table[6] * view_theta)
        elif view_option_str == "relative_frame_full_interaction":
            coeff_x = (coeff_x_table[0] + coeff_x_table[1] * theta_normal2 + coeff_x_table[2] * phi_normal2 +
                       coeff_x_table[3] * theta_reflection2 + coeff_x_table[4] + phi_reflection2 + coeff_x_table[
                           5] * view_theta2 + coeff_x_table[6] * view_theta + coeff_x_table[
                           7] * theta_normal2 * view_theta + coeff_x_table[8] * phi_normal2 * view_theta +
                       coeff_x_table[9] * theta_reflection2 * view_theta + coeff_x_table[
                           10] * phi_reflection2 * view_theta)
            coeff_y = (coeff_y_table[0] + coeff_y_table[1] * theta_normal2 + coeff_y_table[2] * phi_normal2 +
                       coeff_y_table[3] * theta_reflection2 + coeff_y_table[4] + phi_reflection2 + coeff_y_table[
                           5] * view_theta2 + coeff_y_table[6] * view_theta + coeff_y_table[
                           7] * theta_normal2 * view_theta + coeff_y_table[8] * phi_normal2 * view_theta +
                       coeff_y_table[9] * theta_reflection2 * view_theta + coeff_y_table[
                           10] * phi_reflection2 * view_theta)
            coeff_z = (coeff_z_table[0] + coeff_z_table[1] * theta_normal2 + coeff_z_table[2] * phi_normal2 +
                       coeff_z_table[3] * theta_reflection2 + coeff_z_table[4] + phi_reflection2 + coeff_z_table[
                           5] * view_theta2 + coeff_z_table[6] * view_theta + coeff_z_table[
                           7] * theta_normal2 * view_theta + coeff_z_table[8] * phi_normal2 * view_theta +
                       coeff_z_table[9] * theta_reflection2 * view_theta + coeff_z_table[
                           10] * phi_reflection2 * view_theta)
            sample_weight = (coeff_weight_table[0] + coeff_weight_table[1] * theta_normal2 + coeff_weight_table[
                2] * phi_normal2 + coeff_weight_table[3] * theta_reflection2 + coeff_weight_table[
                                 4] + phi_reflection2 + coeff_weight_table[5] * view_theta2 + coeff_weight_table[
                                 6] * view_theta + coeff_weight_table[7] * theta_normal2 * view_theta +
                             coeff_weight_table[8] * phi_normal2 * view_theta + coeff_weight_table[
                                 9] * theta_reflection2 * view_theta + coeff_weight_table[
                                 10] * phi_reflection2 * view_theta)
        else:
            raise NotImplementedError





        # coeff_x = coeff_x_table[0] + coeff_x_table[1] * theta2 + coeff_x_table[2] * phi2 + coeff_x_table[
        #     3] * view_theta2 + coeff_x_table[4] * view_theta
        # coeff_y = coeff_y_table[0] + coeff_y_table[1] * theta2 + coeff_y_table[2] * phi2 + coeff_y_table[
        #     3] * view_theta2 + coeff_y_table[4] * view_theta
        # coeff_z = coeff_z_table[0] + coeff_z_table[1] * theta2 + coeff_z_table[2] * phi2 + coeff_z_table[
        #     3] * view_theta2 + coeff_z_table[4] * view_theta
        # sample_weight = coeff_weight_table[0] + coeff_weight_table[1] * theta2 + coeff_weight_table[2] * phi2 + \
        #                 coeff_weight_table[3] * view_theta2 + coeff_weight_table[4] * view_theta
        coeff_x = np.stack((coeff_x, coeff_x, coeff_x), axis=-1)
        coeff_y = np.stack((coeff_y, coeff_y, coeff_y), axis=-1)
        coeff_z = np.stack((coeff_z, coeff_z, coeff_z), axis=-1)


        if not allow_neg_weight:
            sample_weight = np.clip(sample_weight, 0, a_max=None)
        sample_weight = sample_weight * 1

        sample_direction = coeff_x * X + coeff_y * Y + coeff_z * Z
        abs_direction = np.abs(sample_direction)
        max_dir = np.max(abs_direction, axis=-1)
        sample_direction_map = sample_direction / np.stack([max_dir, max_dir, max_dir], axis=-1)
        sample_direction_normalized = sample_direction / np.linalg.norm(sample_direction, axis=-1, keepdims=True)
        if np.sum(sample_direction_normalized * normal_direction) > 0.0:
            all_directions.append(sample_direction_normalized.flatten())
            all_weights.append(sample_weight[0] / 10.0)

    import visualization

    reflect_direction = map_util.get_reflected_vector_vectorized(normal_direction,view_direction)

    print("valid above hemisphere sample count:{} over {}".format(len(all_directions),n_sample_per_frame))

    visualization.plot_nvl_vector(normal_direction.flatten(),view_direction.flatten(),
                                  reflect_direction.flatten(),all_directions,all_weights, X_frame)


if __name__ == '__main__':
    n_mipmap_level = 7
    high_res = 2**n_mipmap_level

    import specular
    info = specular.cubemap_level_params(18)

    test_option = 1
    if test_option == 0:
        visualize_view_dependent_filter_sample_directions(constant=False, adjust_level=True,
                                                          ggx_alpha=info[3].roughness, n_sample_per_frame=96,
                                                          level_to_test=3
                                                          , n_multi_loc=300, optimize_str="bfgs", random_shuffle=True,
                                                          allow_neg_weight=True,
                                                          ggx_ref_jac_weight='light',
                                                          view_option_str="relative_frame_full_interaction",
                                                          view_reflection_parameterization=False, clip_ndf=False)
    elif test_option == 1:
        compare_view_dependent_ggx_kernel(constant=False, adjust_level=True, ggx_alpha=info[3].roughness,
                                          n_sample_per_frame=96, level_to_test=3
                                          , n_multi_loc=300, optimize_str="bfgs", random_shuffle=True,
                                          allow_neg_weight=True,
                                          ggx_ref_jac_weight='light', view_option_str="relative_frame_full_interaction",
                                          view_reflection_parameterization=False, clip_ndf=False, synthetic=True)
    elif test_option == 2:
        compare_view_independent_ggx_kernel(constant=False, adjust_level=True, ggx_alpha=info[3].roughness,
                                            n_sample_per_frame=32, level_to_test=3
                                            , n_multi_loc=300, optimize_str="bfgs", random_shuffle=True,
                                            allow_neg_weight=True,
                                            ggx_ref_jac_weight='light', view_option_str="None",
                                            view_reflection_parameterization=False, clip_ndf=False, synthetic=False)





    #
    #
    # #test_ref_coef(False,32)
    #
    # import specular
    # info = specular.cubemap_level_params(18)
    #
    # level_jacobian = True
    #
    # #test_ref_coef_const()
    #
    # #test_coef(False,info[4].roughness,8, info[4].level,80, level_jacobian,"adam")
    # test_coef(False, 0.100, 8, 3, 1000, level_jacobian, "bfgs", random_shuffle=True,allow_neg_weight=True)
    #
    # j_inverse = False
    # code_follow = False
    #
    # mipmap_l0 = image_read.envmap_to_cubemap('exr_files/08-21_Swiss_A.hdr',high_res)
    # mipmaps = interpolation.downsample_full(mipmap_l0,n_mipmap_level,j_inverse)
    #
    #
    # # #generate preview images
    # # for output_level in range(n_mipmap_level):
    # #     image_read.gen_cubemap_preview_image(mipmaps[output_level],high_res>>output_level,filename="preview_l"+str(output_level)+".exr")
    #
    #
    #
    # for output_level in tqdm(range(n_mipmap_level)):
    #     this_face = fetch_samples(mipmaps, output_level, code_follow)
    #     if not j_inverse:
    #         filename = "filter_l" + str(output_level) + ".exr"
    #     else:
    #         filename = "filter_l" + str(output_level) + "_j_inv.exr"
    #
    #     if code_follow:
    #         filename = filename[:6] + "_fcode" + filename[6:]
    #
    #     image_read.gen_cubemap_preview_image(this_face, high_res >> output_level, None, filename)