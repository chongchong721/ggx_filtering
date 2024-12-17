import numpy as np

import map_util
import mat_util
from datetime import datetime

from reference import compute_ggx_ndf_reference,compute_ggx_ndf_reference_torch_vectorized,compute_ggx_ndf_reference_half_vector_torch_vectorized,compute_ggx_ndf_ref_view_dependent_torch_vectorized,compute_ggx_vndf_ref_view_dependent_torch_vectorized

import scipy
import torch
import torch.nn as nn
import torch.optim as optim

import os
import logging

import torch_util
import torch_util_all_locations

import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method

import time

import sys

import argparse

from LBFGS import FullBatchLBFGS,LBFGS

import reference
from filter import synthetic_filter_showcase

import image_read


def process_view_option(view_option_str:str,view_reflection_parameterization:bool):
    valid_option = ["view_only","even_only",'odd',"reflect_norm","relative_frame","relative_frame_full","relative_frame_full_interaction"
                    ,"relative_frame_full_interaction_view_theta2"]
    if view_option_str.lower() in valid_option:
        view_dependent = True
    else:
        view_dependent = False
        if view_reflection_parameterization:
            print("Warning: Setting view_reflection_parameterization to False has no effect when view_dependent is False")

    return view_dependent,view_option_str.lower()


def process_cmd():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optimizer", type=str)
    parser.add_argument("-l", "--ggxlevel",type=int)
    parser.add_argument("-n", "--ndir",type=int)
    parser.add_argument("-r", "--ggxalpha", type=float)
    parser.add_argument("-f","--nsampleframe",type=int)
    parser.add_argument("-v","--view",type=str,default="None",
                        help='How to parameterize view direction. Choose from view_only/even_only/odd')
    parser.add_argument("-vreflect", "--view-reflection-parameterization",type=str2bool, nargs='?',
                        const=True,
                        default=False,
                        help='Whether to generate theta phi according to reflected directions'
                        )
    parser.add_argument("-vclip", "--view-ndf-clipping",type=str2bool, nargs='?',
                        const=True,
                        default=False,
                        help='Whether to clip below horizon ndf during optimization'
                        )

    parser.add_argument("-s", "--shuffle", type=str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help='Whether to use random directions for each iteration'
                        )
    parser.add_argument(
        '-w', '--negweight',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Allow negative weight or not'
    )
    parser.add_argument(
        '-c', '--constant',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Enable or disable constant mode (True/False).'
    )

    parser.add_argument(
        '-a', '--adjustlevel',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Enable or disable adjust level mode (True/False).'
    )
    parser.add_argument(
        '-j', '--jacref',
        type=str,
        default='None',
        help='apply what jacobian to reference'
    )

    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer (default: 0.01)')

    parser.add_argument(
        '-vndf', '--use-vndf',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Whether to use vndf in view dependent reference.'
    )

    parser.add_argument(
        '--fixed-costheta-res',type=int
        ,default = -1, help='resolution of fixed cos theta(default -1)\nif negative, no fixed cos theta optimization'
    )

    parser.add_argument(
        '--fixed-costheta-idx',type=int,default=-1,help='only valid when fixed-costheta-res is larger than 0'
    )


    args = parser.parse_args()

    optimizer = args.optimizer
    ggx_level = args.ggxlevel
    ggx_alpha_input = args.ggxalpha
    ndir = args.ndir
    constant = args.constant
    adjust_level = args.adjustlevel
    optimizer = optimizer.lower()
    random_shuffle = args.shuffle
    allow_neg_weight = args.negweight
    n_sample_per_frame = args.nsampleframe
    ggx_ref_jac_weight = args.jacref
    lr = args.learning_rate
    view_option = args.view
    view_reflection_parameterization = args.view_reflection_parameterization
    view_ndf_cliping = args.view_ndf_clipping
    use_vndf = args.use_vndf

    fixed_costheta_res = args.fixed_costheta_res
    fixed_costheta_idx = args.fixed_costheta_idx

    if ggx_alpha_input is not None and ggx_level is not None:
        raise ValueError('ggx_alpha and ggx_level cannot be both specified.')



    if view_option == "None":
        if fixed_costheta_res >= 0 or fixed_costheta_idx >= 0:
            raise NotImplementedError
        fixed_costheta_info = None
    else:
        fixed_costheta_info = (fixed_costheta_res,fixed_costheta_idx)

    return n_sample_per_frame,ndir, ggx_level if ggx_level is not None else ggx_alpha_input, constant, adjust_level, optimizer, random_shuffle, allow_neg_weight, ggx_ref_jac_weight, lr, view_option, view_reflection_parameterization, view_ndf_cliping, use_vndf, fixed_costheta_info





"""
Optimization routine

All the mipmaps here have only one channel, which is the weight/contribution

"""
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device




def gen_path(is_constant,texel_location,current_loss):
    name = ""
    if is_constant:
        name += "constant_"
    else:
        name += "quad_"
    texel_location = texel_location.flatten()
    x = texel_location[0]
    y = texel_location[1]
    z = texel_location[2]
    location_text = "{.2f}_{.2f}_{.2f}_".format(x,y,z)
    loss_text = "loss{.3f}".format(current_loss)
    name = name + location_text + loss_text + '.pt'
    return name

def save_model(model:nn.Module,path):
    torch.save(model.state_dict(), path)

def load_model(path):

    if os.path.exists(path):
        module = torch.load(path)
        return module
    else:
        return None





def L1_error_one_texel(ggx_kernel, contribution_map):
    """
    Comparison should be made between *normalized* map
    :param ggx_kernel:
    :param contribution_map:
    :return:
    """

    l1_loss = torch.abs(ggx_kernel - contribution_map / torch.sum(contribution_map))
    return l1_loss





def error_func(x, texel_direction, n_sample_per_frame, ggx_ref, constant= False):
    """
        A wrapper to compute error as scipy required
    :param x:
    :param texel_direction:
    :param n_sample_per_frame:
    :param ggx_ref:
    :param constant: whether to use constant coeff table
    :return:
    """
    if not constant:
        x = x.reshape((5, 3, n_sample_per_frame * 3))
    else:
        x = x.reshape((5, n_sample_per_frame * 3))
    error, result = test_one_texel_full_optimization(texel_direction, n_sample_per_frame, ggx_ref, x, constant)
    return error, result


def multiple_texel_full_optimization_view_dependent_vectorized(n_sample_per_frame, n_sample_per_level, ggx_ref_list, weight_list, xyz_list, theta_phi_list, view_theta_list, coef_table=None, constant = False ,adjust_level = False, allow_neg_weight = False, view_option_str = 'None' ,device = torch.device('cpu')):
    """

    If we add one more parameter of the cos(theta_d)? or cos(theta_h&n)?, we need more coefficients,
    do we use different theta in different frames?

    we need two more parameters

    We still want to keep quadratic polynomial, we want to make sure for each view direction, the preimage is symmetric?

    k1 * view_theta * 2 + k2 * view_theta?

    Now experiment with three options

    1. Direction is determined by c0 + c1 * theta^2 + c2 * phi^2+ c3 * theta_view^2 + c4 * theta_view + c5 * theta_view * theta2 + c6 * theta_view * phi2 (odd)
    2. Direction is determined by c0 + c1 * theta^2 + c2 * phi^2+ c3 * theta_view^2 + c4 * theta_view (even_only)
    3. Direction is determined by c0 + c1 * theta_view^2 + c2 * theta_view  (view_only)

    :param n_sample_per_frame:
    :param n_sample_per_level:
    :param ggx_ref_list:
    :param weight_list:
    :param xyz_list:
    :param theta_phi_list:
    :param view_theta_list: view dot n -> in shape of (n_sample_per_level,)
    :param coef_table:
    :param constant:
    :param adjust_level:
    :param device:
    :return:
    """
    if coef_table is None:
        # a single coefficient table in shape [5,3,nsample_per_frame * 3]
        coefficient_table = torch.rand((5, 5, n_sample_per_frame * 3))
    else:
        coefficient_table = coef_table

    sample_directions = torch.empty((0, 3), device=device)
    sample_weights = torch.empty((0,), device=device)
    sample_levels = torch.empty((0,), device=device)
    sample_idx = torch.empty((0,), device=device,dtype=torch.int)


    for dir_idx in range(n_sample_per_level):
        for i in range(3):
            X, Y, Z = xyz_list[i][0][dir_idx],xyz_list[i][1][dir_idx],xyz_list[i][2][dir_idx]
            theta2, phi2 = theta_phi_list[i][2][dir_idx],theta_phi_list[i][3][dir_idx]
            frame_weight = weight_list[i][dir_idx]

            view_theta,view_theta2 = view_theta_list[0][dir_idx],view_theta_list[1][dir_idx]

            if frame_weight == 0:
                continue

            coeff_start = i * n_sample_per_frame
            coeff_end = coeff_start + n_sample_per_frame



            coeff_x_table = coefficient_table[0, :, coeff_start:coeff_end]
            coeff_y_table = coefficient_table[1, :, coeff_start:coeff_end]
            coeff_z_table = coefficient_table[2, :, coeff_start:coeff_end]
            coeff_level_table = coefficient_table[3, :, coeff_start:coeff_end]
            coeff_weight_table = coefficient_table[4, :, coeff_start:coeff_end]


            if view_option_str == "even_only":
                coeff_x = coeff_x_table[0] + coeff_x_table[1] * theta2 + coeff_x_table[2] * phi2 + coeff_x_table[3] * view_theta2 + coeff_x_table[4] * view_theta
                coeff_y = coeff_y_table[0] + coeff_y_table[1] * theta2 + coeff_y_table[2] * phi2 + coeff_y_table[3] * view_theta2 + coeff_y_table[4] * view_theta
                coeff_z = coeff_z_table[0] + coeff_z_table[1] * theta2 + coeff_z_table[2] * phi2 + coeff_z_table[3] * view_theta2 + coeff_z_table[4] * view_theta
                level = coeff_level_table[0] + coeff_level_table[1] * theta2 + coeff_level_table[2] * phi2 + coeff_level_table[3] * view_theta2 + coeff_level_table[4] * view_theta
                weight = coeff_weight_table[0] + coeff_weight_table[1] * theta2 + coeff_weight_table[2] * phi2 + coeff_weight_table[3] * view_theta2 + coeff_weight_table[4] * view_theta
            elif view_option_str == "view_only":
                coeff_x = coeff_x_table[0] + coeff_x_table[1] * view_theta2 + coeff_x_table[2] * view_theta
                coeff_y = coeff_y_table[0] + coeff_y_table[1] * view_theta2 + coeff_y_table[2] * view_theta
                coeff_z = coeff_z_table[0] + coeff_z_table[1] * view_theta2 + coeff_z_table[2] * view_theta
                level = coeff_level_table[0] + coeff_level_table[1] * view_theta2 + coeff_level_table[2] * view_theta
                weight = coeff_weight_table[0] + coeff_weight_table[1] * view_theta2 + coeff_weight_table[2] * view_theta
            elif view_option_str == "odd":
                coeff_x = coeff_x_table[0] + coeff_x_table[1] * theta2 + coeff_x_table[2] * phi2 + coeff_x_table[3] * view_theta2 + coeff_x_table[4] * view_theta + coeff_x_table[5] * view_theta * theta2 + coeff_x_table[6] * view_theta * phi2
                coeff_y = coeff_y_table[0] + coeff_y_table[1] * theta2 + coeff_y_table[2] * phi2 + coeff_y_table[3] * view_theta2 + coeff_y_table[4] * view_theta + coeff_y_table[5] * view_theta * theta2 + coeff_y_table[6] * view_theta * phi2
                coeff_z = coeff_z_table[0] + coeff_z_table[1] * theta2 + coeff_z_table[2] * phi2 + coeff_z_table[3] * view_theta2 + coeff_z_table[4] * view_theta + coeff_z_table[5] * view_theta * theta2 + coeff_z_table[6] * view_theta * phi2
                level = coeff_level_table[0] + coeff_level_table[1] * theta2 + coeff_level_table[2] * phi2 + coeff_level_table[3] * view_theta2 + coeff_level_table[4] * view_theta + coeff_level_table[5] * view_theta * theta2 + coeff_level_table[6] * view_theta * phi2
                weight = coeff_weight_table[0] + coeff_weight_table[1] * theta2 + coeff_weight_table[2] * phi2 + coeff_weight_table[3] * view_theta2 + coeff_weight_table[4] * view_theta + coeff_weight_table[5] * view_theta * theta2 + coeff_weight_table[6] * view_theta * phi2
            else:
                raise NotImplementedError

            coeff_x = torch.stack((coeff_x, coeff_x, coeff_x), dim=-1)
            coeff_y = torch.stack((coeff_y, coeff_y, coeff_y), dim=-1)
            coeff_z = torch.stack((coeff_z, coeff_z, coeff_z), dim=-1)


            sample_direction = coeff_x * X + coeff_y * Y + coeff_z * Z
            abs_direction = torch.abs(sample_direction)
            max_dir = torch.max(abs_direction, dim=-1).values
            sample_direction_map = sample_direction / torch.stack([max_dir, max_dir, max_dir], dim=-1)

            weight_cur_frame = weight * frame_weight

            #TODO: how to make weight/level always positive?
            # What if we do not clip weight?
            if not allow_neg_weight:
                weight_cur_frame = torch.clip(weight_cur_frame, 0)


            sample_directions = torch.concatenate((sample_directions, sample_direction_map))
            sample_weights = torch.concatenate((sample_weights, weight_cur_frame))
            sample_idx = torch.concatenate((sample_idx, torch.full([n_sample_per_frame],dir_idx,device=device,dtype=torch.int)))

            #TODO: Jacobian?
            if adjust_level:
                j = 3 / 4 * torch.log2(torch_util.torch_dot_vectorized_2D(sample_direction_map, sample_direction_map))
                adjusted_level = level + j
            else:
                adjusted_level = level

            adjusted_level = torch.clip(adjusted_level,0.0,6.0)

            sample_levels = torch.concatenate((sample_levels, adjusted_level))


    result = torch_util_all_locations.compute_contribution_full(sample_directions,sample_levels,sample_weights,sample_idx,n_level=7,n_sample_per_level = n_sample_per_level)





    return result



def stack_parameter(view_option_str, view_theta,view_theta2,theta_phi_normal_tensor,theta_phi_reflection_tensor,frame_idx, use_reflection_parameterization):
    if view_option_str == "even_only":
        theta2_normal = theta_phi_normal_tensor[frame_idx, 2, :]
        phi2_normal = theta_phi_normal_tensor[frame_idx, 3, :]
        parameter = torch.stack((theta2_normal, phi2_normal, view_theta2, view_theta))
    elif view_option_str == "odd":
        theta2_normal = theta_phi_normal_tensor[frame_idx, 2, :]
        phi2_normal = theta_phi_normal_tensor[frame_idx, 3, :]
        parameter = torch.stack(
            (theta2_normal, phi2_normal, view_theta2, view_theta, view_theta * theta2_normal, view_theta * phi2_normal))
    elif view_option_str == "view_only":
        parameter = torch.stack((view_theta2, view_theta))
    elif view_option_str == "reflect_norm":
        theta2_normal = theta_phi_normal_tensor[frame_idx, 2, :]
        phi2_normal = theta_phi_normal_tensor[frame_idx, 3, :]
        theta2_reflection = theta_phi_reflection_tensor[frame_idx, 2, :]
        phi2_reflection = theta_phi_reflection_tensor[frame_idx, 3, :]
        parameter = torch.stack(
            (theta2_normal, phi2_normal, theta2_reflection, phi2_reflection, view_theta2, view_theta))
    elif view_option_str == "relative_frame":
        if use_reflection_parameterization:
            theta2 = theta_phi_reflection_tensor[frame_idx, 2, :]
            phi2 = theta_phi_reflection_tensor[frame_idx, 3, :]
        else:
            theta2 = theta_phi_normal_tensor[frame_idx, 2, :]
            phi2 = theta_phi_normal_tensor[frame_idx, 3, :]
        parameter = torch.stack((theta2, phi2, view_theta2, view_theta))
    elif view_option_str == "relative_frame_full":
        theta2_normal = theta_phi_normal_tensor[frame_idx, 2, :]
        phi2_normal = theta_phi_normal_tensor[frame_idx, 3, :]
        theta2_reflection = theta_phi_reflection_tensor[frame_idx, 2, :]
        phi2_reflection = theta_phi_reflection_tensor[frame_idx, 3, :]
        parameter = torch.stack(
            (theta2_normal, phi2_normal, theta2_reflection, phi2_reflection, view_theta2, view_theta))
    elif view_option_str == "relative_frame_full_interaction":
        theta2_normal = theta_phi_normal_tensor[frame_idx, 2, :]
        phi2_normal = theta_phi_normal_tensor[frame_idx, 3, :]
        theta2_reflection = theta_phi_reflection_tensor[frame_idx, 2, :]
        phi2_reflection = theta_phi_reflection_tensor[frame_idx, 3, :]
        parameter = torch.stack(
            (theta2_normal, phi2_normal, theta2_reflection, phi2_reflection, view_theta2, view_theta,
             view_theta * theta2_normal, view_theta * phi2_normal, view_theta * theta2_reflection, view_theta * phi2_reflection)
        )
    elif view_option_str == "relative_frame_full_interaction_view_theta2":
        theta2_normal = theta_phi_normal_tensor[frame_idx, 2, :]
        phi2_normal = theta_phi_normal_tensor[frame_idx, 3, :]
        theta2_reflection = theta_phi_reflection_tensor[frame_idx, 2, :]
        phi2_reflection = theta_phi_reflection_tensor[frame_idx, 3, :]
        parameter = torch.stack(
            (theta2_normal, phi2_normal, theta2_reflection, phi2_reflection, view_theta2, view_theta,
             view_theta * theta2_normal, view_theta * phi2_normal, view_theta * theta2_reflection, view_theta * phi2_reflection,
             view_theta2 * theta2_normal, view_theta2 * phi2_normal, view_theta2 * theta2_reflection, view_theta2 * phi2_reflection)
        )
    else:
        raise NotImplementedError

    return parameter


def multiple_texel_full_optimization_view_dependent_vectorized_vec_prepare(n_sample_per_frame, n_sample_per_level, ggx_ref_list, all_params_list ,
                                                                           coef_table=None, adjust_level = False, allow_neg_weight = False,
                                                                           view_option_str = 'None', use_reflection_parameterization = False ,device = torch.device('cpu'), normal_directions = None):
    #First use coefficient to compute...

    #coefficient [5,3,n_sample_per_frame * 3]

    if view_option_str[:14] != "relative_frame" :
        weight_list, xyz_list, theta_phi_normal_list, theta_phi_reflection_list, view_theta_list = all_params_list
        #theta,phi parameter list([list(theta[200],phi,theta2,phi2)](frame0),[],[])
        frame_count = len(theta_phi_normal_list)


        #theta and phi convert to tensor of size [3,4,n_sample_per_level]
        theta_phi_normal_tensor = torch.stack([torch.stack(theta_phi_normal_list[i], dim=0) for i in range(frame_count)], dim=0)
        theta_phi_reflection_tensor = torch.stack([torch.stack(theta_phi_reflection_list[i], dim=0) for i in range(frame_count)], dim=0)

        #theta2 -> [3,n_sample_per_level] phi2 ->[3,n_sample_per_level]
    else:
        weight_list, xyz_list, theta_phi_normal_list, theta_phi_reflection_list ,view_theta_list = all_params_list
        frame_count = len(xyz_list)
        theta_phi_normal_tensor = torch.stack(
            [torch.stack(theta_phi_normal_list[i], dim=0) for i in range(frame_count)], dim=0)
        theta_phi_reflection_tensor = torch.stack(
            [torch.stack(theta_phi_reflection_list[i], dim=0) for i in range(frame_count)], dim=0)

    final_result = torch.empty((5,0), device=device)

    view_theta = view_theta_list[0]
    view_theta2 = view_theta_list[1]

    for frame_idx in range(frame_count):


        #e.g. coefficient be in shape [5,2,96], theta2 phi2 be in shape [ 2, n_sample_per_level * 3]
        # to compute outerproduct, we need coef.unsqueeze(-1)   and theta_phi.unsqueeze(0).unsqueeze(2)
        # then we have [5,2,96,n_sample_per_level * 3]
        non_const_coef = coef_table[:, 1:, frame_idx * n_sample_per_frame: frame_idx * n_sample_per_frame + n_sample_per_frame]
        constant_tmp = coef_table[:,0,frame_idx * n_sample_per_frame: frame_idx * n_sample_per_frame + n_sample_per_frame]
        constant_tmp = constant_tmp.repeat(1,n_sample_per_level)

        parameter = stack_parameter(view_option_str,view_theta,view_theta2, theta_phi_normal_tensor, theta_phi_reflection_tensor, frame_idx, use_reflection_parameterization)

        outer = non_const_coef.unsqueeze(-1) * parameter.unsqueeze(0).unsqueeze(2)

        #order is dir0-frame0 -> dir1-frame0 -> dir0-frame1 -> dir1-frame1...
        outer_T = torch.transpose(outer,2,3)

        #outer = outer.view((5,2,n_sample_per_frame * n_sample_per_level))
        outer_T = outer_T.reshape((5,parameter.shape[0],n_sample_per_level * n_sample_per_frame))


        test_sum = torch.sum(outer_T, dim=1) + constant_tmp

        final_result = torch.cat((final_result,test_sum),dim=-1)


    #exclude those dir/frame pair that has a 0 weight
    #weight list convert to [3,n_sample_per_level]
    frame_weight_tensor = torch.stack(weight_list,dim=0).flatten()
    frame_weight_tensor = torch.repeat_interleave(frame_weight_tensor, n_sample_per_frame, dim=0)

    valid_idx = (frame_weight_tensor != 0.0)

    frame_weight_tensor = frame_weight_tensor[valid_idx]

    final_result = final_result[:,valid_idx]



    X_tensor = torch.stack( [xyz_list[i][0] for i in range(frame_count)] )
    Y_tensor = torch.stack( [xyz_list[i][1] for i in range(frame_count)] )
    Z_tensor = torch.stack( [xyz_list[i][2] for i in range(frame_count)] )

    X_tensor = X_tensor.view(-1,3)
    Y_tensor = Y_tensor.view(-1,3)
    Z_tensor = Z_tensor.view(-1,3)

    X_tensor = torch.repeat_interleave(X_tensor, n_sample_per_frame, dim=0)
    Y_tensor = torch.repeat_interleave(Y_tensor, n_sample_per_frame, dim=0)
    Z_tensor = torch.repeat_interleave(Z_tensor, n_sample_per_frame, dim=0)


    X_tensor = X_tensor[valid_idx,:]
    Y_tensor = Y_tensor[valid_idx,:]
    Z_tensor = Z_tensor[valid_idx,:]

    direction = final_result[0,:].view(-1,1) * X_tensor + final_result[1,:].view(-1,1) * Y_tensor + final_result[2,:].view(-1,1) * Z_tensor

    max = torch.max(torch.abs(direction), dim=-1).values
    direction_map = direction / max.view(-1,1)

    weight_all = final_result[4,:] * frame_weight_tensor
    level_all = final_result[3,:]

    if not allow_neg_weight:
        weight_all = torch.clip(weight_all, 0)


    # TODO: Jacobian?
    if adjust_level:
        j = 3 / 4 * torch.log2(torch_util.torch_dot_vectorized_2D(direction_map, direction_map))
        adjusted_level_all = level_all + j
    else:
        adjusted_level_all = level_all

    adjusted_level_all = torch.clip(adjusted_level_all, 0.0, 6.0)


    #create sample_idx pattern
    sample_idx_all = torch.arange(0, n_sample_per_level, dtype=torch.int, device=device)
    sample_idx_all = sample_idx_all.repeat(frame_count)
    sample_idx_all = torch.repeat_interleave(sample_idx_all, n_sample_per_frame, dim = 0)
    sample_idx_all = sample_idx_all[valid_idx]

    #from [n_sample_per_level] to n_sampler_per_level * n_sample_per_frame * frame_count]
    all_normal_location = normal_directions.repeat(frame_count,1)
    all_normal_location = torch.repeat_interleave(all_normal_location, n_sample_per_frame, dim = 0)
    all_normal_location = all_normal_location[valid_idx]
    direction = direction / torch.linalg.norm(direction,dim=-1,keepdim=True)

    all_NdotL = torch.sum(all_normal_location * direction, dim = -1)


    # xyz_list -> X -> [3,n_sample_per_level,3] Y->[3,n_sample_per_level,3] Z->[3,n_sample_per_level,3]

    result = torch_util_all_locations.compute_contribution_full(direction_map, adjusted_level_all, weight_all,
                                                                sample_idx_all, n_level=7,
                                                                n_sample_per_level=n_sample_per_level)

    #Compute below horizon directions, simply return the cosine



    return result, all_NdotL









def multiple_texel_full_optimization_vectorized_vec_prepare(n_sample_per_frame, n_sample_per_level, ggx_ref_list, all_precomputed_info, coef_table=None, constant= False, adjust_level = False, allow_neg_weight = False, device = torch.device('cpu')):
    if constant:
        raise NotImplementedError

    weight_list, xyz_list, theta_phi_list = all_precomputed_info

    #First use coefficient to compute...

    #coefficient [5,3,n_sample_per_frame * 3]

    #theta,phi parameter list([list(theta[200],phi,theta2,phi2)](frame0),[],[])

    #theta and phi convert to tensor of size [3,4,n_sample_per_level]
    theta_phi_tensor = torch.stack([torch.stack(theta_phi_list[i], dim=0) for i in range(3)], dim=0)

    #theta2 -> [3,n_sample_per_level] phi2 ->[3,n_sample_per_level]

    final_result = torch.empty((5,0),device=device)


    for frame_idx in range(3):
        theta2 = theta_phi_tensor[frame_idx,2,:]
        phi2 = theta_phi_tensor[frame_idx,3,:]

        #e.g. coefficient be in shape [5,2,96], theta2 phi2 be in shape [ 2, n_sample_per_level * 3]
        # to compute outerproduct, we need coef.unsqueeze(-1)   and theta_phi.unsqueeze(0).unsqueeze(2)
        # then we have [5,2,96,n_sample_per_level * 3]
        non_const_coef = coef_table[:, 1:, frame_idx * n_sample_per_frame: frame_idx * n_sample_per_frame + n_sample_per_frame]
        constant_tmp = coef_table[:,0,frame_idx * n_sample_per_frame: frame_idx * n_sample_per_frame + n_sample_per_frame]
        constant_tmp = constant_tmp.repeat(1,n_sample_per_level)

        parameter = torch.stack((theta2,phi2))

        outer = non_const_coef.unsqueeze(-1) * parameter.unsqueeze(0).unsqueeze(2)

        #order is dir0-frame0 -> dir1-frame0 -> dir0-frame1 -> dir1-frame1...
        outer_T = torch.transpose(outer,2,3)

        #outer = outer.view((5,2,n_sample_per_frame * n_sample_per_level))
        outer_T = outer_T.reshape((5,2,n_sample_per_level * n_sample_per_frame))


        test_sum = outer_T[:,0] + outer_T[:,1] + constant_tmp

        final_result = torch.cat((final_result,test_sum),dim=-1)


    #exclude those dir/frame pair that has a 0 weight
    #weight list convert to [3,n_sample_per_level]
    frame_weight_tensor = torch.stack(weight_list,dim=0).flatten()
    frame_weight_tensor = torch.repeat_interleave(frame_weight_tensor, n_sample_per_frame, dim=0)

    valid_idx = (frame_weight_tensor != 0.0)

    frame_weight_tensor = frame_weight_tensor[valid_idx]

    final_result = final_result[:,valid_idx]



    X_tensor = torch.stack( [xyz_list[i][0] for i in range(3)] )
    Y_tensor = torch.stack( [xyz_list[i][1] for i in range(3)] )
    Z_tensor = torch.stack( [xyz_list[i][2] for i in range(3)] )

    X_tensor = X_tensor.view(-1,3)
    Y_tensor = Y_tensor.view(-1,3)
    Z_tensor = Z_tensor.view(-1,3)

    X_tensor = torch.repeat_interleave(X_tensor, n_sample_per_frame, dim=0)
    Y_tensor = torch.repeat_interleave(Y_tensor, n_sample_per_frame, dim=0)
    Z_tensor = torch.repeat_interleave(Z_tensor, n_sample_per_frame, dim=0)


    X_tensor = X_tensor[valid_idx,:]
    Y_tensor = Y_tensor[valid_idx,:]
    Z_tensor = Z_tensor[valid_idx,:]

    direction = final_result[0,:].view(-1,1) * X_tensor + final_result[1,:].view(-1,1) * Y_tensor + final_result[2,:].view(-1,1) * Z_tensor

    max = torch.max(torch.abs(direction), dim=-1).values
    direction_map = direction / max.view(-1,1)

    weight_all = final_result[4,:] * frame_weight_tensor
    level_all = final_result[3,:]

    if not allow_neg_weight:
        weight_all = torch.clip(weight_all, 0)


    # TODO: Jacobian?
    if adjust_level:
        j = 3 / 4 * torch.log2(torch_util.torch_dot_vectorized_2D(direction_map, direction_map))
        adjusted_level_all = level_all + j
    else:
        adjusted_level_all = level_all

    adjusted_level_all = torch.clip(adjusted_level_all, 0.0, 6.0)


    #create sample_idx pattern
    sample_idx_all = torch.arange(0, n_sample_per_level, dtype=torch.int, device=device)
    sample_idx_all = sample_idx_all.repeat(3)
    sample_idx_all = torch.repeat_interleave(sample_idx_all, n_sample_per_frame, dim = 0)
    sample_idx_all = sample_idx_all[valid_idx]




    # xyz_list -> X -> [3,n_sample_per_level,3] Y->[3,n_sample_per_level,3] Z->[3,n_sample_per_level,3]

    result = torch_util_all_locations.compute_contribution_full(direction_map, adjusted_level_all, weight_all,
                                                                sample_idx_all, n_level=7,
                                                                n_sample_per_level=n_sample_per_level)

    return result




def multiple_texel_full_optimization_vectorized(n_sample_per_frame, n_sample_per_level, ggx_ref_list, weight_list, xyz_list, theta_phi_list, coef_table=None, constant= False, adjust_level = False, allow_neg_weight = False, device = torch.device('cpu')):
    """

    :param n_sample_per_frame:
    :param n_sample_per_level:
    :param ggx_ref_list:
    :param weight_list:
    :param xyz_list:
    :param theta_phi_list:
    :param coef_table:
    :param constant:
    :param adjust_level:
    :param device:
    :return:
    """
    if coef_table is None:
        # a single coefficient table in shape [5,3,nsample_per_frame * 3]

        coefficient_table = torch.rand((5, 3, n_sample_per_frame * 3))
    else:
        coefficient_table = coef_table

    sample_directions = torch.empty((0, 3), device=device)
    sample_weights = torch.empty((0,), device=device)
    sample_levels = torch.empty((0,), device=device)
    sample_idx = torch.empty((0,), device=device,dtype=torch.int)


    for dir_idx in range(n_sample_per_level):
        for i in range(3):
            X, Y, Z = xyz_list[i][0][dir_idx],xyz_list[i][1][dir_idx],xyz_list[i][2][dir_idx]
            theta2, phi2 = theta_phi_list[i][2][dir_idx],theta_phi_list[i][3][dir_idx]
            frame_weight = weight_list[i][dir_idx]

            if frame_weight == 0:
                continue

            coeff_start = i * n_sample_per_frame
            coeff_end = coeff_start + n_sample_per_frame
            if not constant:
                coeff_x_table = coefficient_table[0, :, coeff_start:coeff_end]
                coeff_y_table = coefficient_table[1, :, coeff_start:coeff_end]
                coeff_z_table = coefficient_table[2, :, coeff_start:coeff_end]
                coeff_level_table = coefficient_table[3, :, coeff_start:coeff_end]
                coeff_weight_table = coefficient_table[4, :, coeff_start:coeff_end]

                coeff_x = coeff_x_table[0] + coeff_x_table[1] * theta2 + coeff_x_table[2] * phi2
                coeff_y = coeff_y_table[0] + coeff_y_table[1] * theta2 + coeff_y_table[2] * phi2
                coeff_z = coeff_z_table[0] + coeff_z_table[1] * theta2 + coeff_z_table[2] * phi2


                level = coeff_level_table[0] + coeff_level_table[1] * theta2 + coeff_level_table[2] * phi2
                weight = coeff_weight_table[0] + coeff_weight_table[1] * theta2 + coeff_weight_table[2] * phi2
            else:
                coeff_x = coefficient_table[0, coeff_start:coeff_end]
                coeff_y = coefficient_table[1, coeff_start:coeff_end]
                coeff_z = coefficient_table[2, coeff_start:coeff_end]
                level = coefficient_table[3, coeff_start:coeff_end]
                weight = coefficient_table[4, coeff_start:coeff_end]

            coeff_x = torch.stack((coeff_x, coeff_x, coeff_x), dim=-1)
            coeff_y = torch.stack((coeff_y, coeff_y, coeff_y), dim=-1)
            coeff_z = torch.stack((coeff_z, coeff_z, coeff_z), dim=-1)


            sample_direction = coeff_x * X + coeff_y * Y + coeff_z * Z
            abs_direction = torch.abs(sample_direction)
            max_dir = torch.max(abs_direction, dim=-1).values
            sample_direction_map = sample_direction / torch.stack([max_dir, max_dir, max_dir], dim=-1)

            weight_cur_frame = weight * frame_weight

            #TODO: how to make weight/level always positive?
            # What if we do not clip weight?
            if not allow_neg_weight:
                weight_cur_frame = torch.clip(weight_cur_frame, 0)


            sample_directions = torch.concatenate((sample_directions, sample_direction_map))
            sample_weights = torch.concatenate((sample_weights, weight_cur_frame))
            sample_idx = torch.concatenate((sample_idx, torch.full([n_sample_per_frame],dir_idx,device=device,dtype=torch.int)))

            #TODO: Jacobian?
            if adjust_level:
                j = 3 / 4 * torch.log2(torch_util.torch_dot_vectorized_2D(sample_direction_map, sample_direction_map))
                adjusted_level = level + j
            else:
                adjusted_level = level

            adjusted_level = torch.clip(adjusted_level,0.0,6.0)

            sample_levels = torch.concatenate((sample_levels, adjusted_level))


    result = torch_util_all_locations.compute_contribution_full(sample_directions,sample_levels,sample_weights,sample_idx,n_level=7,n_sample_per_level = n_sample_per_level)





    return result

def multiple_texel_full_optimization(texel_dirs, n_sample_per_frame, n_sample_per_level, ggx_ref_list, weight_list, xyz_list, theta_phi_list, coef_table=None, constant= False, adjust_level = False, allow_neg_weight = False, device = torch.device('cpu')):
    """

    :param texel_dirs:
    :param n_sample_per_frame: how many samples we take to approximate a texel
    :param n_sample_per_level: how many texels we use to measure error
    :param ggx_ref_list:
    :param weight_list: a list with three element, each contains the weight of all directions for this frame
    :param xyz_list: the same as weight_list
    :param theta_phi_list: the same theta_phi
    :param coef_table:
    :param constant:
    :return:
    """
    if coef_table is None:
        # a single coefficient table in shape [5,3,nsample_per_frame * 3]

        coefficient_table = torch.rand((5, 3, n_sample_per_frame * 3))
    else:
        coefficient_table = coef_table


    result_list = []
    e_list = []
    for dir_idx in range(n_sample_per_level):
        #start_time = time.time()


        #texel_direction = texel_dirs[dir_idx,:]
        sample_directions = torch.empty((0, 3),device=device)
        sample_weights = torch.empty((0,),device=device)
        sample_levels = torch.empty((0,),device=device)
        for i in range(3):
            X, Y, Z = xyz_list[i][0][dir_idx],xyz_list[i][1][dir_idx],xyz_list[i][2][dir_idx]
            theta2, phi2 = theta_phi_list[i][2][dir_idx],theta_phi_list[i][3][dir_idx]
            frame_weight = weight_list[i][dir_idx]

            if frame_weight == 0:
                continue

            coeff_start = i * n_sample_per_frame
            coeff_end = coeff_start + n_sample_per_frame
            if not constant:
                coeff_x_table = coefficient_table[0, :, coeff_start:coeff_end]
                coeff_y_table = coefficient_table[1, :, coeff_start:coeff_end]
                coeff_z_table = coefficient_table[2, :, coeff_start:coeff_end]
                coeff_level_table = coefficient_table[3, :, coeff_start:coeff_end]
                coeff_weight_table = coefficient_table[4, :, coeff_start:coeff_end]

                coeff_x = coeff_x_table[0] + coeff_x_table[1] * theta2 + coeff_x_table[2] * phi2
                coeff_y = coeff_y_table[0] + coeff_y_table[1] * theta2 + coeff_y_table[2] * phi2
                coeff_z = coeff_z_table[0] + coeff_z_table[1] * theta2 + coeff_z_table[2] * phi2


                level = coeff_level_table[0] + coeff_level_table[1] * theta2 + coeff_level_table[2] * phi2
                weight = coeff_weight_table[0] + coeff_weight_table[1] * theta2 + coeff_weight_table[2] * phi2
            else:
                coeff_x = coefficient_table[0, coeff_start:coeff_end]
                coeff_y = coefficient_table[1, coeff_start:coeff_end]
                coeff_z = coefficient_table[2, coeff_start:coeff_end]
                level = coefficient_table[3, coeff_start:coeff_end]
                weight = coefficient_table[4, coeff_start:coeff_end]

            coeff_x = torch.stack((coeff_x, coeff_x, coeff_x), dim=-1)
            coeff_y = torch.stack((coeff_y, coeff_y, coeff_y), dim=-1)
            coeff_z = torch.stack((coeff_z, coeff_z, coeff_z), dim=-1)


            sample_direction = coeff_x * X + coeff_y * Y + coeff_z * Z
            abs_direction = torch.abs(sample_direction)
            max_dir = torch.max(abs_direction, dim=-1).values
            sample_direction_map = sample_direction / torch.stack([max_dir, max_dir, max_dir], dim=-1)

            weight_cur_frame = weight * frame_weight

            #TODO: how to make weight/level always positive?
            if not allow_neg_weight:
                weight_cur_frame = torch.clip(weight_cur_frame, 0)


            sample_directions = torch.concatenate((sample_directions, sample_direction_map))
            sample_weights = torch.concatenate((sample_weights, weight_cur_frame))

            #TODO: Jacobian?
            if adjust_level:
                j = 3 / 4 * torch.log2(torch_util.torch_dot_vectorized_2D(sample_direction_map, sample_direction_map))
                level += j

            adjusted_level = torch.clip(level,0.0,6.0)

            sample_levels = torch.concatenate((sample_levels, adjusted_level))

        # if dir_idx == 0:
        #     end_time = time.time()
        #     elapsed_time = end_time - start_time
        #     print(f"one pushback preparation took {elapsed_time:.4f} seconds to execute.")


        #start_time = time.time()

        result = torch_util.compute_contribution(sample_directions, sample_levels, sample_weights, 7)

        # if dir_idx == 0:
        #     end_time = time.time()
        #     elapsed_time = end_time - start_time
        #     print(f"one pushing back took {elapsed_time:.4f} seconds to execute.")

        result_list.append(result)
        e_arr = L1_error_one_texel(ggx_ref_list[dir_idx], result)
        e = torch.sum(e_arr)
        # e = np.average(e_arr)
        e_list.append(e)

    return e_list, result_list



def process_input(dir_idx, stream ,results, sample_directions, sample_levels, sample_weights, ggx_ref):
    with torch.cuda.stream(stream):
        result = torch_util.compute_contribution(sample_directions, sample_levels, sample_weights, 7)
        e_arr = L1_error_one_texel(ggx_ref, result)
        e = torch.sum(e_arr)
        # e = np.average(e_arr)
        results[dir_idx] = e



def test_one_texel_full_optimization(texel_direction, n_sample_per_frame, ggx_ref, coef_table=None, constant= False, adjust_level = False, device = torch.device('cpu')):
    """
    How does frame weight work on the optimization part?
    :param texel_direction: the one texel we are testing on optimization
    :param n_sample_per_frame:
    :param ggx_ref: ggx reference kernel of size (6,res,res,1)
    :return:
    """
    if coef_table is None:
        # a single coefficient table in shape [5,3,nsample_per_frame * 3]

        coefficient_table = torch.rand((5, 3, n_sample_per_frame * 3))
    else:
        coefficient_table = coef_table

    # frame_weight:
    frame_weight_list = []
    theta_phi_list = []
    unnormalized_texel_direction = texel_direction / torch.max(torch.abs(texel_direction))
    for i in range(3):
        frame_weight_list.append(torch_util.torch_gen_frame_weight(unnormalized_texel_direction, i))
        theta_phi_list.append(torch_util.torch_gen_theta_phi(unnormalized_texel_direction, i))

    sample_directions = torch.empty((0,3), device=device)
    sample_weights = torch.empty((0,), device=device)
    sample_levels = torch.empty((0,), device=device)


    for i in range(3):
        cur_frame_weight = frame_weight_list[i]

        #if frame_weight is 0, we skip, how to do this in parallel?
        if cur_frame_weight == 0:
            continue

        X, Y, Z = torch_util.torch_gen_frame_xyz(texel_direction, i)

        _, _, theta2, phi2 = theta_phi_list[i]

        coeff_start = i * n_sample_per_frame
        coeff_end = coeff_start + n_sample_per_frame



        if not constant:
            coeff_x_table = coefficient_table[0, :, coeff_start:coeff_end]
            coeff_y_table = coefficient_table[1, :, coeff_start:coeff_end]
            coeff_z_table = coefficient_table[2, :, coeff_start:coeff_end]
            coeff_level_table = coefficient_table[3, :, coeff_start:coeff_end]
            coeff_weight_table = coefficient_table[4, :, coeff_start:coeff_end]

            coeff_x = coeff_x_table[0] + coeff_x_table[1] * theta2 + coeff_x_table[2] * phi2
            coeff_y = coeff_y_table[0] + coeff_y_table[1] * theta2 + coeff_y_table[2] * phi2
            coeff_z = coeff_z_table[0] + coeff_z_table[1] * theta2 + coeff_z_table[2] * phi2


            level = coeff_level_table[0] + coeff_level_table[1] * theta2 + coeff_level_table[2] * phi2
            weight = coeff_weight_table[0] + coeff_weight_table[1] * theta2 + coeff_weight_table[2] * phi2
        else:
            coeff_x = coefficient_table[0, coeff_start:coeff_end]
            coeff_y = coefficient_table[1, coeff_start:coeff_end]
            coeff_z = coefficient_table[2, coeff_start:coeff_end]
            level = coefficient_table[3, coeff_start:coeff_end]
            weight = coefficient_table[4, coeff_start:coeff_end]

        coeff_x = torch.stack((coeff_x, coeff_x, coeff_x), dim=-1)
        coeff_y = torch.stack((coeff_y, coeff_y, coeff_y), dim=-1)
        coeff_z = torch.stack((coeff_z, coeff_z, coeff_z), dim=-1)


        sample_direction = coeff_x * X + coeff_y * Y + coeff_z * Z
        abs_direction = torch.abs(sample_direction)
        max_dir = torch.max(abs_direction, dim=-1).values
        sample_direction_map = sample_direction / torch.stack([max_dir, max_dir, max_dir], dim=-1)

        weight_cur_frame = weight * frame_weight_list[i]

        #TODO: how to make weight/level always positive?
        weight_cur_frame = torch.clip(weight_cur_frame, 0)
        level = torch.clip(level,0.0,6.0)

        sample_directions = torch.concatenate((sample_directions, sample_direction_map))
        sample_weights = torch.concatenate((sample_weights, weight_cur_frame))

        #TODO: Jacobian?
        if adjust_level:
            j = 3 / 4 * torch.log2(torch_util.torch_dot_vectorized_2D(sample_direction_map, sample_direction_map))
            level += j

        sample_levels = torch.concatenate((sample_levels, level))


    result = torch_util.compute_contribution(sample_directions, sample_levels, sample_weights, 7)

    e_arr = L1_error_one_texel(ggx_ref, result)

    e = torch.sum(e_arr)
    #e = np.average(e_arr)

    return e, result


def precompute_opt_info_view_dependent(texel_directions,n_sample_per_level,view_directions,view_thetas, use_reflected_dirs_as_parameter, view_option_str:str, device = torch.device('cpu')):

    texel_directions_normalized = texel_directions / torch.linalg.norm(texel_directions, dim=-1, keepdim=True)

    if view_option_str[:14] != "relative_frame":
        weight_per_frame = []
        xyz_per_frame = []
        theta_phi_normal_per_frame = []
        theta_phi_reflection_per_frame = []

        for frame_idx in range(3):

            frame_xyz = torch_util.torch_gen_frame_xyz_view_dependent(texel_directions_normalized, frame_idx, view_directions)
            frame_weight = torch_util.torch_gen_frame_weight(frame_xyz[-1], frame_idx)

            frame_reflection_theta_phi = torch_util.torch_gen_theta_phi(frame_xyz[-1],frame_idx)

            frame_normal_theta_phi = torch_util.torch_gen_theta_phi(texel_directions, frame_idx)

            weight_per_frame.append(frame_weight)
            xyz_per_frame.append(frame_xyz)
            theta_phi_normal_per_frame.append(frame_normal_theta_phi)
            theta_phi_reflection_per_frame.append(frame_reflection_theta_phi)

        view_theta_list = (view_thetas,view_thetas**2)

        return weight_per_frame, xyz_per_frame, theta_phi_normal_per_frame,theta_phi_reflection_per_frame, view_theta_list
    else:
        weight_per_frame = []
        xyz_per_frame = []
        theta_phi_normal_per_frame = []
        theta_phi_reflection_per_frame = []
        weight_per_frame.append(torch.ones(texel_directions.shape[0],device=device))
        frame_xyz = torch_util.torch_gen_anisotropic_frame_xyz(texel_directions_normalized,view_directions)
        xyz_per_frame.append(frame_xyz)
        frame_normal_theta_phi = torch_util.torch_gen_theta_phi_no_frame(texel_directions)
        theta_phi_normal_per_frame.append(frame_normal_theta_phi)
        frame_reflected_theta_phi = torch_util.torch_gen_theta_phi_no_frame(frame_xyz[-1])
        theta_phi_reflection_per_frame.append(frame_reflected_theta_phi)

        view_theta_list = (view_thetas,view_thetas**2)

        return weight_per_frame,xyz_per_frame, theta_phi_normal_per_frame, theta_phi_reflection_per_frame, view_theta_list


def precompute_opt_info(texel_directions, n_sample_per_level):
    """
    Given the texel directions, precompute XYZ,frame weight and theta_phi
    :param texel_directions: assume the direction is already normalized to the unit cube [N,3]
    :param n_sample_per_level:
    :return:
    """

    weight_per_frame = []
    xyz_per_frame = []
    theta_phi_per_frame = []

    for frame_idx in range(3):
        frame_weight = torch_util.torch_gen_frame_weight(texel_directions,frame_idx)
        frame_xyz = torch_util.torch_gen_frame_xyz(texel_directions,frame_idx)
        frame_theta_phi = torch_util.torch_gen_theta_phi(texel_directions,frame_idx)

        weight_per_frame.append(frame_weight)
        xyz_per_frame.append(frame_xyz)
        theta_phi_per_frame.append(frame_theta_phi)

    return weight_per_frame, xyz_per_frame, theta_phi_per_frame


def visualize_high_loss_view_dependent(pushed_back_param, ref_param, threshold:float, view_theta):
    (n_sample_per_frame, n_sample_per_level, ref_list,all_precomputed_info, params, adjust_level, allow_neg_weight,
     view_option_str,view_reflection_parameterization, device, all_locations) = pushed_back_param
    ggx_alpha, all_locations,tex_directions_res, tex_directions_res_map,view_dirs, ggx_ref_jac_weight, view_ndf_clipping = ref_param

    ref_list = compute_ggx_ndf_ref_view_dependent_torch_vectorized(ggx_alpha, all_locations,
                                                                   tex_directions_res, tex_directions_res_map,
                                                                   view_dirs, ggx_ref_jac_weight, view_ndf_clipping)
    ref_list_normalized = ref_list / torch.sum(ref_list, dim=(1, 2, 3, 4), keepdim=True)

    tmp_pushed_back_result, all_NdotL = multiple_texel_full_optimization_view_dependent_vectorized_vec_prepare(
        n_sample_per_frame, n_sample_per_level, ref_list,
        all_precomputed_info, params, adjust_level, allow_neg_weight, view_option_str,
        view_reflection_parameterization, device, all_locations)

    tmp_pushed_back_sum = torch.sum(tmp_pushed_back_result, dim=[1, 2, 3, 4], keepdim=True)
    tmp_pushed_back_result_normalized = tmp_pushed_back_result /  (tmp_pushed_back_sum + 1e-7)
    diff = torch.abs(ref_list_normalized - tmp_pushed_back_result_normalized)
    diff = torch.sum(diff, dim=[1, 2, 3, 4])

    high_idx = diff > threshold

    ref_high = ref_list[high_idx]
    ref_list_normalized_high = ref_list_normalized[high_idx]
    pushed_back_result_high = tmp_pushed_back_result[high_idx]
    pushed_back_result_high_normalized = tmp_pushed_back_result_normalized[high_idx]

    count = ref_list_normalized_high.shape[0]
    for i in range(count):
        ref_cube = ref_list_normalized_high[i]
        pushed_back_cube = pushed_back_result_high_normalized[i]
        res = 128

        ref_cube = ref_cube.repeat([1,1,1,3])
        pushed_back_cube = pushed_back_cube.repeat([1,1,1,3])

        ref_cube = ref_cube.detach().numpy()
        pushed_back_cube = pushed_back_cube.detach().numpy()

        prefix = "./high_loss_visualization/"

        ref_file_name = prefix + "{}_ref{:.3f}".format(i,ggx_alpha) + ".exr"
        pushed_back_file_name = prefix + "{}_pushed_back{:.3f}".format(i,ggx_alpha) + ".exr"

        image_read.gen_cubemap_preview_image(ref_cube,res,filename = ref_file_name)
        image_read.gen_cubemap_preview_image(pushed_back_cube,res,filename = pushed_back_file_name)




def optimize_multiple_locations(n_sample_per_level, constant, n_sample_per_frame, ggx_alpha = 0.1, adjust_level = False,
                                vectorize = True, optimizer_type = "adam", random_shuffle = False, allow_neg_weight = False,
                                ggx_ref_jac_weight = 'None', learning_rate = 1e-4, view_option_str = "None"
                                ,view_reflection_parameterization = False, view_ndf_clipping = False
                                ,use_vndf = False, fixed_cos_view_theta = None):
    """
    To speed up, a lot of things can be precomputed, including the relative XYZ,the frame weight_g
    we don't have to compute this in every iteration
    :param n_sample_per_level:
    :return:
    """
    visualize_loss = True

    device = get_device()

    view_dependent, view_option_str = process_view_option(view_option_str, view_reflection_parameterization)

    view_model_dict = torch_util.create_view_model_dict()

    logger = logging.getLogger(__name__)

    log_name = map_util.log_filename(ggx_alpha,n_sample_per_frame,n_sample_per_level,constant,adjust_level,optimizer_type,
                                     random_shuffle=random_shuffle, allow_neg_weight=allow_neg_weight,
                                     ggx_ref_jac_weight=ggx_ref_jac_weight,
                                     view_dependent=view_dependent, view_option_str=view_option_str
                                     ,reflection_parameterization=view_reflection_parameterization,
                                     view_ndf_clipping=view_ndf_clipping, use_vndf=use_vndf,
                                     fixed_cos_view_theta=fixed_cos_view_theta.item() if fixed_cos_view_theta is not None else None)

    if fixed_cos_view_theta is not None:
        fixed_cos_dir_name = map_util.fixed_view_diretory_name(
            ggx_alpha, n_sample_per_frame, n_sample_per_level, constant, adjust_level, optimizer_type,
            random_shuffle=random_shuffle, allow_neg_weight=allow_neg_weight,
            ggx_ref_jac_weight=ggx_ref_jac_weight,
            view_dependent=view_dependent, view_option_str=view_option_str
            , reflection_parameterization=view_reflection_parameterization,
            view_ndf_clipping=view_ndf_clipping, use_vndf=use_vndf, fixed_cos_view_theta=fixed_cos_view_theta.item(),
            fixed_cos_view_theta_res=fixed_costheta_res_g
        )
        fixed_cos_dir_name = fixed_cos_dir_name + "/"

        #test if directory exist
        if not os.path.exists("./logs/" + fixed_cos_dir_name):
            os.mkdir("./logs/" + fixed_cos_dir_name)
        if not os.path.exists("./model/" + fixed_cos_dir_name):
            os.mkdir("./model/" + fixed_cos_dir_name)

    else:
        fixed_cos_dir_name = ""

    if fixed_cos_view_theta is not None:
        log_name = "./logs/"+ fixed_cos_dir_name + log_name
    else:
        log_name = "./logs/"+ log_name

    logging.basicConfig(filename=log_name, filemode='a', level=logging.INFO,
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    logger.info("\n\n")

    if not random_shuffle:
        rng = np.random.default_rng()
        dir_name = map_util.dir_filename(ggx_alpha, constant, n_sample_per_frame, n_sample_per_level, adjust_level,
                                         optimizer_type, allow_neg_weight=allow_neg_weight,
                                         ggx_ref_jac_weight=ggx_ref_jac_weight, view_dependent=view_dependent,view_option_str=view_option_str
                                         ,reflection_parameterization=view_reflection_parameterization,
                                         view_ndf_clipping=view_ndf_clipping, use_vndf=use_vndf,
                                         fixed_cos_view_theta=fixed_cos_view_theta.item() if fixed_cos_view_theta is not None else None)
        if view_dependent:
            if fixed_cos_view_theta is None:
                view_dirs, view_theta, all_locations_cube,all_locations = torch_util.sample_view_dependent_location(n_sample_per_level, torch.manual_seed(rng.integers(low=1,high=425124123)), no_parallel= True)
            else:
                view_dirs, view_theta, all_locations_cube, all_locations = torch_util.sample_view_dependent_location_fixed_view_theta(
                    n_sample_per_level, torch.manual_seed(rng.integers(low=1, high=425124123)), no_parallel=True,fixed_cos_view_theta=fixed_cos_view_theta)
            torch.save(torch.concatenate(view_dirs,all_locations_cube),dir_name)
        else:
            all_locations = map_util.sample_location(n_sample_per_level,rng)
            all_locations_cube = all_locations / np.max(all_locations,axis=1,keepdims=True)
            torch.save(torch.from_numpy(all_locations), dir_name)

    else:
        rng = torch.Generator(device=device)
        rng.seed()
        #rng.manual_seed(12345)
        if view_dependent:
            if fixed_cos_view_theta is None:
                view_dirs, view_theta, all_locations_cube, all_locations = torch_util.sample_view_dependent_location(n_sample_per_level,
                                                                                                         rng, no_parallel= True)
            else:
                view_dirs, view_theta, all_locations_cube, all_locations = torch_util.sample_view_dependent_location_fixed_view_theta(n_sample_per_level,
                                                                                                         rng, no_parallel= False,fixed_cos_view_theta=fixed_cos_view_theta)
        else:
            all_locations_cube,all_locations = torch_util.sample_location(n_sample_per_level,rng)


    model_name = map_util.model_filename(ggx_alpha,constant,n_sample_per_frame,n_sample_per_level,adjust_level,optimizer_type,
                                         random_shuffle=random_shuffle, allow_neg_weight=allow_neg_weight,
                                         ggx_ref_jac_weight=ggx_ref_jac_weight,
                                         view_dependent=view_dependent, view_option_str=view_option_str,
                                         reflection_parameterization=view_reflection_parameterization,
                                         view_ndf_clipping=view_ndf_clipping,use_vndf=use_vndf,
                                         fixed_cos_view_theta=fixed_cos_view_theta.item() if fixed_cos_view_theta is not None else None)


    if fixed_cos_view_theta is not None:
        model_name = fixed_cos_dir_name + model_name



    if view_dependent:
        model = (view_model_dict[view_option_str])(n_sample_per_frame)
    else:
        if not constant:
            model = torch_util.QuadModel(n_sample_per_frame)
        else:
            model = torch_util.ConstantModel(n_sample_per_frame)


    if os.path.exists("./model/" + model_name):
        logger.info("Read model from dict")
        model.load_state_dict(torch.load("./model/" + model_name,map_location=device))


    model.to(device)

    tex_directions_res_map = map_util.texel_directions(128).astype(np.float32)
    tex_directions_res = mat_util.normalized(tex_directions_res_map)
    # stack this for N times, is this necessary?
    tex_directions_res_map = np.tile(tex_directions_res_map, (n_sample_per_level, 1, 1, 1, 1))
    tex_directions_res = np.tile(tex_directions_res, (n_sample_per_level, 1, 1, 1, 1))
    tex_directions_res_map = torch.from_numpy(tex_directions_res_map).to(device)
    tex_directions_res = torch.from_numpy(tex_directions_res).to(device)


    mipmaps = reference.get_synthetic_mipmap(np.array([0, 0, 1]), 128)


    if view_dependent:
        if use_vndf:
            ref_list_global = compute_ggx_vndf_ref_view_dependent_torch_vectorized(ggx_alpha, all_locations,
                                                                                  tex_directions_res,
                                                                                  tex_directions_res_map,
                                                                                  view_dirs, ggx_ref_jac_weight)
        else:
            ref_list_global = compute_ggx_ndf_ref_view_dependent_torch_vectorized(ggx_alpha, all_locations,
                                                                                  tex_directions_res,
                                                                                  tex_directions_res_map,
                                                                                  view_dirs, ggx_ref_jac_weight)
    else:
        ref_list_global = compute_ggx_ndf_reference_half_vector_torch_vectorized(128, ggx_alpha, all_locations, tex_directions_res, tex_directions_res_map, ggx_ref_jac_weight)







    #view_cos_theta_global = torch.sum(view_dirs * all_locations, dim=1)
    if view_dependent:
        #weight_per_frame_global, xyz_per_frame_global, theta_phi_normal_per_frame_global, theta_phi_reflection_per_frame_global, view_theta_list_global
        all_precomputed_info_global = precompute_opt_info_view_dependent(all_locations_cube,n_sample_per_level, view_dirs, view_theta, view_reflection_parameterization, view_option_str, device=device)

    else:
        #weight_per_frame_global, xyz_per_frame_global, theta_phi_per_frame_global =
        all_precomputed_info_global = precompute_opt_info(all_locations_cube, n_sample_per_level)






    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr= learning_rate)
    elif optimizer_type == "bfgs":
        #optimizer = optim.LBFGS(model.parameters(), lr = 0.7, line_search_fn="strong_wolfe", max_iter = 50, tolerance_grad = 1e-8, tolerance_change = 1e-10)
        optimizer = FullBatchLBFGS(model.parameters(), lr = learning_rate, history_size=120,line_search="Wolfe")
    else:
        raise ValueError("Unknown optimizer type")
    n_epoch = 1000000



    def closure():
        optimizer.zero_grad()
        params = model()

        # start_time = time.time()

        if random_shuffle:
            if not view_dependent:
                all_locations_cube,all_locations =  torch_util.sample_location(n_sample_per_level,rng)
            else:
                if fixed_cos_view_theta is None:
                    view_dirs, view_theta, all_locations_cube, all_locations = torch_util.sample_view_dependent_location(
                        n_sample_per_level,
                        rng, no_parallel=True)
                else:
                    view_dirs, view_theta, all_locations_cube, all_locations = torch_util.sample_view_dependent_location_fixed_view_theta(
                        n_sample_per_level,
                        rng, no_parallel=False, fixed_cos_view_theta=fixed_cos_view_theta)
            #new parameter

            #new reference
            if not view_dependent:
                all_precomputed_info = precompute_opt_info(all_locations_cube,n_sample_per_level)
                ref_list = compute_ggx_ndf_reference_half_vector_torch_vectorized(128, ggx_alpha, all_locations, tex_directions_res, tex_directions_res_map, ggx_ref_jac_weight)
            else:
                if use_vndf:
                    ref_list = compute_ggx_vndf_ref_view_dependent_torch_vectorized(ggx_alpha, all_locations,
                                                                                   tex_directions_res, tex_directions_res_map,
                                                                                   view_dirs, ggx_ref_jac_weight, view_ndf_clipping)
                else:
                    ref_list = compute_ggx_ndf_ref_view_dependent_torch_vectorized(ggx_alpha, all_locations,
                                                                                   tex_directions_res, tex_directions_res_map,
                                                                                   view_dirs, ggx_ref_jac_weight, view_ndf_clipping)
                all_precomputed_info = precompute_opt_info_view_dependent(all_locations_cube,
                            n_sample_per_level, view_dirs, view_theta, view_reflection_parameterization, view_option_str, device=device)


            ref_list = ref_list / torch.sum(ref_list,dim=(1,2,3,4),keepdim=True)

            # ref_list_test = compute_ggx_distribution_reference_torch_vectorized(128, ggx_alpha, all_locations,
            #                                                                tex_directions_res, tex_directions_res_map,
            #                                                                False)
            #
            # ref_list_test = ref_list_test.reshape(ref_list_test.shape + (1,))
            # ref_list_test = ref_list_test / torch.sum(ref_list_test,dim=(1,2,3,4),keepdim=True)
            #
            # ref_list_diff = ref_list - ref_list_test
            #
            # diff = torch.abs(ref_list_diff)
            # diff = torch.sum(diff,dim=[1,2,3,4])
            # mean_error = torch.mean(diff)
        else:
            ref_list = ref_list_global
            all_precomputed_info = all_precomputed_info_global
        if vectorize:
            if not view_dependent:
                tmp_pushed_back_result = multiple_texel_full_optimization_vectorized_vec_prepare(n_sample_per_frame,
                                                                                     n_sample_per_level, ref_list,
                                                                                     all_precomputed_info, params,
                                                                                     constant, adjust_level, allow_neg_weight, device)

                regularization_term = 0
                # tmp_pushed_back_result = multiple_texel_full_optimization_vectorized(n_sample_per_frame,
                #                                                                      n_sample_per_level, ref_list,
                #                                                                      weight_per_frame, xyz_per_frame,
                #                                                                      theta_phi_per_frame, params,
                #                                                                      constant, adjust_level, allow_neg_weight, device)
            else:
                tmp_pushed_back_result, all_NdotL = multiple_texel_full_optimization_view_dependent_vectorized_vec_prepare(n_sample_per_frame,n_sample_per_level, ref_list,
                                                                                              all_precomputed_info,params,adjust_level,allow_neg_weight,view_option_str,
                                                                                              view_reflection_parameterization,device,all_locations)

                # tmp_pushed_back_result = multiple_texel_full_optimization_view_dependent_vectorized(n_sample_per_frame,n_sample_per_level,
                #                                                                                     ref_list,weight_per_frame,xyz_per_frame,
                #                                                                                     theta_phi_per_frame,view_theta_list,params,constant,
                #                                                                                     adjust_level, allow_neg_weight, view_option_str ,device)

                # We should not clip our result, we will want the full result to be as close as the clipped NDF/VNDF
                # if view_ndf_clipping:
                #     tmp_pushed_back_result = torch_util.clip_below_horizon_part_view_dependent(all_locations,tmp_pushed_back_result,tex_directions_res)
            # normalize pushed_back result, if NdotL is less than zero, which means we have sampled some invalid location, we add penalty
                regularization_term = torch.nn.functional.relu(-all_NdotL)
                reg_lambda = 1 / n_sample_per_frame / n_sample_per_level * 0.7
                regularization_term = torch.sum(regularization_term) * reg_lambda

                if visualize_loss:
                    push_back_param = (n_sample_per_frame,n_sample_per_level, ref_list,
                                    all_precomputed_info,params,adjust_level,allow_neg_weight,view_option_str,
                                    view_reflection_parameterization,device,all_locations)
                    ref_param = (ggx_alpha, all_locations,tex_directions_res, tex_directions_res_map,
                                                view_dirs, ggx_ref_jac_weight, view_ndf_clipping)
                    visualize_high_loss_view_dependent(push_back_param,ref_param,1.0, view_theta)


            tmp_pushed_back_sum = torch.sum(tmp_pushed_back_result, dim=[1, 2, 3, 4], keepdim=True)
            tmp_pushed_back_result /= (tmp_pushed_back_sum + 1e-7)
            diff = torch.abs(ref_list - tmp_pushed_back_result)
            diff = torch.sum(diff, dim=[1, 2, 3, 4])
            mean_error = torch.mean(diff) + regularization_term
        else:
            # if view_dependent:
            #     raise NotImplementedError
            # error_list, result_list = multiple_texel_full_optimization(all_locations, n_sample_per_frame,
            #                                                            n_sample_per_level, ref_list,
            #                                                            weight_per_frame, xyz_per_frame,
            #                                                            theta_phi_per_frame, coef_table=params,
            #                                                            constant=constant,
            #                                                            adjust_level=adjust_level,
            #                                                            device=device)
            #
            # tmp = torch.stack(error_list)
            # mean_error = tmp.mean()
            raise NotImplementedError

        #If we are using the LBFGS from github. Thenwe should not call backward
        #mean_error.backward()
        return mean_error


    if optimizer_type == "adam":
        for i in range(n_epoch):
            loss = closure()
            loss.backward()
            if not torch.isnan(loss):
                optimizer.step()
            else:
                logger.info("NaN loss detected")
                logger.info("Current parameters are", model().detach().numpy())
                # pushed_back_result = multiple_texel_full_optimization_vectorized(n_sample_per_frame,
                #                                                                  n_sample_per_level, ref_list_global,
                #                                                                  weight_per_frame_global, xyz_per_frame_global,
                #                                                                  theta_phi_per_frame_global, model(),
                #                                                                  constant, adjust_level, allow_neg_weight, device)
                # # normalize pushed_back result
                # pushed_back_sum = torch.sum(pushed_back_result, dim=[1, 2, 3, 4], keepdim=True)
                # logger.info("is pushed_back_result NaN? ", torch.isnan(pushed_back_result).any())
                # logger.info("is pushed_back_sum 0?", pushed_back_sum == 0.0)


            logger.info("[it{}]:loss is{}".format(i, loss.item()))

            if i % 500 == 0:
                # normalize result
                # result /= torch.sum(result)
                # visualization.visualize_optim_result(ggx_ref, result)
                logger.info(f"saving model")
                save_model(model, "./model/" + model_name)
                # logger.info(f"[it{i}]Loss: {mean_error.item()}")
                if not view_dependent:
                    synthetic_filter_showcase(model().cpu().detach().numpy(), constant, adjust_level, ggx_alpha,
                                              n_sample_per_frame, 2, n_sample_per_level, optimizer_type, random_shuffle,
                                              allow_neg_weight, ggx_ref_jac_weight, mipmaps, view_dependent,view_option_str,"it" + str(i))



    else:
        for i in range(n_epoch):
            options = {
                "closure" : closure,
            }

            obj,grad,lr,backtracks,clos_evals,grad_evals,desc_dir,fail = optimizer.step(options=options)
            logger.info("[it{}]:loss is{}".format(i, obj.item()))
            if(torch.isnan(obj)):
                logger.info("NaN loss detected in LBFGS, Save last param and terminate!")
                save_model(model, "./model/" + model_name + "_nan")
            if i % 50 == 0:
                logger.info(f"saving model")
                save_model(model, "./model/" + model_name)
                if not view_dependent:
                    synthetic_filter_showcase(model().cpu().detach().numpy(), constant, adjust_level, ggx_alpha,
                                          n_sample_per_frame, 2, n_sample_per_level, optimizer_type, random_shuffle,
                                          allow_neg_weight, ggx_ref_jac_weight ,mipmaps, view_dependent,view_option_str,"it" + str(i))


    logger.info(f"MAX n_iter {n_epoch} reached")

def optimize_function():
    import visualization


    ggx_alpha = 0.1

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='optim_info_ggx' + str(ggx_alpha) + '.log',filemode='a', level=logging.INFO,format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',datefmt='%H:%M:%S')
    logger.info("\n\n")



    face = 4
    u,v = 0.8,0.2
    location_global = map_util.uv_to_xyz((u, v), face)
    location_global = location_global.reshape((1, -1))
    ggx_ref = compute_ggx_ndf_reference(128, ggx_alpha, location_global)
    location_global = torch.from_numpy(location_global)
    ggx_ref = torch.from_numpy(ggx_ref)
    #normalize GGX
    ggx_ref /= torch.sum(ggx_ref)

    constant = True
    n_sample_per_frame = 8

    if constant:
        model_name = "constant_ggx_" + "{:.3f}".format(ggx_alpha)
    else:
        model_name = "quad_ggx_" + "{:.3f}".format(ggx_alpha)

    # if adjust_level:
    #     model_name = model_name + "_ladj"

    if not constant:
        model = torch_util.QuadModel(n_sample_per_frame)
    else:
        model = torch_util.ConstantModel(n_sample_per_frame)

    if os.path.exists("./model/" + model_name):
        logger.info("Read model from dict")
        model.load_state_dict(torch.load("./model/" + model_name))

    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    n_epoch = 100000




    #with torch.autograd.set_detect_anomaly(True):
    for i in range(n_epoch):
        optimizer.zero_grad()
        params = model()
        loss, result = error_func(params, location_global, n_sample_per_frame, ggx_ref, constant)
        loss.backward()
        optimizer.step()

        logger.info(f"[it{i}]Loss: {loss.item()}")

        if i % 500 == 0:
            #normalize result
            result /= torch.sum(result)
            visualization.visualize_optim_result(ggx_ref, result)
            save_model(model, "./model/" + model_name)

        print("it{}:loss is{}".format(i,loss.item()))
    print("Done")




def verify_push_back():
    face = 0
    u = 0.25
    v = 0.75
    location_global = map_util.uv_to_xyz((u, v), face)
    location_global = location_global.reshape((1, -1))
    # u = 0.2
    # location_global2 = map_util.uv_to_xyz((u, v), face).reshape((1, -1))
    #
    # location_global = np.concatenate((location_global, location_global2))

    # BFGS optimization test
    # test_optimize(0.01, 128, location_global[0:1, :], 8)

    # ggx = compute_ggx_distribution_reference(128,0.01,location_global[0:1,:])
    #
    # ggx = torch.from_numpy(ggx)
    location_global = torch.from_numpy(location_global)
    # test_one_texel_full_optimization(location_global[0:1,:],8,ggx)
    #
    level_global = torch.tensor([6.0])
    n_level_global = 7
    initial_weight_global = torch.tensor([1.0])
    #
    t = torch_util.initialize_mipmaps(n_level_global)
    sample_info = torch_util.process_trilinear_samples(location_global, level_global, n_level_global, initial_weight_global)
    t = torch_util.process_bilinear_samples(sample_info, t)
    final_image = torch_util.push_back(t)

    #create a random 6,128,128 image

    rng = np.random.default_rng(12345)

    random_img = rng.random((6,128,128,1)) * 1000
    from interpolation import downsample_full
    downsample_full = downsample_full(random_img,7)
    final_image_np = final_image.detach().numpy()

    result_t = np.sum(final_image_np * random_img)

    result_1 = downsample_full[-1][0,0,0,0]



    print("?")








def calculate_one_texel(texel_dir, texel_idx, n_sample_per_frame, ggx_ref, weight, xyz, theta_phi, constant, coef_table, queue):
    """
    The routine used to compute error for a single texel, this is the function that will be called using Python's
    multiprocessing library to accelerate optimization
    :param texel_dir:
    :param texel_idx: which texel is this
    :param n_sample_per_frame:
    :param ggx_ref:
    :param weight:
    :param xyz:
    :param theta_phi:
    :param coef_table:
    :param constant:
    :param mp_queue: the queue used to put returned values in
    :return:
    """
    coefficient_table = coef_table

    sample_directions = torch.empty((0,3))
    sample_weights = torch.empty((0,))
    sample_levels = torch.empty((0,))

    for i in range(3):
        cur_frame_weight = weight[i]

        #if frame_weight is 0, we skip, how to do this in parallel?
        if cur_frame_weight == 0:
            continue

        X, Y, Z = xyz[i]

        _, _, theta2, phi2 = theta_phi[i]

        coeff_start = i * n_sample_per_frame
        coeff_end = coeff_start + n_sample_per_frame



        if not constant:
            coeff_x_table = coefficient_table[0, :, coeff_start:coeff_end]
            coeff_y_table = coefficient_table[1, :, coeff_start:coeff_end]
            coeff_z_table = coefficient_table[2, :, coeff_start:coeff_end]
            coeff_level_table = coefficient_table[3, :, coeff_start:coeff_end]
            coeff_weight_table = coefficient_table[4, :, coeff_start:coeff_end]

            coeff_x = coeff_x_table[0] + coeff_x_table[1] * theta2 + coeff_x_table[2] * phi2
            coeff_y = coeff_y_table[0] + coeff_y_table[1] * theta2 + coeff_y_table[2] * phi2
            coeff_z = coeff_z_table[0] + coeff_z_table[1] * theta2 + coeff_z_table[2] * phi2


            level = coeff_level_table[0] + coeff_level_table[1] * theta2 + coeff_level_table[2] * phi2
            weight = coeff_weight_table[0] + coeff_weight_table[1] * theta2 + coeff_weight_table[2] * phi2
        else:
            coeff_x = coefficient_table[0, coeff_start:coeff_end]
            coeff_y = coefficient_table[1, coeff_start:coeff_end]
            coeff_z = coefficient_table[2, coeff_start:coeff_end]
            level = coefficient_table[3, coeff_start:coeff_end]
            weight = coefficient_table[4, coeff_start:coeff_end]

        coeff_x = torch.stack((coeff_x, coeff_x, coeff_x), dim=-1)
        coeff_y = torch.stack((coeff_y, coeff_y, coeff_y), dim=-1)
        coeff_z = torch.stack((coeff_z, coeff_z, coeff_z), dim=-1)


        sample_direction = coeff_x * X + coeff_y * Y + coeff_z * Z
        abs_direction = torch.abs(sample_direction)
        print("before max")
        max_dir = torch.max(abs_direction, dim=-1).values
        print("after max")
        sample_direction_map = sample_direction / torch.stack([max_dir, max_dir, max_dir], dim=-1)

        weight_cur_frame = weight * cur_frame_weight

        #TODO: how to make weight/level always positive?
        weight_cur_frame = torch.clip(weight_cur_frame, 0)
        level = torch.clip(level,0.0,6.0)

        sample_directions = torch.concatenate((sample_directions, sample_direction_map))
        sample_weights = torch.concatenate((sample_weights, weight_cur_frame))

        #TODO: Jacobian?

        sample_levels = torch.concatenate((sample_levels, level))

    print("start computing result for ",texel_idx)

    result = torch_util.compute_contribution(sample_directions, sample_levels, sample_weights, 7)

    e_arr = L1_error_one_texel(ggx_ref, result)

    e = torch.sum(e_arr)
    print("result for ",texel_idx, " ", e.detach().item())


    queue.put((texel_idx,e))

    #return e


def chunk_list(lst, chunk_size):
    """
    Splits the list `lst` into chunks of size `chunk_size`.

    Args:
        lst (list): The list to split.
        chunk_size (int): The size of each chunk.

    Returns:
        list of lists: A list where each element is a chunk of the original list.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def test_multiple_texel_opt_cuda_stream(n_sample_per_frame, n_sample_per_level ,ggx_ref_list, weight_list, xyz_list, theta_phi_list ,streams,results,coef_table=None, constant= False, adjust_level = False, device = torch.device('cpu')):
    """

    :param n_sample_per_frame:
    :param n_sample_per_level:
    :param ggx_ref_list:
    :param weight_list:
    :param xyz_list:
    :param theta_phi_list:
    :param coef_table:
    :param constant:
    :param adjust_level:
    :param device:
    :return:
    """
    if coef_table is None:
        # a single coefficient table in shape [5,3,nsample_per_frame * 3]

        coefficient_table = torch.rand((5, 3, n_sample_per_frame * 3))
    else:
        coefficient_table = coef_table

    for dir_idx in range(n_sample_per_level):
        sample_directions = torch.empty((0, 3), device=device)
        sample_weights = torch.empty((0,), device=device)
        sample_levels = torch.empty((0,), device=device)
        for i in range(3):
            X, Y, Z = xyz_list[i][0][dir_idx], xyz_list[i][1][dir_idx], xyz_list[i][2][dir_idx]
            theta2, phi2 = theta_phi_list[i][2][dir_idx], theta_phi_list[i][3][dir_idx]
            frame_weight = weight_list[i][dir_idx]

            if frame_weight == 0:
                continue

            coeff_start = i * n_sample_per_frame
            coeff_end = coeff_start + n_sample_per_frame
            if not constant:
                coeff_x_table = coefficient_table[0, :, coeff_start:coeff_end]
                coeff_y_table = coefficient_table[1, :, coeff_start:coeff_end]
                coeff_z_table = coefficient_table[2, :, coeff_start:coeff_end]
                coeff_level_table = coefficient_table[3, :, coeff_start:coeff_end]
                coeff_weight_table = coefficient_table[4, :, coeff_start:coeff_end]

                coeff_x = coeff_x_table[0] + coeff_x_table[1] * theta2 + coeff_x_table[2] * phi2
                coeff_y = coeff_y_table[0] + coeff_y_table[1] * theta2 + coeff_y_table[2] * phi2
                coeff_z = coeff_z_table[0] + coeff_z_table[1] * theta2 + coeff_z_table[2] * phi2

                level = coeff_level_table[0] + coeff_level_table[1] * theta2 + coeff_level_table[2] * phi2
                weight = coeff_weight_table[0] + coeff_weight_table[1] * theta2 + coeff_weight_table[2] * phi2
            else:
                coeff_x = coefficient_table[0, coeff_start:coeff_end]
                coeff_y = coefficient_table[1, coeff_start:coeff_end]
                coeff_z = coefficient_table[2, coeff_start:coeff_end]
                level = coefficient_table[3, coeff_start:coeff_end]
                weight = coefficient_table[4, coeff_start:coeff_end]

            coeff_x = torch.stack((coeff_x, coeff_x, coeff_x), dim=-1)
            coeff_y = torch.stack((coeff_y, coeff_y, coeff_y), dim=-1)
            coeff_z = torch.stack((coeff_z, coeff_z, coeff_z), dim=-1)

            sample_direction = coeff_x * X + coeff_y * Y + coeff_z * Z
            abs_direction = torch.abs(sample_direction)
            max_dir = torch.max(abs_direction, dim=-1).values
            sample_direction_map = sample_direction / torch.stack([max_dir, max_dir, max_dir], dim=-1)

            weight_cur_frame = weight * frame_weight

            # TODO: how to make weight/level always positive?
            weight_cur_frame = torch.clip(weight_cur_frame, 0)
            level = torch.clip(level, 0.0, 6.0)

            sample_directions = torch.concatenate((sample_directions, sample_direction_map))
            sample_weights = torch.concatenate((sample_weights, weight_cur_frame))

            # TODO: Jacobian?
            if adjust_level:
                j = 3 / 4 * torch.log2(torch_util.torch_dot_vectorized_2D(sample_direction_map, sample_direction_map))
                level += j

            sample_levels = torch.concatenate((sample_levels, level))

        stream = streams[dir_idx]
        process_input(dir_idx,stream,results,sample_directions,sample_levels,sample_weights,ggx_ref_list[dir_idx])

    for stream in streams:
        stream.synchronize()

    tmp = torch.stack(results)
    mean_error = tmp.mean()
    return mean_error




def multiple_texel_full_optimization_parallel_cuda(n_sample_per_frame, n_sample_per_level ,constant= False, n_process = 8, ggx_alpha = 0.1):
    """

    :param texel_dirs:
    :param n_sample_per_frame: how many samples we take to approximate a texel
    :param n_sample_per_level: how many texels we use to measure error
    :param ggx_ref_list:
    :param weight_list: a list with three element, each contains the weight of all directions for this frame
    :param xyz_list: the same as weight_list
    :param theta_phi_list: the same theta_phi
    :param coef_table:
    :param constant:
    :param n_process: how many process in parallel
    :return:
    """

    queue = mp.Queue()

    # logger = logging.getLogger(__name__)
    # logging.basicConfig(
    #     filename='optim_info_multi_ggx_' + "{:.3f}".format(ggx_alpha) + "_" + str(n_sample_per_level) + '.log',
    #     filemode='a', level=logging.INFO,
    #     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    # logger.info("\n\n")

    rng = np.random.default_rng(12345)
    all_locations = map_util.sample_location(n_sample_per_level, rng)

    if constant:
        model_name = "constant_ggx_multi_" + "{:.3f}".format(ggx_alpha) + "_" + str(n_sample_per_level)
    else:
        model_name = "quad_ggx_multi_" + "{:.3f}".format(ggx_alpha) + "_" + str(n_sample_per_level)

    if not constant:
        model = torch_util.QuadModel(n_sample_per_frame)
    else:
        model = torch_util.ConstantModel(n_sample_per_frame)

    if os.path.exists("./model/" + model_name):
        # logger.info("Read model from dict")
        model.load_state_dict(torch.load("./model/" + model_name))

    model.share_memory()

    ref_list = []
    for i in range(n_sample_per_level):
        location = all_locations[i, :]
        ggx_ref = compute_ggx_ndf_reference(128, ggx_alpha, location)
        ggx_ref = torch.from_numpy(ggx_ref)
        ggx_ref /= torch.sum(ggx_ref)
        ref_list.append(ggx_ref)

    all_locations = torch.from_numpy(all_locations)
    weight_per_frame, xyz_per_frame, theta_phi_per_frame = precompute_opt_info(all_locations, n_sample_per_level)

    #convert per frame list to per sample list

    weight_per_sample = []
    xyz_per_sample = []
    theta_phi_per_sample = []

    for i in range(n_sample_per_level):
        cur_sample_weight = []
        cur_sample_xyz = []
        cur_sample_theta_phi = []
        for frame_idx in range(3):
            cur_sample_weight.append(weight_per_frame[frame_idx][i])
            cur_sample_xyz.append((xyz_per_frame[frame_idx][0][i],xyz_per_frame[frame_idx][1][i],xyz_per_frame[frame_idx][2][i]))
            cur_sample_theta_phi.append((theta_phi_per_frame[frame_idx][0][i],theta_phi_per_frame[frame_idx][1][i],theta_phi_per_frame[frame_idx][2][i],theta_phi_per_frame[frame_idx][3][i]))

        weight_per_sample.append(cur_sample_weight)
        xyz_per_sample.append(cur_sample_xyz)
        theta_phi_per_sample.append(cur_sample_theta_phi)


    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    n_epoch = 1000000


    dir_idx = list(range(0,n_sample_per_level))
    chunks = chunk_list(dir_idx,n_process)
    n_chunk = len(chunks)


    pre_computed_args = []
    for j in range(n_sample_per_level):
        params = (all_locations[j], j ,n_sample_per_frame, ref_list[j], weight_per_sample[j], xyz_per_sample[j], theta_phi_per_sample[j], constant)
        pre_computed_args.append(params)



    for i in range(n_epoch):
        optimizer.zero_grad()
        params = model()


        #prepare arguments
        args = []
        for j in range(n_sample_per_level):
            cur_param_tuple = pre_computed_args[j]
            cur_param_tuple = cur_param_tuple + (params,) + (queue,)
            args.append(cur_param_tuple)

        error_list = []
        for chunk_idx in range(n_chunk):
            processes = []
            for j in range(n_process):
                process = mp.Process(
                    target=calculate_one_texel,
                    args = args[chunk_idx * n_process + j]
                )
                processes.append(process)
                process.start()

            for j in range(n_process):
                idx,error = queue.get()
                error_list.append(error)

            for process in processes:
                process.join()



        # for j in range(n_sample_per_level):
        #     error_list.append(results[j])

        tmp = torch.stack(error_list)
        mean_error = tmp.mean()
        mean_error.backward()
        optimizer.step()
        print("[it{}]:loss is{}".format(i, mean_error.item()))
        # logger.info("[it{}]:loss is{}".format(i, mean_error.item()))

        #if i % 500 == 0:
            # normalize result
            # result /= torch.sum(result)
            # visualization.visualize_optim_result(ggx_ref, result)
            # logger.info(f"saving model")
            #save_model(model, "./model/" + model_name)
            # logger.info(f"[it{i}]Loss: {mean_error.item()}")

    # logger.info(f"MAX n_iter {n_epoch} reached")






def test_vectorized_multiple_opt(ggx_alpha):
    model_name = "quad_ggx_multi_0.471_20_ladj"

    model = torch_util.QuadModel(8)

    if os.path.exists("./model/" + model_name):
        model.load_state_dict(torch.load("./model/" + model_name))

    params = model()

    n_sample_per_level = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(12345)
    all_locations = map_util.sample_location(n_sample_per_level,rng)



    ref_list = []
    for i in range(n_sample_per_level):
        location = all_locations[i,:]
        ggx_ref = compute_ggx_ndf_reference(128, ggx_alpha, location)
        ggx_ref = torch.from_numpy(ggx_ref).to(device)
        ggx_ref /= torch.sum(ggx_ref)
        ref_list.append(ggx_ref)

    all_locations = torch.from_numpy(all_locations).to(device)
    weight_per_frame, xyz_per_frame, theta_phi_per_frame = precompute_opt_info(all_locations, n_sample_per_level)

    for _ in range(20):
        start_time = time.time()

        pushed_back_result = multiple_texel_full_optimization_vectorized(8, n_sample_per_level, None, weight_per_frame, xyz_per_frame, theta_phi_per_frame, adjust_level=True, device = device, coef_table=params)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"one pushback preparation took {elapsed_time:.4f} seconds to execute.")



        start_time = time.time()
        error_list, result_list = multiple_texel_full_optimization(all_locations, 8,
                                                                   n_sample_per_level, ref_list, weight_per_frame,
                                                                   xyz_per_frame, theta_phi_per_frame,
                                                                   coef_table=params, constant=False,
                                                                   adjust_level=True, device=device)


        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"one pushback preparation took {elapsed_time:.4f} seconds to execute.")

    # for i in range(n_sample_per_level):
    #     result_vectorized = pushed_back_result[i]
    #     result_single = result_list[i]
    #
    #     diff = result_vectorized - result_single
    #     print("1")


if __name__ == "__main__":
    print("CUDA Availability:",torch.cuda.is_available())

    (n_sample_per_frame_g,n_sample_per_level_g, ggx_info, flag_constant, flag_adjust_level, optimizer_string,
     random_shuffle, allow_neg_weight, ggx_ref_jac_weight, lr,
     view_option, view_reflection_parameterization_g , view_ndf_clipping_g, use_vndf_g, fixed_costheta_info)=  process_cmd()
    #
    if isinstance(ggx_info, int):
        import specular
        info = specular.cubemap_level_params(18)
        ggx_alpha = info[ggx_info].roughness
    else:
        ggx_alpha = ggx_info

    if fixed_costheta_info is not None:
        fixed_costheta_res_g, fixed_costheta_idx_g = fixed_costheta_info
        if fixed_costheta_res_g > 0:
            costheta_list = torch_util.gen_cos_theta_list(fixed_costheta_res_g)
            fixed_costheta = costheta_list[fixed_costheta_idx_g]
        else:
            fixed_costheta = None
    else:
        fixed_costheta = None

    print("Computing GGX alpha {}, using {} directions,{} samples per frame.\n"
          "Adjust Level with Jacobian:{}\n"
          "Using constant params:{}\n"
          "Optimizer:{} - LR{}\n"
          "random shuffle:{}\n"
          "Allow negative weight:{},\n"
          "Use Jacobian weighted ggx as reference:{}\n"
          "View dependency Option:{}\n"
          "Parameterize using reflected direction:{}\n"
          "Clipping below horizong ndf:{}\n"
          "Use VNDF as reference:{}\n"
          "fixed cos theta:{}".format(ggx_alpha,n_sample_per_level_g,n_sample_per_frame_g,flag_adjust_level,
                                      flag_constant,optimizer_string,lr,random_shuffle,allow_neg_weight,
                                      ggx_ref_jac_weight,view_option,view_reflection_parameterization_g,
                                      view_ndf_clipping_g,use_vndf_g, fixed_costheta))

    # #test_vectorized_multiple_opt(ggx_alpha)
    #
    #
    # set_start_method('spawn')




    #multiple_texel_full_optimization_parallel(8,50, False, 1, ggx_alpha)

    #t = create_downsample_pattern(130)
    #t = torch_uv_to_xyz_vectorized(t,0)
    optimize_multiple_locations(n_sample_per_level_g,flag_constant,n_sample_per_frame_g, ggx_alpha, adjust_level=flag_adjust_level,
                                vectorize=True, optimizer_type=optimizer_string, random_shuffle=random_shuffle,
                                allow_neg_weight=allow_neg_weight, ggx_ref_jac_weight=ggx_ref_jac_weight,
                                view_option_str=view_option ,learning_rate=lr,view_reflection_parameterization=view_reflection_parameterization_g,
                                view_ndf_clipping=view_ndf_clipping_g, use_vndf = use_vndf_g, fixed_cos_view_theta=fixed_costheta)
    #optimize_function()

    # dummy location

    # test if reverse work as desired


    # u = 0.2
    # location_global2 = map_util.uv_to_xyz((u, v), face).reshape((1, -1))
    #
    # location_global = np.concatenate((location_global, location_global2))

    # BFGS optimization test
    #test_optimize(0.01, 128, location_global[0:1, :], 8)




    # level_global = torch.tensor([6.0])
    # n_level_global = 7
    # initial_weight_global = torch.tensor([1.0])

    # t = torch_util.initialize_mipmaps(n_level_global)
    # sample_info = torch_util.process_trilinear_samples(location_global, level_global, n_level_global,initial_weight_global)
    # t = torch_util.process_bilinear_samples(sample_info,t)
    # final_image = torch_util.push_back(t)

    print("Done")


