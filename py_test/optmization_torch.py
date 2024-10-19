import numpy as np
from numpy import dtype
from sympy.physics.units import action

import map_util
import mat_util
from datetime import datetime

from reference import compute_ggx_distribution_reference

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

    args = parser.parse_args()

    optimizer = args.optimizer
    ggx_level = args.ggxlevel
    ndir = args.ndir
    constant = args.constant
    adjust_level = args.adjustlevel
    optimizer = optimizer.lower()

    return ndir, ggx_level, constant, adjust_level, optimizer





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



def log_filename(ggx_alpha, n_sample_per_level,constant,adjust_level, optim_method:str ,post_fix_for_dirs = None):
    log_name = 'optim_info_multi_ggx_' + "{:.3f}".format(ggx_alpha) + "_" + str(n_sample_per_level)

    if constant:
        log_name = log_name + "_constant"
    else:
        log_name = log_name + "_quad"

    if adjust_level:
        log_name = log_name + "_ladj"

    log_name = log_name + "_" + optim_method
    if post_fix_for_dirs is not None:
        log_name = log_name + "_" + post_fix_for_dirs

    log_name = log_name + '.log'

    return log_name

def dir_filename(ggx_alpha, constant, n_sample_per_level, adjust_level, optim_method:str ,post_fix_for_dirs = None):
    if constant:
        dir_name = "constant_ggx_multi_" + "{:.3f}".format(ggx_alpha) + "_" + str(n_sample_per_level)
    else:
        dir_name = "quad_ggx_multi_" + "{:.3f}".format(ggx_alpha) + "_" + str(n_sample_per_level)

    if adjust_level:
        dir_name = dir_name + "_ladj"

    dir_name = dir_name + "_" + optim_method
    dir_name = dir_name + "_dirs"
    if post_fix_for_dirs is not None:
        dir_name = dir_name + "_" + post_fix_for_dirs
    dir_name = dir_name + ".pt"

    return dir_name

def model_filename(ggx_alpha, constant, n_sample_per_level, adjust_level, optim_method:str ,post_fix_for_dirs = None):
    if constant:
        model_name = "constant_ggx_multi_" + "{:.3f}".format(ggx_alpha) + "_" + str(n_sample_per_level)
    else:
        model_name = "quad_ggx_multi_" + "{:.3f}".format(ggx_alpha) + "_" + str(n_sample_per_level)

    if adjust_level:
        model_name = model_name + "_ladj"

    model_name = model_name + "_" + optim_method
    if post_fix_for_dirs is not None:
        model_name = model_name + "_" + post_fix_for_dirs

    return model_name



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



def test_multiple_texel_full_optimization_vectorized(n_sample_per_frame, n_sample_per_level ,ggx_ref_list, weight_list, xyz_list, theta_phi_list ,coef_table=None, constant= False, adjust_level = False, device = torch.device('cpu')):
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
            weight_cur_frame = torch.clip(weight_cur_frame, 0)
            level = torch.clip(level,0.0,6.0)

            sample_directions = torch.concatenate((sample_directions, sample_direction_map))
            sample_weights = torch.concatenate((sample_weights, weight_cur_frame))
            sample_idx = torch.concatenate((sample_idx, torch.full([n_sample_per_frame],dir_idx,device=device,dtype=torch.int)))

            #TODO: Jacobian?
            if adjust_level:
                j = 3 / 4 * torch.log2(torch_util.torch_dot_vectorized_2D(sample_direction_map, sample_direction_map))
                level += j

            sample_levels = torch.concatenate((sample_levels, level))


    result = torch_util_all_locations.compute_contribution_full(sample_directions,sample_levels,sample_weights,sample_idx,n_level=7,n_sample_per_level = n_sample_per_level)





    return result

def test_multiple_texel_full_optimization(texel_dirs,n_sample_per_frame, n_sample_per_level ,ggx_ref_list, weight_list, xyz_list, theta_phi_list ,coef_table=None, constant= False, adjust_level = False, device = torch.device('cpu')):
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
            weight_cur_frame = torch.clip(weight_cur_frame, 0)
            level = torch.clip(level,0.0,6.0)

            sample_directions = torch.concatenate((sample_directions, sample_direction_map))
            sample_weights = torch.concatenate((sample_weights, weight_cur_frame))

            #TODO: Jacobian?
            if adjust_level:
                j = 3 / 4 * torch.log2(torch_util.torch_dot_vectorized_2D(sample_direction_map, sample_direction_map))
                level += j

            sample_levels = torch.concatenate((sample_levels, level))

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
    # e = np.average(e_arr)

    return e, result





class SimpleModel(nn.Module):
    def __init__(self, n_sample_per_frame):
        super(SimpleModel, self).__init__()
        self.params = nn.Parameter(torch.rand((5,3,3*n_sample_per_frame)), requires_grad=True)

    def forward(self):
        return self.params


class ConstantModel(nn.Module):
    def __init__(self, n_sample_per_frame):
        super(ConstantModel, self).__init__()
        self.params = nn.Parameter(torch.concatenate(
            (torch.rand((2, 3 * n_sample_per_frame)) / 30.0,
             torch.ones((1,3 * n_sample_per_frame)) - 0.01,
             (torch.rand((2, 3 * n_sample_per_frame)) + 1) / 2.0,
             )
        ) , requires_grad=True)
    def forward(self):
        return self.params


def precompute_opt_info(texel_directions, n_sample_per_level):
    """
    Given the texel directions, precompute XYZ,frame weight and theta_phi
    :param texel_directions: assume the direction is already normalized to the unit cube
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


def optimize_multiple_locations(n_sample_per_level, constant, n_sample_per_frame, ggx_alpha = 0.1, adjust_level = False, cuda_stream = False, vectorize = True, optimizer_type = "adam"):
    """
    To speed up, a lot of things can be precomputed, including the relative XYZ,the frame weight
    we don't have to compute this in every iteration
    :param n_sample_per_level:
    :return:
    """
    device = get_device()



    logger = logging.getLogger(__name__)

    log_name = log_filename(ggx_alpha,n_sample_per_level,constant,adjust_level,optimizer_type)

    logging.basicConfig(filename=log_name, filemode='a', level=logging.INFO,
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    logger.info("\n\n")

    rng = np.random.default_rng(12345)
    all_locations = map_util.sample_location(n_sample_per_level,rng)

    dir_name = dir_filename(ggx_alpha, constant, n_sample_per_level, adjust_level, optimizer_type)
    torch.save(torch.from_numpy(all_locations), dir_name)


    model_name = model_filename(ggx_alpha,constant,n_sample_per_level,adjust_level,optimizer_type)

    if not constant:
        model = SimpleModel(n_sample_per_frame)
    else:
        model = ConstantModel(n_sample_per_frame)

    if os.path.exists("./model/" + model_name):
        logger.info("Read model from dict")
        model.load_state_dict(torch.load("./model/" + model_name))

    model.to(device)

    ref_list = []
    for i in range(n_sample_per_level):
        location = all_locations[i,:]
        ggx_ref = compute_ggx_distribution_reference(128, ggx_alpha, location)
        ggx_ref = torch.from_numpy(ggx_ref).to(device)
        ggx_ref /= torch.sum(ggx_ref)
        ref_list.append(ggx_ref)
    if vectorize:
        ref_list = torch.stack(ref_list)

    all_locations = torch.from_numpy(all_locations).to(device)
    weight_per_frame,xyz_per_frame,theta_phi_per_frame = precompute_opt_info(all_locations, n_sample_per_level)

    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    elif optimizer_type == "bfgs":
        optimizer = optim.LBFGS(model.parameters(), lr = 0.7, line_search_fn="strong_wolfe")
    else:
        raise ValueError("Unknown optimizer type")
    n_epoch = 1000000


    if cuda_stream:
        streams = [torch.cuda.Stream(device=device) for _ in range(n_sample_per_level)]
        results = [None] * 50

    final_loss_record = [0] #used in BFGS

    def closure():
        optimizer.zero_grad()
        params = model()

        # start_time = time.time()

        if not cuda_stream:
            if vectorize:
                tmp_pushed_back_result = test_multiple_texel_full_optimization_vectorized(n_sample_per_frame,
                                                                                      n_sample_per_level, ref_list,
                                                                                      weight_per_frame, xyz_per_frame,
                                                                                      theta_phi_per_frame, params,
                                                                                      constant, adjust_level, device)
                # normalize pushed_back result
                tmp_pushed_back_sum = torch.sum(tmp_pushed_back_result, dim=[1, 2, 3, 4], keepdim=True)
                tmp_pushed_back_result /= tmp_pushed_back_sum
                diff = torch.abs(ref_list - tmp_pushed_back_result)
                diff = torch.sum(diff, dim=[1, 2, 3, 4])
                mean_error = torch.mean(diff)
                final_loss_record[0] = mean_error.item()
            else:
                error_list, result_list = test_multiple_texel_full_optimization(all_locations, n_sample_per_frame,
                                                                                n_sample_per_level, ref_list,
                                                                                weight_per_frame, xyz_per_frame,
                                                                                theta_phi_per_frame, coef_table=params,
                                                                                constant=constant,
                                                                                adjust_level=adjust_level,
                                                                                device=device)

                tmp = torch.stack(error_list)
                mean_error = tmp.mean()
        else:
            mean_error = test_multiple_texel_opt_cuda_stream(n_sample_per_frame, n_sample_per_level, ref_list,
                                                             weight_per_frame, xyz_per_frame, theta_phi_per_frame,
                                                             streams, results, coef_table=params, constant=constant,
                                                             adjust_level=adjust_level, device=device)

        mean_error.backward()
        return mean_error


    if optimizer_type == "adam":
        for i in range(n_epoch):
            loss = closure()
            if not torch.isnan(loss):
                optimizer.step()
            else:
                logger.info("NaN loss detected")
                logger.info("Current parameters are", model().detach().numpy())
                pushed_back_result = test_multiple_texel_full_optimization_vectorized(n_sample_per_frame,
                                                                                      n_sample_per_level, ref_list,
                                                                                      weight_per_frame, xyz_per_frame,
                                                                                      theta_phi_per_frame, model(),
                                                                                      constant, adjust_level, device)
                # normalize pushed_back result
                pushed_back_sum = torch.sum(pushed_back_result, dim=[1, 2, 3, 4], keepdim=True)
                logger.info("is pushed_back_result NaN? ", torch.isnan(pushed_back_result).any())
                logger.info("is pushed_back_sum 0?", pushed_back_sum == 0.0)


            logger.info("[it{}]:loss is{}".format(i, loss.item()))

            if i % 500 == 0:
                # normalize result
                # result /= torch.sum(result)
                # visualization.visualize_optim_result(ggx_ref, result)
                logger.info(f"saving model")
                save_model(model, "./model/" + model_name)
                # logger.info(f"[it{i}]Loss: {mean_error.item()}")
    else:
        for i in range(n_epoch):
            optimizer.step(closure)
            logger.info("[it{}]:loss is{}".format(i, final_loss_record[0]))
            if i % 500 == 0:
                logger.info(f"saving model")
                save_model(model, "./model/" + model_name)


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
    ggx_ref = compute_ggx_distribution_reference(128,ggx_alpha,location_global)
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

    if adjust_level:
        model_name = model_name + "_ladj"

    if not constant:
        model = SimpleModel(n_sample_per_frame)
    else:
        model = ConstantModel(n_sample_per_frame)

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
        model = SimpleModel(n_sample_per_frame)
    else:
        model = ConstantModel(n_sample_per_frame)

    if os.path.exists("./model/" + model_name):
        # logger.info("Read model from dict")
        model.load_state_dict(torch.load("./model/" + model_name))

    model.share_memory()

    ref_list = []
    for i in range(n_sample_per_level):
        location = all_locations[i, :]
        ggx_ref = compute_ggx_distribution_reference(128, ggx_alpha, location)
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

    model = SimpleModel(8)

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
        ggx_ref = compute_ggx_distribution_reference(128, ggx_alpha, location)
        ggx_ref = torch.from_numpy(ggx_ref).to(device)
        ggx_ref /= torch.sum(ggx_ref)
        ref_list.append(ggx_ref)

    all_locations = torch.from_numpy(all_locations).to(device)
    weight_per_frame, xyz_per_frame, theta_phi_per_frame = precompute_opt_info(all_locations, n_sample_per_level)

    for _ in range(20):
        start_time = time.time()

        pushed_back_result = test_multiple_texel_full_optimization_vectorized(8,n_sample_per_level,None,weight_per_frame,xyz_per_frame,theta_phi_per_frame,adjust_level=True,device = device, coef_table=params)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"one pushback preparation took {elapsed_time:.4f} seconds to execute.")



        start_time = time.time()
        error_list, result_list = test_multiple_texel_full_optimization(all_locations, 8,
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
    n_sample_per_level, mipmap_level_to_compute, flag_constant, flag_adjust_level, optimizer_string =  process_cmd()

    #verify_push_back()
    import specular
    info = specular.cubemap_level_params(18)

    ggx_alpha = info[mipmap_level_to_compute].roughness
    print("Computing GGX alpha {}, using {} directions.\nAdjust Level with Jacobian:{}\nUsing constant params:{}\nOptimizer:{}".format(ggx_alpha,n_sample_per_level,flag_adjust_level,flag_constant,optimizer_string))

    #test_vectorized_multiple_opt(ggx_alpha)


    set_start_method('spawn')




    #multiple_texel_full_optimization_parallel(8,50, False, 1, ggx_alpha)

    #t = create_downsample_pattern(130)
    #t = torch_uv_to_xyz_vectorized(t,0)
    optimize_multiple_locations(80,flag_constant,8, ggx_alpha, adjust_level=flag_adjust_level, cuda_stream=False,vectorize=True, optimizer_type=optimizer_string)
    #optimize_function()

    # dummy location

    # test if reverse work as desired

    face = 0
    u = 0.5
    v = 0.5
    location_global = map_util.uv_to_xyz((u, v), face)
    location_global = location_global.reshape((1, -1))
    # u = 0.2
    # location_global2 = map_util.uv_to_xyz((u, v), face).reshape((1, -1))
    #
    # location_global = np.concatenate((location_global, location_global2))

    # BFGS optimization test
    #test_optimize(0.01, 128, location_global[0:1, :], 8)




    #ggx = compute_ggx_distribution_reference(128,0.01,location_global[0:1,:])
    #
    #ggx = torch.from_numpy(ggx)
    #location_global = torch.from_numpy(location_global)
    #test_one_texel_full_optimization(location_global[0:1,:],8,ggx)

    # level_global = torch.tensor([6.0])
    # n_level_global = 7
    # initial_weight_global = torch.tensor([1.0])

    # t = torch_util.initialize_mipmaps(n_level_global)
    # sample_info = torch_util.process_trilinear_samples(location_global, level_global, n_level_global,initial_weight_global)
    # t = torch_util.process_bilinear_samples(sample_info,t)
    # final_image = torch_util.push_back(t)

    print("Done")


