import numpy as np
from numpy import dtype

import map_util
import mat_util
from datetime import datetime
from filter import frame_axis_index
from interpolation import get_edge_information
from reference import compute_ggx_distribution_reference

import scipy
import torch
import torch.nn as nn
import torch.optim as optim

"""
Optimization routine

All the mipmaps here have only one channel, which is the weight/contribution

"""

chan_count = 1


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
    module = torch.load(path)
    return module


def torch_jacobian_vertorized(xyz):
    """

    :param xyz: in shape (N,3) or (M,N,3)
    :return:
    """
    power_2 = xyz * xyz
    sum_xyz = torch.sum(power_2, dim=-1)
    j = 1 / torch.pow(sum_xyz, 3/2)

    return j


def torch_uv_to_xyz_vectorized(uv:torch.Tensor ,idx, normalize_flag=False):
    """

    :param uv: array in shape of (N,2) or (M,N,2)
    :param idx:
    :param normalize_flag: whether we normalize the xyz vector or not
    when computing the Jacobian, we should not normalize it
    :return: vectorized direction in shape of (N,3)
    """

    if uv.ndim == 2:
        u = uv[:,0]
        v = uv[:,1]
    elif uv.ndim == 3:
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        raise NotImplementedError

    uc = 2.0 * u - 1.0
    vc = 2.0 * v - 1.0

    if idx == 0:
        x = torch.ones_like(uc)
        y = vc
        z = -uc
    elif idx == 1:
        x = -torch.ones_like(uc)
        y = vc
        z = uc
    elif idx == 2:
        x = uc
        y = torch.ones_like(uc)
        z = -vc
    elif idx == 3:
        x = uc
        y = -torch.ones_like(uc)
        z = vc
    elif idx == 4:
        x = uc
        y = vc
        z = torch.ones_like(uc)
    elif idx == 5:
        x = -uc
        y = vc
        z = -torch.ones_like(uc)
    else:
        raise NotImplementedError


    vec = torch.stack((x, y, z), dim=-1)
    if normalize_flag:
        vec = torch_normalized(vec)

    return vec





def torch_normalized(a, axis=-1, order=2):
    # https://stackoverflow.com/a/21032099
    norm = torch.linalg.norm(a, ord=order, dim = axis)
    return a / norm

def torch_gen_frame_xyz(faces_xyz, frame_idx):
    """
    The frame xyz is used exclusively in sampling parameters
    The original direction of the texel is considered the Z axis, we note the normal of this face a
    The X axis is cross(a,z)  The y axis is cross(Z,X)
    :param faces_xyz: (6,res,res,3)
    :param frame_idx: this affects how we construct the up vector
    :return:
    """
    Z = torch_normalized(faces_xyz, axis=-1)
    polar_axis = torch.zeros_like(Z)
    if frame_idx == 0 or frame_idx == 1 or frame_idx == 2:
        polar_axis[...,frame_idx] = 1.0
    else:
        raise NotImplementedError

    X = torch_normalized(torch.linalg.cross(polar_axis, Z),axis=-1)

    # This is guaranteed to be unit vector
    Y = torch.linalg.cross(Z,X)

    return X,Y,Z


def torch_gen_frame_weight(facex_xyz, frame_idx, follow_code = False):
    """
    Compute frame weight for each texel according to the paper, the up/bot face have little weight
    :param facex_xyz:
    :param frame_idx:
    :return:
    """
    new_x_idx, new_y_idx, new_z_idx = frame_axis_index(frame_idx, follow_code)
    faces_xyz_abs = torch.abs(facex_xyz)

    frame_weight = torch.clip(4 * torch.maximum(faces_xyz_abs[...,new_x_idx], faces_xyz_abs[...,new_y_idx]) - 3,0.0,1.0)

    return frame_weight


def torch_gen_theta_phi(faces_xyz,frame_idx, follow_code = False):
    """
    Generate a theta phi table of shape (6,res,res,2) according to the paper
    :param faces_xyz: original xyz direction for each texel
    :param frame_idx: the index of the frame, used to determine the new x,y,z axis
    :return: theta,phi,theta^2,phi^2
    """
    new_x_idx, new_y_idx, new_z_idx = frame_axis_index(frame_idx,follow_code)   #write mipmap for preview

    #TODO: Why use abs(z) in original code?
    nx = faces_xyz[...,new_x_idx]
    ny = faces_xyz[...,new_y_idx]
    nz = faces_xyz[...,new_z_idx]
    max_xy = torch.maximum(torch.abs(nx),torch.abs(ny))

    #normalize nx,ny, in 2/3 of the cases, one of nx and ny should be 1 without normalizing it
    nx = nx / max_xy
    ny = ny / max_xy



    theta = torch.zeros_like(nx)
    theta[(ny < nx) & (ny <= -0.999)] = nx[(ny < nx) & (ny <= -0.999)]
    theta[(ny < nx) & (ny > -0.999)] = ny[(ny < nx) & (ny > -0.999)]
    theta[(nx <= ny) & (ny >= 0.999)] = -nx[(nx <= ny) & (ny >= 0.999)]
    theta[(nx <= ny) & (ny < 0.999)] = -ny[(nx <= ny) & (ny < 0.999)]


    phi = torch.zeros_like(nx)
    phi[nz <= -0.999] = -max_xy[nz <= -0.999]
    phi[nz >= 0.999] = max_xy[nz >= 0.999]
    phi[(nz > -0.999) & (nz < 0.999)] = nz[(nz > -0.999) & (nz < 0.999)]

    theta2 = theta * theta
    phi2 = phi * phi

    return theta,phi,theta2,phi2





def torch_create_pixel_index(resolution,dimension):
    """
    Util function that create a pixel index from 0.5 to resolution - 0.5
    The length of this array is resolution, it can not be used for array index
    it should be used to create interpolator and compute actual position
    Given that every pixel is in the middle

    !Origin is at top-left!

    :param resolution: int or (int,int)
    :param dimension: the dimension of the resolution, must be 1 or 2
    :return: an 1D array if dimension == 1. two arrays generated from meshgrid if dimension == 2
    """
    if dimension == 1:
        #resolution + 0.5 is used because arange only generates [start,stop)
        idx_array = torch.arange(0.5, resolution+0.5,1.0)
        return idx_array
    elif dimension == 2:
        row_res = resolution[0]
        col_res = resolution[1]
        row_array = torch.arange(0.5, row_res+0.5,1.0)
        col_array = torch.arange(0.5, col_res+0.5,1.0)
        # xv is row, yv is col
        xv,yv = torch.meshgrid(row_array,col_array,indexing='ij')
        return xv,yv



def torch_gen_boundary_uv_for_interp(edge_side,reverse_flag,cubemap_res):
    """
    How to generate this is documented in downsample
    :param edge_side: L(eft) R(ight) U(p) D(own)
    :param reverse_flag: True or False
    :param cubemap_res:
    :return:
    """
    uv = torch.zeros((cubemap_res,2))
    uv_ascending_order = torch_create_pixel_index(cubemap_res,1)

    uv_ascending_order = uv_ascending_order / cubemap_res

    uv_min = uv_ascending_order.min().detach().item()
    uv_max = uv_ascending_order.max().detach().item()

    epsilon = 1.0 - uv_max
    assert 1.0 - uv_max == uv_min


    if edge_side == "L":
        # u is 0 while v goes from 1 to 0
        uv[:,0] = torch.full((cubemap_res,),uv_min - 2 * epsilon)
        uv[:,1] = torch.flip(uv_ascending_order,[0])
    elif edge_side == "R":
        # u is 1 while v gose from 1 to 0
        uv[:,0] = torch.full((cubemap_res,),uv_max + 2 * epsilon)
        uv[:,1] = torch.flip(uv_ascending_order,[0])
    elif edge_side == "U":
        # v is 1 while u goes from 0 to 1
        uv[:,0] = uv_ascending_order
        uv[:,1] = torch.full((cubemap_res,),uv_max + 2 * epsilon)
    elif edge_side == "D":
        # v is 0 while u goes from 0 to 1
        uv[:,0] = uv_ascending_order
        uv[:,1] = torch.full((cubemap_res,),uv_min - 2 * epsilon)
    else:
        raise NotImplementedError

    if reverse_flag:
        uv = torch.flip(uv,[0])

    return uv


def torch_gen_extended_uv_table(face_res):
    face_extended_res = face_res + 2
    left_uv = torch_gen_boundary_uv_for_interp('L', False, face_res)
    right_uv = torch_gen_boundary_uv_for_interp('R', False, face_res)
    up_uv = torch_gen_boundary_uv_for_interp('U', False, face_res)
    down_uv = torch_gen_boundary_uv_for_interp('D', False, face_res)

    # generate face uv
    uv_table = torch.zeros((face_extended_res, face_extended_res, 2))
    uv_ascending_order = torch_create_pixel_index(face_res, 1)
    uv_ascending_order /= face_res
    epsilon = uv_ascending_order.min()

    # yv for u, xv for v
    xv, yv = torch.meshgrid(torch.flip(uv_ascending_order,[0]), uv_ascending_order, indexing='ij')
    uv_table[1:-1, 1:-1, 0] = yv
    uv_table[1:-1, 1:-1, 1] = xv

    uv_table[1:-1, 0, :] = left_uv
    uv_table[1:-1, -1, :] = right_uv
    uv_table[0, 1:-1, :] = up_uv
    uv_table[-1, 1:-1, :] = down_uv

    # corner
    uv_table[0, 0, :] = torch.tensor([-epsilon, 1 + epsilon])
    uv_table[0, -1, :] = torch.tensor([1 + epsilon, 1 + epsilon])
    uv_table[-1, 0, :] = torch.tensor([-epsilon, -epsilon])
    uv_table[-1, -1, :] = torch.tensor([1 + epsilon, -epsilon])
    return uv_table



def torch_xyz_to_uv_vectorized(xyz:torch.Tensor):
    """

    :param xyz: array of xyz. xyz.shape[-1] must be 3(indicating xyz)
    :return:
    """
    x,y,z = xyz[...,0],xyz[...,1],xyz[...,2]
    abs_x = torch.abs(x)
    abs_y = torch.abs(y)
    abs_z = torch.abs(z)

    is_x_positive = x > 0
    is_y_positive = y > 0
    is_z_positive = z > 0

    face_idx = torch.zeros(xyz.shape[:-1],dtype = torch.int)
    u_idx = torch.zeros_like(face_idx,dtype=abs_x.dtype)
    v_idx = torch.zeros_like(face_idx,dtype=abs_x.dtype)
    max_axis = torch.zeros_like(face_idx,dtype=abs_x.dtype)

    face_0_condition = is_x_positive & (abs_x >= abs_y) & (abs_x >= abs_z)
    face_1_condition = (~is_x_positive) & (abs_x >= abs_y) & (abs_x >= abs_z)
    face_2_condition = is_y_positive & (abs_y >= abs_z) & (abs_y >= abs_x)
    face_3_condition = (~is_y_positive) & (abs_y >= abs_z) & (abs_y >= abs_x)
    face_4_condition = is_z_positive & (abs_z >= abs_y) & (abs_z >= abs_x)
    face_5_condition = (~is_z_positive) & (abs_z >= abs_y) & (abs_z >= abs_x)



    face_idx[face_0_condition] = 0
    face_idx[face_1_condition] = 1
    face_idx[face_2_condition] = 2
    face_idx[face_3_condition] = 3
    face_idx[face_4_condition] = 4
    face_idx[face_5_condition] = 5

    max_axis[face_0_condition | face_1_condition] = abs_x[face_0_condition | face_1_condition]
    max_axis[face_2_condition | face_3_condition] = abs_y[face_2_condition | face_3_condition]
    max_axis[face_4_condition | face_5_condition] = abs_z[face_4_condition | face_5_condition]

    u_idx[face_0_condition] = -z[face_0_condition]
    u_idx[face_1_condition] = z[face_1_condition]
    u_idx[face_2_condition] = x[face_2_condition]
    u_idx[face_3_condition] = x[face_3_condition]
    u_idx[face_4_condition] = x[face_4_condition]
    u_idx[face_5_condition] = -x[face_5_condition]

    v_idx[face_0_condition] = y[face_0_condition]
    v_idx[face_1_condition] = y[face_1_condition]
    v_idx[face_2_condition] = -z[face_2_condition]
    v_idx[face_3_condition] = z[face_3_condition]
    v_idx[face_4_condition] = y[face_4_condition]
    v_idx[face_5_condition] = y[face_5_condition]

    #normalize
    u_idx = 0.5 * (u_idx / max_axis + 1.0)
    v_idx = 0.5 * (v_idx / max_axis + 1.0)

    return u_idx, v_idx, face_idx










def initialize_mipmaps(n_level):
    """
    initialize an all zero mipmap chain
    :param n_level:
    :return:
    """
    mipmaps = []
    for i in range(n_level):
        res = level_to_res(i, n_level)
        tmp = torch.zeros((6, res, res, chan_count))
        mipmaps.append(tmp)
    return mipmaps


def level_to_res(level, n_level):
    """
    Get this level's mipmap resolution. Assume the lowest resolution is 2*2
    :param level: the level of mipmap, in shape of [N,]
    :param n_level: how many levels in mipmap
    :return: the resolution
    """
    return np.left_shift(2, int(n_level - 1 - level))


def process_trilinear_samples(location, level, n_level, initial_weight):
    """
    TODO, add initial portion
    :param location: xyz location in [N,3]
    :param level: mipmap level [N,
    :param n_level number of mipmap level
    :param initial_weight: [N,]
    :return:uv coordinates in [N,2]. high_res portion in [N,] low_res portion in[N,], face_idx in [N,]
    """
    # first clamp the level
    level = torch.clip(level, 0, n_level - 1)

    high_level = torch.floor(level)
    low_level = torch.ceil(level)

    high_res_portion = 1 - (level - high_level)
    low_res_portion = 1 - high_res_portion

    high_res_portion *= initial_weight
    low_res_portion *= initial_weight


    u, v, face_idx = torch_xyz_to_uv_vectorized(location)
    uv = torch.stack((u, v), dim=-1)
    face_idx = face_idx.int()

    # return {"uv":uv,
    #         "high_res_portion":high_res_portion,"low_res_portion":low_res_portion,
    #         "face_idx":face_idx,
    #         "high_res":high_res,"low_res":low_res,
    #         "high_level":high_level,"low_level":low_level}

    # concatenate high and low information
    uv = torch.vstack((uv, uv))
    face_idx = torch.concatenate((face_idx, face_idx))
    level = torch.concatenate((high_level, low_level))
    portion = torch.concatenate((high_res_portion, low_res_portion))

    return {"uv": uv,
            "portion": portion,
            "face_idx": face_idx,
            "level": level}


def process_bilinear_samples(trilerp_sample_info: {}, mipmaps: []):
    """

    :param trilerp_sample_info: dictionary returned from process_trilinear_samples
    :param mipmaps: all zero mipmaps
    :return:
    """
    n_level = len(mipmaps)

    uv = trilerp_sample_info['uv']
    portion = trilerp_sample_info['portion']
    face_idx = trilerp_sample_info['face_idx']
    level = trilerp_sample_info['level']

    # given a uv and its resolution/level, we find the four points that contribute to this bilerped sample
    # the location is expressed using index

    # generate per level uv mesh(with boundary extension)
    uv_grid_list = []
    for level_idx in range(n_level):
        cur_res = 2 << (n_level - level_idx - 1)
        uv_grid_list.append(torch_gen_extended_uv_table(cur_res))

    # loop each level(this design is mainly for numpy parallelization
    for level_idx in range(n_level):
        cur_res = 2 << (n_level - level_idx - 1)
        condition = (level == level_idx)
        cur_uv_grid = uv_grid_list[level_idx]

        cur_uv_ascending_order = cur_uv_grid[0, :, 0].contiguous()
        cur_uv_descending_order = torch.flip(cur_uv_ascending_order,[0])

        cur_u = (uv[:, 0])[condition]
        cur_v = (uv[:, 1])[condition]

        if cur_u.size == 0:
            """
            We do not find any samples located in this level, skip it
            """
            continue

        u_location = torch.searchsorted(cur_uv_ascending_order, cur_u, right=False)  # j in np array order

        v_location_inv = torch.searchsorted(cur_uv_ascending_order, cur_v, right=False)  # -i in np array order
        v_location = (
                                 cur_res + 2) - 1 - v_location_inv  # extended_res - 1 - v_location   # Visually, this v_location is the upper v(where v has a higher value)


        # The following two lines uses uv_grid ordering, where the order of v is inversed

        # note that u[location] and v[location] will never equal u/v if we use searchsorted(side='left')

        # For u, we found the location u[u_location-1] < cur_u <= u[u_location]
        # for v, we found the location v[location] >= cur_v > v[location+1]
        # in the corner(extreme) case, our index is at upper-right side of a uv grid.
        u_location_left = u_location - 1
        # v_location_bot = v_location - 1 which one is correct?
        v_location_bot = v_location + 1

        u_right = cur_uv_ascending_order[u_location]
        u_left = cur_uv_ascending_order[u_location_left]
        v_up = cur_uv_descending_order[v_location]
        v_bot = cur_uv_descending_order[v_location_bot]

        cur_uv = torch.stack((cur_u, cur_v), dim=-1)
        cur_portion = portion[condition]
        cur_face = face_idx[condition]

        p0, p1, p2, p3 = bilerp_inverse(cur_uv, cur_portion, u_right, u_left, v_up, v_bot)

        # assign all p0,p1,p2,p3
        # we create a temporary extended mipmap for now, and we will add them back later
        extended_mipmap_cur_level = torch.zeros((6, cur_res + 2, cur_res + 2, mipmaps[level_idx].shape[-1]),dtype=p0.dtype)

        extended_mipmap_cur_level.index_put_((cur_face,v_location,u_location_left),p0.reshape((-1,chan_count)),accumulate=True)
        extended_mipmap_cur_level.index_put_((cur_face,v_location,u_location),p1.reshape((-1,chan_count)),accumulate=True)
        extended_mipmap_cur_level.index_put_((cur_face,v_location_bot,u_location_left),p2.reshape((-1,chan_count)),accumulate=True)
        extended_mipmap_cur_level.index_put_((cur_face,v_location_bot,u_location),p3.reshape((-1,chan_count)),accumulate=True)


        # now move the extended boundaries to where they belong
        original_mipmap_current_level = process_extended_face(extended_mipmap_cur_level)

        mipmaps[level_idx] = original_mipmap_current_level

    return mipmaps


def process_extended_face(extended_mipmap):
    """
    move the value in the extended boundaries to the right place
    :param extended_mipmap: [6,N,N,channel]
    :return: original_mipmap
    """

    def move_edge(tmp_val, tmp_idx, tmp_edge, tmp_reverse):
        if tmp_reverse:
            tmp_val = torch.flip(tmp_val,dims=[0])

        if tmp_edge == 'L':
            original_map[tmp_idx][:, 0, :] += tmp_val
        elif tmp_edge == 'R':
            original_map[tmp_idx][:, -1, :] += tmp_val
        elif tmp_edge == 'U':
            original_map[tmp_idx][0, :, :] += tmp_val
        elif tmp_edge == 'D':
            original_map[tmp_idx][-1, :, :] += tmp_val
        else:
            raise NotImplementedError

    # for now we ignore the corner
    extended_res = extended_mipmap.shape[1]
    original_res = extended_res - 2
    original_map = torch.zeros((6, original_res, original_res, extended_mipmap.shape[-1]))

    for face_idx in range(6):
        idx_info, edge_side_info, reverse_info = get_edge_information(face_idx)

        idx_info = list(idx_info)
        edge_side_info = list(edge_side_info)
        reverse_info = list(reverse_info)

        cur_face = extended_mipmap[face_idx]

        for i in range(len(idx_info)):
            idx = idx_info[i]
            edge = edge_side_info[i]
            reverse = reverse_info[i]

            if i == 0:
                val = cur_face[1:-1, 0, :]
            elif i == 1:
                val = cur_face[1:-1, -1, :]
            elif i == 2:
                val = cur_face[0, 1:-1, :]
            else:
                val = cur_face[-1, 1:-1, :]

            move_edge(val, idx, edge, reverse)

        # copy the center
        original_map[face_idx] += cur_face[1:-1, 1:-1, :]

    # process the corner separately.
    """
    There are eight corners, each corner has 3 neighboring faces
    this part is hard-coded
    """

    corner_face_idx = [[4, 1, 2],  # corner of face 4,1,2
                       [4, 0, 2],
                       [4, 1, 3],
                       [4, 0, 3],
                       [5, 1, 2],
                       [5, 0, 2],
                       [5, 1, 3],
                       [5, 0, 3], ]
    corner_pixel_i = [[0, 0, -1],  # for face 4, i is 0, for face 1, i is 0, for face 2, i is -1
                      [0, 0, -1],
                      [-1, -1, 0],
                      [-1, -1, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [-1, -1, -1],
                      [-1, -1, -1], ]
    corner_pixel_j = [[0, -1, 0],
                      [-1, 0, -1],
                      [0, -1, 0],
                      [-1, 0, -1],
                      [-1, 0, 0],
                      [0, -1, -1],
                      [-1, 0, 0],
                      [0, -1, -1], ]

    for i in range(len(corner_pixel_i)):
        corner_face_list = corner_face_idx[i]
        pixel_location_i = corner_pixel_i[i]
        pixel_location_j = corner_pixel_j[i]

        val = torch.zeros(extended_mipmap.shape[-1])

        for j in range(len(corner_face_list)):
            face_idx = corner_face_list[j]
            location_i = pixel_location_i[j]
            location_j = pixel_location_j[j]
            val += extended_mipmap[face_idx, location_i, location_j]

        # one third of the whole contribution will be given to one corner
        val /= 3

        for j in range(len(corner_face_list)):
            face_idx = corner_face_list[j]
            location_i = pixel_location_i[j]
            location_j = pixel_location_j[j]
            original_map[face_idx, location_i, location_j] += val

    return original_map


def bilerp_inverse(location, portion, u_right, u_left, v_up, v_bot):
    """

    :param location: the location: [N,2]
    :param portion: the portion of this sample
    :param u_right: the right u value
    :param u_left: the left u value
    :param v_up: the up v value
    :param v_bot: the bot v value
    :return:
    """

    u = location[:, 0]
    v = location[:, 1]

    u_len = u_right - u_left
    v_len = v_up - v_bot

    u0_portion = portion * (u_right - u) / u_len * (v - v_bot) / v_len
    u1_portion = portion * (u - u_left) / u_len * (v - v_bot) / v_len
    u2_portion = portion * (u_right - u) / u_len * (v_up - v) / v_len
    u3_portion = portion * (u - u_left) / u_len * (v_up - v) / v_len

    return u0_portion, u1_portion, u2_portion, u3_portion



def downsample_xyz_pattern_full(face_extended_res):

    xyz = torch.zeros((6, face_extended_res - 2, face_extended_res - 2, 3))

    for face_idx in range(6):
        uv = create_downsample_pattern(face_extended_res)
        xyz[face_idx] = torch_uv_to_xyz_vectorized(uv, face_idx)

    return xyz




def create_downsample_pattern(face_extended_res):
    """
    create the downsample pattern (the 1/4 3/4 location for bilerp), this is useful in computing Jacobian
    because every bilerp location is within the face, so we do not need the extended rows/cols
    :param extended_res:
    :return: return the uv index
    """
    #point pattern 1, start from 1.25 -> 3.25 -> 5.25 -> res - 2.75
    point_idx_pattern1 = torch.arange(5/4 ,face_extended_res - 3/4, 2)
    #point pattern 2, start from 2.75 -> 4.75 -> res - 1.25
    point_idx_pattern2 = torch.arange(2+3/4,face_extended_res - 1 - 1/4 + 2, 2)

    point_idx_pattern1 -= 1
    point_idx_pattern2 -= 1

    point_idx = torch.stack(( point_idx_pattern1, point_idx_pattern2), dim=1).reshape(-1)

    #point ys is u    point xs is v
    point_xs,point_ys = torch.meshgrid(torch.flip(point_idx, dims=[0]),point_idx,indexing='ij')

    uv = torch.stack((point_ys, point_xs), dim=-1)

    uv /= (face_extended_res - 2)

    return uv






def push_back(mipmaps):
    """
    Jacobian should be taken into consideration when pushing back,
    each 2*2 tile has a Jacobian weight
    each has a contribution of (1/8 + (1/2) * j[i] / (j[0] + j[1] + j[2] + j[3]))



    push all lower level values to the higher level
    :param mipmaps: a mipmap that is initialized to zero(if push back func is not called, all weight is zero)
    :return: the highest level of mipmap only, since all the values have been pushed back to this level
    """
    for level_idx in reversed(range(len(mipmaps))):
        if level_idx == 0:
            break

        cur_level = mipmaps[level_idx]
        res = level_to_res(level_idx, 7)
        extended_upper_res = res * 2 + 2
        extended_upper_level = torch.zeros((6, extended_upper_res, extended_upper_res, cur_level.shape[-1]))


        """
        xyz_bilerp_upper has a resolution of 2res * 2res, the neighboring 4 are the jacobian used for the bilerp samples
        """
        xyz_bilerp_upper = downsample_xyz_pattern_full(extended_upper_res)
        j = torch_jacobian_vertorized(xyz_bilerp_upper)
        j_sum = j.view(6,res,2,res,2).sum(dim=(2,4))

        # instead of looping through each element, we'd better loop through each kernel item
        """
        For four corner 1/64 contribution
        The pattern is:

        x - x - x - x - - - 
        - - - - - - - - - -
        x - x - x - x - - - 
        - - - - - - - - - -
        x - x - x - x - - - 
        - - - - - - - - - -
        x - x - x - x - - - 
        - - - - - - - - - -
        - - - - - - - - - -
        - - - - - - - - - -

        this is for upper-left pattern, the upper-right is simply moving to the right for 2 unit
        the bottom left is moving this down for 2 unit, the bottom right is moving this down and right both for 2 units

        """

        start_idx = [0, 3]

        cur_level_unit = cur_level / 64.0

        for row_start in start_idx:
            # Note that, the row_end here needs to be included(while python slice exclude row end
            # row_end = row_start + 2 * (extended_upper_res - 1)
            row_end = row_start + 2 * (res - 1) + 2
            for col_start in start_idx:
                col_end = col_start + 2 * (res - 1) + 2
                extended_upper_level[:, row_start:row_end:2, col_start:col_end:2] += cur_level_unit

        """
        For the four 3/64 contributions at left and right side
        """
        cur_level_unit = cur_level_unit * 3

        start_idx_row = [1, 2]
        start_idx_col = [0, 3]
        for row_start in start_idx_row:
            row_end = row_start + 2 * (res - 1) + 2
            for col_start in start_idx_col:
                col_end = col_start + 2 * (res - 1) + 2
                extended_upper_level[:, row_start:row_end:2, col_start:col_end:2] += cur_level_unit

        """
        For the four 3/64 contributions at up and bot side
        """
        start_idx_row, start_idx_col = start_idx_col, start_idx_row
        for row_start in start_idx_row:
            row_end = row_start + 2 * (res - 1) + 2
            for col_start in start_idx_col:
                col_end = col_start + 2 * (res - 1) + 2
                extended_upper_level[:, row_start:row_end:2, col_start:col_end:2] += cur_level_unit

        """
        For the centric 9/64 contributions in shape of 2 * 2 
        """
        cur_level_unit = cur_level_unit * 3
        tile = torch.repeat_interleave(torch.repeat_interleave(cur_level_unit, 2, dim=1), 2, dim=2)
        extended_upper_level[:, 1:-1, 1:-1] += tile

        """
        Add the pushed result to the level
        """
        upper_level_original = process_extended_face(extended_upper_level)
        mipmaps[level_idx - 1] += upper_level_original

    return mipmaps[0]


def compute_contribution(location, level, initial_weight, n_level):
    """
    compute the contribution of all trilinear samples
    :param location: the location: [N,3] in xyz form
    :param level: level of each trilinear sample [N,]
    :param initial_weight: the initial weight of each sample
    :param n_level: number of mipmap levels in total
    :return: a 6*res*res map
    """
    zero_map = initialize_mipmaps(n_level)
    trilinear_samples = process_trilinear_samples(location, level, n_level, initial_weight)
    bilinear_samples = process_bilinear_samples(trilinear_samples, zero_map)
    result = push_back(bilinear_samples)

    return result


def L1_error_one_texel(ggx_kernel, contribution_map):
    """
    Comparison should be made between *normalized* map
    :param ggx_kernel:
    :param contribution_map:
    :return:
    """

    l1_loss = torch.abs(ggx_kernel - contribution_map / torch.sum(contribution_map))
    return l1_loss



def test_optimize(ggx_alpha, res, texel_direction, n_sample_per_frame):
    ggx_ref = compute_ggx_distribution_reference(res, ggx_alpha, texel_direction)

    # normalize ggx kernel
    ggx_ref /= np.sum(ggx_ref)

    rng = np.random.default_rng(int(datetime.now().timestamp()))
    # initialize some random parameter very close to zero?
    param = rng.random(n_sample_per_frame * 3 * 5 * 3)
    # param /= 50.0

    result = scipy.optimize.minimize(error_func, param, args=(texel_direction, n_sample_per_frame, ggx_ref),
                                     method='BFGS', options={'disp': True, 'gtol': 1e-4})

    print("done")


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
    error = test_one_texel_full_optimization(texel_direction, n_sample_per_frame, ggx_ref, x, constant)
    return error


def test_one_texel_full_optimization(texel_direction, n_sample_per_frame, ggx_ref, coef_table=None, constant= False):
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
        frame_weight_list.append(torch_gen_frame_weight(unnormalized_texel_direction, i))
        theta_phi_list.append(torch_gen_theta_phi(unnormalized_texel_direction, i))

    all_directions = []
    all_levels = []
    all_weights = []

    sample_directions = torch.empty((0,3))
    sample_weights = torch.empty((0,))
    sample_levels = torch.empty((0,))


    for i in range(3):
        cur_frame_weight = frame_weight_list[i]

        #if frame_weight is 0, we skip, how to do this in parallel?
        if cur_frame_weight == 0:
            continue

        X, Y, Z = torch_gen_frame_xyz(texel_direction, i)

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

        sample_directions = torch.concatenate((sample_directions, sample_direction_map))
        sample_weights = torch.concatenate((sample_weights, weight_cur_frame))
        sample_levels = torch.concatenate((sample_levels, level))

        # all_weights.append(weight_cur_frame)
        # all_levels.append(level)
        # all_directions.append(sample_direction)


    result = compute_contribution(sample_directions, sample_levels, sample_weights, 7)

    e_arr = L1_error_one_texel(ggx_ref, result)

    e = torch.sum(e_arr)
    # e = np.average(e_arr)

    return e





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

def optimize_function():
    face = 4
    u,v = 0.8,0.2
    location_global = map_util.uv_to_xyz((u, v), face)
    location_global = location_global.reshape((1, -1))
    ggx_ref = compute_ggx_distribution_reference(128,0.1,location_global)
    location_global = torch.from_numpy(location_global)
    ggx_ref = torch.from_numpy(ggx_ref)
    #normalize GGX
    ggx_ref /= torch.sum(ggx_ref)

    constant = True
    n_sample_per_frame = 8
    if not constant:
        model = SimpleModel(n_sample_per_frame)
    else:
        model = ConstantModel(n_sample_per_frame)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    n_epoch = 10000

    with torch.autograd.set_detect_anomaly(True):
        for _ in range(n_epoch):
            optimizer.zero_grad()
            params = model()
            loss = error_func(params, location_global, n_sample_per_frame, ggx_ref, constant)
            loss.backward()
            optimizer.step()

            print("it{}:loss is{}".format(_,loss.item()))
    print("Done")






if __name__ == "__main__":
    #t = create_downsample_pattern(130)
    #t = torch_uv_to_xyz_vectorized(t,0)

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
    location_global = torch.from_numpy(location_global)
    #test_one_texel_full_optimization(location_global[0:1,:],8,ggx)
    #
    level_global = torch.tensor([6.0])
    n_level_global = 7
    initial_weight_global = torch.tensor([1.0])
    #
    t = initialize_mipmaps(n_level_global)
    sample_info = process_trilinear_samples(location_global, level_global, n_level_global,initial_weight_global)
    t = process_bilinear_samples(sample_info,t)
    final_image = push_back(t)

    print("Done")


