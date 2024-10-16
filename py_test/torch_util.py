import torch
from filter import frame_axis_index
from interpolation import get_edge_information
import numpy as np

chan_count = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def level_to_res(level, n_level):
    """
    Get this level's mipmap resolution. Assume the lowest resolution is 2*2
    :param level: the level of mipmap, in shape of [N,]
    :param n_level: how many levels in mipmap
    :return: the resolution
    """
    return np.left_shift(2, int(n_level - 1 - level))


def initialize_mipmaps(n_level):
    """
    initialize an all zero mipmap chain
    :param n_level:
    :return:
    """
    mipmaps = []
    for i in range(n_level):
        res = level_to_res(i, n_level)
        tmp = torch.zeros((6, res, res, chan_count), device=device)
        mipmaps.append(tmp)
    return mipmaps


def torch_jacobian_vertorized(xyz):
    """

    :param xyz: in shape (N,3) or (M,N,3)
    :return:
    """
    power_2 = xyz * xyz
    sum_xyz = torch.sum(power_2, dim=-1)
    j = 1 / torch.pow(sum_xyz, 3/2)

    return j

def torch_dot_vectorized_4D(v1,v2):
    result = torch.einsum('ijkl,ijkl->ijk', v1, v2)
    return result

def torch_dot_vectorized_2D(v1,v2):
    result = torch.einsum('ij,ij->i', v1, v2)
    return result


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
        x = torch.ones_like(uc, device=device)
        y = vc
        z = -uc
    elif idx == 1:
        x = -torch.ones_like(uc, device=device)
        y = vc
        z = uc
    elif idx == 2:
        x = uc
        y = torch.ones_like(uc, device=device)
        z = -vc
    elif idx == 3:
        x = uc
        y = -torch.ones_like(uc, device=device)
        z = vc
    elif idx == 4:
        x = uc
        y = vc
        z = torch.ones_like(uc, device=device)
    elif idx == 5:
        x = -uc
        y = vc
        z = -torch.ones_like(uc, device=device)
    else:
        raise NotImplementedError


    vec = torch.stack((x, y, z), dim=-1)
    if normalize_flag:
        vec = torch_normalized(vec)

    return vec





def torch_normalized(a, axis=-1, order=2):
    # https://stackoverflow.com/a/21032099
    norm = torch.linalg.norm(a, ord=order, dim = axis)

    norm = torch.stack((norm,norm,norm),dim=axis)

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
    polar_axis = torch.zeros_like(Z, device=device)
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



    theta = torch.zeros_like(nx, device=device)
    theta[(ny < nx) & (ny <= -0.999)] = nx[(ny < nx) & (ny <= -0.999)]
    theta[(ny < nx) & (ny > -0.999)] = ny[(ny < nx) & (ny > -0.999)]
    theta[(nx <= ny) & (ny >= 0.999)] = -nx[(nx <= ny) & (ny >= 0.999)]
    theta[(nx <= ny) & (ny < 0.999)] = -ny[(nx <= ny) & (ny < 0.999)]


    phi = torch.zeros_like(nx, device=device)
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
        idx_array = torch.arange(0.5, resolution+0.5,1.0, device=device)
        return idx_array
    elif dimension == 2:
        row_res = resolution[0]
        col_res = resolution[1]
        row_array = torch.arange(0.5, row_res+0.5,1.0, device=device)
        col_array = torch.arange(0.5, col_res+0.5,1.0, device=device)
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
    uv = torch.zeros((cubemap_res,2), device=device)
    uv_ascending_order = torch_create_pixel_index(cubemap_res,1)

    uv_ascending_order = uv_ascending_order / cubemap_res

    uv_min = uv_ascending_order.min().detach().item()
    uv_max = uv_ascending_order.max().detach().item()

    epsilon = 1.0 - uv_max
    assert 1.0 - uv_max == uv_min


    if edge_side == "L":
        # u is 0 while v goes from 1 to 0
        uv[:,0] = torch.full((cubemap_res,),uv_min - 2 * epsilon, device=device)
        uv[:,1] = torch.flip(uv_ascending_order,[0])
    elif edge_side == "R":
        # u is 1 while v gose from 1 to 0
        uv[:,0] = torch.full((cubemap_res,),uv_max + 2 * epsilon, device=device)
        uv[:,1] = torch.flip(uv_ascending_order,[0])
    elif edge_side == "U":
        # v is 1 while u goes from 0 to 1
        uv[:,0] = uv_ascending_order
        uv[:,1] = torch.full((cubemap_res,),uv_max + 2 * epsilon, device=device)
    elif edge_side == "D":
        # v is 0 while u goes from 0 to 1
        uv[:,0] = uv_ascending_order
        uv[:,1] = torch.full((cubemap_res,),uv_min - 2 * epsilon, device=device)
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
    uv_table = torch.zeros((face_extended_res, face_extended_res, 2), device=device)
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

    face_idx = torch.zeros(xyz.shape[:-1],dtype = torch.int, device=device)
    u_idx = torch.zeros_like(face_idx,dtype=abs_x.dtype, device=device)
    v_idx = torch.zeros_like(face_idx,dtype=abs_x.dtype, device=device)
    max_axis = torch.zeros_like(face_idx,dtype=abs_x.dtype, device=device)

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
    #level = torch.clip(level, 0, n_level - 1)

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
        extended_mipmap_cur_level = torch.zeros((6, cur_res + 2, cur_res + 2, mipmaps[level_idx].shape[-1]),dtype=p0.dtype, device=device)

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
    original_map = torch.zeros((6, original_res, original_res, extended_mipmap.shape[-1]), device=device)

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

        val = torch.zeros(extended_mipmap.shape[-1], device=device)

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

    xyz = torch.zeros((6, face_extended_res - 2, face_extended_res - 2, 3), device=device)

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
    point_idx_pattern1 = torch.arange(5/4 ,face_extended_res - 3/4, 2, device=device)
    #point pattern 2, start from 2.75 -> 4.75 -> res - 1.25
    point_idx_pattern2 = torch.arange(2+3/4,face_extended_res - 1 - 1/4 + 2, 2, device=device)

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

    for 1/64 texel,
    the final contribution is (1/128) + 1/32 * (j[i]/ (j[0] + j[1] + j[2] + j[3]))


    for start_idx, it ranges from 0,3, whic covers all 4*4 texels
    [0:2,0:2] will use upper-left jacobian
    [0:2,2:4] will use upper-right jacobian
    [2:4,0:2] will use lower-left jacobian
    [2:4,2:4] will use lower-right jacobian

    TODO:do the above, current method is wrong



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
        extended_upper_level = torch.zeros((6, extended_upper_res, extended_upper_res, cur_level.shape[-1]), device=device)


        """
        xyz_bilerp_upper has a resolution of 2res * 2res, the neighboring 4 are the jacobian used for the bilerp samples
        """
        xyz_bilerp_upper = downsample_xyz_pattern_full(extended_upper_res)
        j = torch_jacobian_vertorized(xyz_bilerp_upper)
        j_sum = j.view(6,res,2,res,2).sum(dim=(2,4))

        j_pattern_upper_left = j[:,0::2,0::2]
        j_pattern_upper_right = j[:,0::2,1::2]
        j_pattern_bot_left = j[:,1::2,0::2]
        j_pattern_bot_right = j[:,1::2,1::2]

        j_selection = torch.zeros((2,2) + j_pattern_upper_left.shape, device=device)
        j_selection[0,0] = j_pattern_upper_left
        j_selection[0,1] = j_pattern_upper_right
        j_selection[1,0] = j_pattern_bot_left
        j_selection[1,1] = j_pattern_bot_right

        # since we are doing 1 element of each sample(and repeat 16 times), each element has the same jacobian location,
        # we can vectorize this, and we can precaculate the result

        cur_level_unit_selection = torch.zeros(j_selection.shape + (1,), device=device)
        cur_level_unit_selection[0,0] = cur_level * (1 / 128.0 + 1 / 32.0 * j_pattern_upper_left / j_sum).unsqueeze(-1)
        cur_level_unit_selection[0,1] = cur_level * (1 / 128.0 + 1 / 32.0 * j_pattern_upper_right / j_sum).unsqueeze(-1)
        cur_level_unit_selection[1,0] = cur_level * (1 / 128.0 + 1 / 32.0 * j_pattern_bot_left / j_sum).unsqueeze(-1)
        cur_level_unit_selection[1,1] = cur_level * (1 / 128.0 + 1 / 32.0 * j_pattern_bot_right / j_sum).unsqueeze(-1)

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
            """
            0,0 is upper-left(J[0,0])
            0,3 is upper-right(J[0,1])
            3,0 is bottom-left(J[1,0])
            3,3 is bottom-right(J[1,1])
            """

            # Note that, the row_end here needs to be included(while python slice exclude row end
            # row_end = row_start + 2 * (extended_upper_res - 1)
            row_end = row_start + 2 * (res - 1) + 2
            for col_start in start_idx:
                col_end = col_start + 2 * (res - 1) + 2
                #current_j = j[row_start%2,col_start%2]
                #extended_upper_level[:, row_start:row_end:2, col_start:col_end:2] += cur_level * (1 / 128.0 + 1 / 32.0 * current_j / j_sum).unsqueeze(-1)
                extended_upper_level[:, row_start:row_end:2, col_start:col_end:2] += cur_level_unit_selection[int(row_start/2),int(col_start/2)]

        """
        For the four 3/64 contributions at left and right side
        """
        #cur_level_unit = cur_level_unit * 3

        start_idx_row = [1, 2]
        start_idx_col = [0, 3]

        """
        1,0 is top-left(J[0,0]
        1,3 is top-right(J[0,1])
        2,0 is bot-left(J[1,0])
        2,3 is bot-right(J[1,1])
        """

        for row_start in start_idx_row:
            row_end = row_start + 2 * (res - 1) + 2
            for col_start in start_idx_col:
                col_end = col_start + 2 * (res - 1) + 2
                current_j = j[row_start%2,col_start%2]
                #extended_upper_level[:, row_start:row_end:2, col_start:col_end:2] += cur_level_unit
                extended_upper_level[:, row_start:row_end:2, col_start:col_end:2] += cur_level_unit_selection[int(row_start/2),int(col_start/2)] * 3

        """
        For the four 3/64 contributions at up and bot side
        """

        """
        0,1 is upper-left
        0,2 is upper-right
        3,1 is bot-left
        3,2 is bot-right
        """

        start_idx_row, start_idx_col = start_idx_col, start_idx_row
        for row_start in start_idx_row:
            row_end = row_start + 2 * (res - 1) + 2
            for col_start in start_idx_col:
                col_end = col_start + 2 * (res - 1) + 2
                #current_j = j[row_start%2,col_start%2]
                extended_upper_level[:, row_start:row_end:2, col_start:col_end:2] += cur_level_unit_selection[int(row_start/2),int(col_start/2)] * 3

        """
        For the centric 9/64 contributions in shape of 2 * 2 
        """
        #cur_level_unit = cur_level_unit * 3

        j_sum_ext = torch.repeat_interleave(torch.repeat_interleave(j_sum, 2, dim=1), 2, dim=2)
        tile = torch.repeat_interleave(torch.repeat_interleave(cur_level, 2, dim=1), 2, dim=2)
        cur_level_unit = 9 * tile * (1 / 128.0 + 1 / 32.0 * j / j_sum_ext).unsqueeze(-1)
        extended_upper_level[:, 1:-1, 1:-1] += cur_level_unit

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