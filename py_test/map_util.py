from datetime import datetime

import numpy as np
import numba

"""
currently long lat assume +Y is up

which means tho = ar
"""

def xyz_to_latlon(xyz:np.ndarray):
    """
    :param xyz: normalized vector xyz
    :return:
    """

    x,y,z = xyz[0],xyz[1],xyz[2]

    theta = np.arccos(y)


    phi = np.arctan2(x,z)

    # move phi(longitude to positive)
    # if phi < 0.0:
    #     phi += 2.0*np.pi

    return theta,phi


def xyz_to_latlon_vectorized(xyz:np.ndarray):
    """

    :param xyz: vector in shape of (N,3)
    :return: theta,phi pair, each a vector of (N,)
    """
    x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]
    theta = np.arccos(y)
    #arctan2(x,z) generate the same longitude in the paper
    phi = np.arctan2(x,z)
    #phi = np.where(phi < 0.0, phi + np.pi *2 , phi)

    return theta,phi


def latlon_to_xyz(lat,lon):
    """
    :param lon: longitude
    :param lat: latitude
    :return: normalized direction vector
    """
    y = np.cos(lat)
    x = np.sin(lat)*np.cos(lon)
    z = np.sin(lat)*np.sin(lon)

    return np.array([x,y,z])


def xyz_to_uv_vectorized(xyz:np.ndarray):
    """

    :param xyz: array of xyz. xyz.shape[-1] must be 3(indicating xyz)
    :return:
    """
    x,y,z = xyz[...,0],xyz[...,1],xyz[...,2]
    abs_x = np.abs(x)
    abs_y = np.abs(y)
    abs_z = np.abs(z)

    is_x_positive = x > 0
    is_y_positive = y > 0
    is_z_positive = z > 0

    face_idx = np.zeros(xyz.shape[:-1])
    u_idx = np.zeros_like(face_idx)
    v_idx = np.zeros_like(face_idx)
    max_axis = np.zeros_like(face_idx)

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




def xyz_to_uv(xyz:np.ndarray):
    """
    :param xyz: normalized vector xyz
    :return: (u,v),face index
    """
    x,y,z = xyz[0],xyz[1],xyz[2]
    abs_x = np.abs(x)
    abs_y = np.abs(y)
    abs_z = np.abs(z)

    is_x_positive = True if x > 0 else False
    is_y_positive = True if y > 0 else False
    is_z_positive = True if z > 0 else False


    if is_x_positive and abs_x >= abs_y and abs_x >= abs_z:
        # positive x face
        max_axis = abs_x
        uc = -z
        vc = y
        index = 0
    elif not is_x_positive and abs_x >= abs_y and abs_x >= abs_z:
        # negative x face
        max_axis = abs_x
        uc = z
        vc = y
        index = 1
    elif is_y_positive and abs_y >= abs_z and abs_y >= abs_x:
        # positive y face
        max_axis = abs_y
        uc = x
        vc = -z
        index = 2
    elif not is_y_positive and abs_y >= abs_z and abs_y >= abs_x:
        # negative y face
        max_axis = abs_y
        uc = x
        vc = z
        index = 3
    elif is_z_positive and abs_z >= abs_y and abs_z >= abs_x:
        # positive z face
        max_axis = abs_z
        uc = x
        vc = y
        index = 4
    elif not is_z_positive and abs_z >= abs_y and abs_z >= abs_x:
        # negative z face
        max_axis = abs_z
        uc = -x
        vc = y
        index = 5
    else:
        raise NotImplementedError

    # convert range to [0,1]
    u = 0.5 * (uc / max_axis + 1.0)
    v = 0.5 * (vc / max_axis + 1.0)

    return (u,v),index


def uv_to_xyz_vectorized(uv:np.ndarray,idx, normalize_flag=False):
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
        x = 1.0
        y = vc
        z = -uc
    elif idx == 1:
        x = -1.0
        y = vc
        z = uc
    elif idx == 2:
        x = uc
        y = 1.0
        z = -vc
    elif idx == 3:
        x = uc
        y = -1.0
        z = vc
    elif idx == 4:
        x = uc
        y = vc
        z = 1.0
    elif idx == 5:
        x = -uc
        y = vc
        z = -1.0
    else:
        raise NotImplementedError



    if uv.ndim == 2:
        vec = np.zeros((uv.shape[0], 3))
        vec[:, 0] = x
        vec[:, 1] = y
        vec[:, 2] = z
        # normalize xyz
        if normalize_flag:
            norm = np.linalg.norm(vec,axis=1)
            norm = np.tile(norm, (3, 1)).T
            vec = vec / norm
    elif uv.ndim==3:
        vec = np.zeros((uv.shape[0], uv.shape[1],3))
        vec[:,:,0] = x
        vec[:,:,1] = y
        vec[:,:,2] = z
        if normalize_flag:
            norm = np.linalg.norm(vec,axis=2)
            norm = np.stack((norm,norm,norm),axis=2)
            vec = vec / norm
    else:
        raise NotImplementedError

    return vec




def uv_to_xyz(uv, idx):
    """

    :param uv: u,v coordinate
    :param idx: face index
    :return: normalized 3D vector
    """
    u,v = uv
    uc = 2.0 * u - 1.0
    vc = 2.0 * v - 1.0

    if idx == 0:
        x = 1.0
        y = vc
        z = -uc
    elif idx == 1:
        x = -1.0
        y = vc
        z = uc
    elif idx == 2:
        x = uc
        y = 1.0
        z = -vc
    elif idx == 3:
        x = uc
        y = -1.0
        z = vc
    elif idx == 4:
        x = uc
        y = vc
        z = 1.0
    elif idx == 5:
        x = -uc
        y = vc
        z = -1.0
    else:
        raise NotImplementedError

    # normalize xyz
    vec = np.array([x,y,z])
    vec = vec / np.linalg.norm(vec)

    return vec

def jacobian_vertorized(xyz):
    """

    :param xyz: in shape (N,3) or (M,N,3)
    :return:
    """
    if xyz.ndim == 2:
        power_2 = xyz * xyz
        sum_xyz = np.sum(power_2, axis=1)
        j = 1 / np.pow(sum_xyz,3/2)
    elif xyz.ndim == 3:
        power_2 = xyz * xyz
        sum_xyz = np.sum(power_2, axis=2)
        j = 1 / np.pow(sum_xyz,3/2)
    elif xyz.ndim == 4:
        power_2 = xyz * xyz
        sum_xyz = np.sum(power_2, axis=3)
        j = 1 / np.pow(sum_xyz,3/2)
    else:
        raise NotImplementedError

    return j


def jacobian(xyz):
    x,y,z = xyz[0],xyz[1],xyz[2]
    return 1 / np.pow((x**2+y**2+z**2),3/2)



def create_pixel_index(resolution,dimension):
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
        idx_array = np.arange(0.5, resolution+0.5,1.0)
        return idx_array
    elif dimension == 2:
        row_res = resolution[0]
        col_res = resolution[1]
        row_array = np.arange(0.5, row_res+0.5,1.0)
        col_array = np.arange(0.5, col_res+0.5,1.0)
        # xv is row, yv is col
        xv,yv = np.meshgrid((row_array,col_array),indexing='ij')
        return xv,yv



def dot_vectorized_3D(v1, v2):
    """
    :param v1: v1
    :param v2: v2 both in shape of (M,N,3)
    :return:
    """
    result = np.einsum('ijk,ijk->ij', v1, v2)
    return result


def dot_vectorized_4D(v1, v2):
    """
    :param v1: v1
    :param v2: v2 both in shape of (6,M,M,3)
    :return:
    """
    result = np.einsum('ijkl,ijkl->ijk', v1, v2)
    return result

def dot_vectorized_2D(v1,v2):
    result = np.einsum('ij,ij->i', v1, v2)
    return result



def random_dir_sphere(uv = None, n_dir = 1):
    if uv is None:
        rng = np.random.default_rng(int(datetime.now().timestamp()))
        u = rng.random(n_dir)
        v = rng.random(n_dir)
    else:
        assert uv.shape[1] == 2 and uv.shape[0] == n_dir
        u = uv[:, 0]
        v = uv[:, 1]

    #uniformly sample direction from a sphere
    phi = u * 2 * np.pi
    v *= 2
    v -= 1
    cos_theta = v
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)

    x = np.cos(phi) * sin_theta
    y = np.sin(phi) * sin_theta
    z = cos_theta

    return np.stack((x, y, z), axis=1)


def dir_to_cube_coordinate(xyz):
    max = np.max(np.abs(xyz),axis=-1)
    max_stack = np.stack((max,max,max),axis=-1)
    return xyz/max_stack


def random_dir_cube(uv = None, n_dir = 1):
    xyz = random_dir_sphere(uv,n_dir)
    xyz_cube = dir_to_cube_coordinate(xyz)
    return xyz_cube



def sample_location(n_sample_per_level, rng = None):
    """
    Generate directions for optimization
    Since we can not afford computing error for all direction, we need to sample a few texel directions
    :param n_sample_per_level:
    :param rng:
    :return:
    """
    if rng is None:
        rng = np.random.default_rng(int(datetime.now().timestamp()))
    uv = rng.random((n_sample_per_level, 2))
    xyz = random_dir_cube(uv, n_sample_per_level)
    return xyz



def is_level(ggx_alpha,texel_direction):
    pass




if __name__ == "__main__":
    face = 4
    u,v = 0.8,0.2
    location_global = uv_to_xyz((u, v), face)
    print(location_global)