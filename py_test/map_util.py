from datetime import datetime

import numpy as np
import numba
from networkx import node_link_data

"""
currently long lat assume +Y is up

which means tho = ar
"""

def get_reflected_vector_vectorized(n: np.ndarray, wi: np.ndarray):
    """

    :param n: [N,3] or [3,] normal, should be normalized
    :param wi: [N,3] or [3,]  in most cases, wi we have is the direction that is pointing outward. Need to negate it
    :return:
    """
    if wi.ndim == 1:
        num_leading_dims = n.ndim - 1  # Subtract 1 because the last dimension is 3
        # Create the new shape: (1, 1, ..., 3)
        new_shape = (1,) * num_leading_dims + (3,)
        wi = wi.reshape(new_shape)

    wi_neg = -wi
    wi_neg = wi_neg / np.linalg.norm(wi_neg, axis=-1, keepdims=True)
    #n_normalized = n / torch.linalg.norm(n, dim=-1, keepdim=True)

    r = wi_neg - 2 * np.sum(wi_neg * n, axis=-1, keepdims=True) * n


    return r


def get_half_vector_vectorized(l,v):
    """
    :param l: [6,128,128,3]
    :param v: [3,]
    :return:
    """
    num_leading_dims = l.ndim - 1  # Subtract 1 because the last dimension is 3
    # Create the new shape: (1, 1, ..., 3)
    new_shape = (1,) * num_leading_dims + (3,)
    v = v.reshape(new_shape)

    wh = v + l
    wh = wh / np.linalg.norm(wh, axis=-1, keepdims=True)
    return wh



def rotate_90degree_awayfrom_n(vectors,normals):
    """
    Rotate the vector 90 degree in the surface of vector and normal, Should be rotated away from normal
    This can be implemented by two cross product. However, we need to handle edge cases when vector is close to normal
    :param vectors:[N,3] should all be normalized
    :param normals:[N,3] should all be normalized
    :return:
    """
    batch_size = vectors.shape[0]

    #determine near-parallel case
    threshold = 0.99995
    cosine = np.sum(vectors * normals, axis=-1, keepdims=True)
    near_parallel = cosine > threshold


    tmp = np.cross(vectors, normals)
    tmp = tmp / np.linalg.norm(tmp, axis = -1, keepdims = True)

    rotated_vectors = np.cross(vectors, tmp)



    # if np.any(near_parallel):
    #     raise NotImplementedError
    rotated_vectors = rotated_vectors / np.linalg.norm(rotated_vectors, axis = -1, keepdims = True)

    return rotated_vectors



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
        j = 1 / np.power(sum_xyz,3/2)
    elif xyz.ndim == 3:
        power_2 = xyz * xyz
        sum_xyz = np.sum(power_2, axis=2)
        j = 1 / np.power(sum_xyz,3/2)
    elif xyz.ndim == 4:
        power_2 = xyz * xyz
        sum_xyz = np.sum(power_2, axis=3)
        j = 1 / np.power(sum_xyz,3/2)
    else:
        raise NotImplementedError

    return j


def jacobian(xyz):
    x,y,z = xyz[0],xyz[1],xyz[2]
    return 1 / np.power((x**2+y**2+z**2),3/2)



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

def random_dir_hemisphere(uv = None, n_dir = 1):
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


def gen_face_uv(n_res):
    uv_table = np.zeros((n_res,n_res,2))
    uv_ascending_order = create_pixel_index(n_res,1)
    uv_ascending_order /= n_res
    # yv for u, xv for v
    xv, yv = np.meshgrid(np.flip(uv_ascending_order),uv_ascending_order,indexing='ij')
    uv_table[:,:,0] = yv
    uv_table[:,:,1] = xv
    return uv_table


def texel_directions(n_res):
    """
    Generate a (6,n_res,n_res,3) xyz direction table for a given level of cubemap(no coordinate system change)
    :param faces:
    :param n_res:
    :return:
    """
    face_uv = gen_face_uv(n_res)
    faces_xyz = np.zeros((6,n_res,n_res,3))
    for face_idx in range(6):
        face_xyz = uv_to_xyz_vectorized(face_uv, face_idx,False)
        faces_xyz[face_idx] = face_xyz
    return faces_xyz


def normalized(a, axis=-1, order=2):
    # https://stackoverflow.com/a/21032099
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def frame_axis_index(frame_idx, paper = False):
    """
    There are three frames, the first one consider x the up-axis. the second one consider y the up-axis
    the third one consider z the up-axis.
    Note: all system should use the same convention, we choose right-hand here
    :param frame_idx: the index of the frame. 0 means x as up, 1 means y as up, 2 means z as up
    :param paper: whether to follow what the code provided by the author does, there is a mismatch in axis index
    :return:
    """
    if not paper:
        # the original z-axis in a traditional coordinate system
        up_axis = [0,1,2]
        # the original x-axis in a traditional coordinate system
        other_axis0 = [1,2,0]
        # the original y-axis in a traditional coordinate system
        other_axis1 = [2,0,1]
    else:
        up_axis = [0,1,2]
        other_axis0 = [1,0,0]
        other_axis1 = [2,2,1]

    return other_axis0[frame_idx],other_axis1[frame_idx],up_axis[frame_idx]



def gen_frame_xyz(faces_xyz, frame_idx):
    """
    The frame xyz is used exclusively in sampling parameters
    The original direction of the texel is considered the Z axis, we note the normal of this face a
    The X axis is cross(a,z)  The y axis is cross(Z,X)
    :param faces_xyz: (6,res,res,3)
    :param frame_idx: this affects how we construct the up vector
    :return:
    """
    Z = normalized(faces_xyz, axis=-1)
    polar_axis = np.zeros_like(Z)
    if frame_idx == 0 or frame_idx == 1 or frame_idx == 2:
        polar_axis[...,frame_idx] = 1.0
    else:
        raise NotImplementedError

    X = normalized(np.cross(polar_axis, Z),axis=-1)

    # This is guaranteed to be unit vector
    Y = np.cross(Z,X)

    return X,Y,Z


def gen_anisotropic_frame_xyz(faces_xyz_normalized, view_directions):
    """
    All input should already be normalized
    :param faces_xyz_normalized:
    :param view_directions:
    :return:
    """

    reflect_direction = get_reflected_vector_vectorized(faces_xyz_normalized, view_directions)
    Z = reflect_direction / np.linalg.norm(reflect_direction,axis=-1,keepdims=True)

    X = rotate_90degree_awayfrom_n(Z, faces_xyz_normalized)
    X = X / np.linalg.norm(X, axis=-1, keepdims=True)

    Y = np.cross(Z, X)

    return X, Y, Z



def gen_frame_weight(facex_xyz, frame_idx, follow_code = False):
    """
    Compute frame weight for each texel according to the paper, the up/bot face have little weight
    :param facex_xyz:
    :param frame_idx:
    :return:
    """
    new_x_idx, new_y_idx, new_z_idx = frame_axis_index(frame_idx, follow_code)
    faces_xyz_abs = np.abs(facex_xyz)

    frame_weight = np.clip(4 * np.maximum(faces_xyz_abs[...,new_x_idx], faces_xyz_abs[...,new_y_idx]) - 3,0.0,1.0)

    return frame_weight


def gen_theta_phi(faces_xyz,frame_idx, follow_code = False):
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
    max_xy = np.maximum(np.abs(nx),np.abs(ny))

    #normalize nx,ny, in 2/3 of the cases, one of nx and ny should be 1 without normalizing it
    nx = nx / max_xy
    ny = ny / max_xy



    theta = np.zeros_like(nx)
    theta[(ny < nx) & (ny <= -0.999)] = nx[(ny < nx) & (ny <= -0.999)]
    theta[(ny < nx) & (ny > -0.999)] = ny[(ny < nx) & (ny > -0.999)]
    theta[(nx <= ny) & (ny >= 0.999)] = -nx[(nx <= ny) & (ny >= 0.999)]
    theta[(nx <= ny) & (ny < 0.999)] = -ny[(nx <= ny) & (ny < 0.999)]


    phi = np.zeros_like(nx)
    phi[nz <= -0.999] = -max_xy[nz <= -0.999]
    phi[nz >= 0.999] = max_xy[nz >= 0.999]
    phi[(nz > -0.999) & (nz < 0.999)] = nz[(nz > -0.999) & (nz < 0.999)]

    theta2 = theta * theta
    phi2 = phi * phi

    return theta,phi,theta2,phi2


def gen_theta_phi_no_frame(facex_xyz):
    u,v,face = xyz_to_uv_vectorized(facex_xyz)

    u =  2 * u - 1
    v =  2 * v - 1

    return u,v,u*u,v*v


def fixed_view_diretory_name(ggx_alpha, n_sample_per_frame ,n_sample_per_level,constant,adjust_level, optim_method:str ,
                 random_shuffle = False, allow_neg_weight = False, ggx_ref_jac_weight = 'None',view_dependent = False,
                 view_option_str = "None",reflection_parameterization = False,view_ndf_clipping = False,use_vndf = False,post_fix_for_dirs = None,
                 fixed_cos_view_theta = None, fixed_cos_view_theta_res = None):
    if fixed_cos_view_theta is not None:
        fixed_view_dir_name = 'fixed_costheta_res{}_ggx{:.3f}'.format(fixed_cos_view_theta_res,ggx_alpha)
        if constant:
            fixed_view_dir_name = fixed_view_dir_name + "_constant" + str(n_sample_per_frame)
        else:
            fixed_view_dir_name = fixed_view_dir_name + "_quad" + str(n_sample_per_frame)
        if view_dependent:
            fixed_view_dir_name = fixed_view_dir_name + "_view"
            fixed_view_dir_name = fixed_view_dir_name + "_" + view_option_str
            if reflection_parameterization:
                fixed_view_dir_name = fixed_view_dir_name + "_reflectParam"
            if view_ndf_clipping:
                fixed_view_dir_name = fixed_view_dir_name + "_ndfclip"
            if use_vndf:
                fixed_view_dir_name = fixed_view_dir_name + "_vndf"

        if adjust_level:
            fixed_view_dir_name = fixed_view_dir_name + "_ladj"

        fixed_view_dir_name = fixed_view_dir_name + "_" + optim_method

        if random_shuffle:
            fixed_view_dir_name = fixed_view_dir_name + "_randomdir"

        if allow_neg_weight:
            fixed_view_dir_name = fixed_view_dir_name + "_negweight"

        if ggx_ref_jac_weight.lower() != 'none':
            fixed_view_dir_name = fixed_view_dir_name + "_jacref" + ggx_ref_jac_weight.lower()

        if post_fix_for_dirs is not None:
            fixed_view_dir_name = fixed_view_dir_name + "_" + post_fix_for_dirs

        return fixed_view_dir_name
    else:
        raise NotImplementedError



def log_merl_filename(merl_name, n_sample_per_frame, n_sample_per_level, constant, adjust_level, optim_method:str,
                      random_shuffle = False, allow_neg_weight = False, ref_jac_weight ='None', ndf_clipping = False):
    name = 'optim_info_merl_' + merl_name + "_" + str(n_sample_per_level)

    if constant:
        name = name + "_constant" + str(n_sample_per_frame)
    else:
        name = name + "_quad" + str(n_sample_per_frame)

    if ndf_clipping:
        name = name + "_ndfclip"

    if adjust_level:
        name = name + "_ladj"

    name = name + "_" + optim_method

    if random_shuffle:
        name = name + "_randomdir"

    if allow_neg_weight:
        name = name + "_negweight"

    if ref_jac_weight.lower() != 'none':
        name = name + "_jacref" + ref_jac_weight.lower()

    name = name + '.log'

    return name


def model_merl_filename(merl_name, n_sample_per_frame, n_sample_per_level, constant, adjust_level, optim_method: str,
                        random_shuffle=False, allow_neg_weight=False, ref_jac_weight='None', ndf_clipping=False):
    if constant:
        model_name = "constant"+ str(n_sample_per_frame)  + "_merl_" + merl_name + "_" + str(n_sample_per_level)
    else:
        model_name = "quad"+ str(n_sample_per_frame)  + "_merl_" + merl_name + "_" + str(n_sample_per_level)

    if ndf_clipping:
        model_name = model_name + "_ndfclip"

    if adjust_level:
        model_name = model_name + "_ladj"

    model_name = model_name + "_" + optim_method

    if random_shuffle:
        model_name = model_name + "_randomdir"

    if allow_neg_weight:
        model_name = model_name + "_negweight"

    if ref_jac_weight.lower() != 'none':
        model_name = model_name + "_jacref" + ref_jac_weight.lower()

    return model_name



def log_filename(ggx_alpha, n_sample_per_frame ,n_sample_per_level,constant,adjust_level, optim_method:str ,
                 random_shuffle = False, allow_neg_weight = False, ggx_ref_jac_weight = 'None',view_dependent = False,
                 view_option_str = "None",reflection_parameterization = False,view_ndf_clipping = False,use_vndf = False,post_fix_for_dirs = None,
                 fixed_cos_view_theta = None):
    log_name = 'optim_info_multi_ggx_' + "{:.3f}".format(ggx_alpha) + "_" + str(n_sample_per_level)

    if constant:
        log_name = log_name + "_constant" + str(n_sample_per_frame)
    else:
        log_name = log_name + "_quad" + str(n_sample_per_frame)

    if view_dependent:
        log_name = log_name + "_view"
        log_name = log_name + "_" + view_option_str
        if reflection_parameterization:
            log_name = log_name + "_reflectParam"
        if view_ndf_clipping:
            log_name = log_name + "_ndfclip"
        if use_vndf:
            log_name = log_name + "_vndf"
        if fixed_cos_view_theta is not None:
            assert type(fixed_cos_view_theta) == float
            log_name = log_name + "_fixedcos{:.3f}".format(fixed_cos_view_theta)

    if adjust_level:
        log_name = log_name + "_ladj"

    log_name = log_name + "_" + optim_method

    if random_shuffle:
        log_name = log_name + "_randomdir"

    if allow_neg_weight:
        log_name = log_name + "_negweight"

    if ggx_ref_jac_weight.lower() != 'none':
        log_name = log_name + "_jacref" + ggx_ref_jac_weight.lower()

    if post_fix_for_dirs is not None:
        log_name = log_name + "_" + post_fix_for_dirs

    log_name = log_name + '.log'

    return log_name

def dir_filename(ggx_alpha, constant, n_sample_per_frame ,n_sample_per_level, adjust_level, optim_method:str ,
                 allow_neg_weight = False, ggx_ref_jac_weight = 'None',view_dependent = False,view_option_str = "None",
                 reflection_parameterization = False,view_ndf_clipping = False,use_vndf = False,post_fix_for_dirs = None,
                 fixed_cos_view_theta = None):
    if constant:
        dir_name = "constant"+ str(n_sample_per_frame)  + "_ggx_multi_" + "{:.3f}".format(ggx_alpha) + "_" + str(n_sample_per_level)
    else:
        dir_name = "quad"+ str(n_sample_per_frame)  + "_ggx_multi_" + "{:.3f}".format(ggx_alpha) + "_" + str(n_sample_per_level)

    if view_dependent:
        dir_name = dir_name + "_view"
        dir_name = dir_name + "_" + view_option_str
        if reflection_parameterization:
            dir_name = dir_name + "_reflectParam"
        if view_ndf_clipping:
            dir_name = dir_name + "_ndfclip"
        if use_vndf:
            dir_name = dir_name + "_vndf"
        if fixed_cos_view_theta is not None:
            assert type(fixed_cos_view_theta) == float
            dir_name = dir_name + "_fixedcos{:.3f}".format(fixed_cos_view_theta)

    if adjust_level:
        dir_name = dir_name + "_ladj"

    dir_name = dir_name + "_" + optim_method

    if allow_neg_weight:
        dir_name = dir_name + "_negweight"

    if ggx_ref_jac_weight.lower() != 'none':
        dir_name = dir_name + "_jacref" + ggx_ref_jac_weight.lower()

    dir_name = dir_name + "_dirs"
    if post_fix_for_dirs is not None:
        dir_name = dir_name + "_" + post_fix_for_dirs
    dir_name = dir_name + ".pt"

    return dir_name

def model_filename(ggx_alpha, constant, n_sample_per_frame ,n_sample_per_level, adjust_level, optim_method:str ,
                   random_shuffle = False, allow_neg_weight = False, ggx_ref_jac_weight = 'None',
                   view_dependent = False,view_option_str = "None",reflection_parameterization = False,view_ndf_clipping = False,
                   use_vndf = False,post_fix_for_dirs = None, fixed_cos_view_theta = None):
    if constant:
        model_name = "constant"+ str(n_sample_per_frame)  + "_ggx_multi_" + "{:.3f}".format(ggx_alpha) + "_" + str(n_sample_per_level)
    else:
        model_name = "quad"+ str(n_sample_per_frame)  + "_ggx_multi_" + "{:.3f}".format(ggx_alpha) + "_" + str(n_sample_per_level)

    if view_dependent:
        model_name = model_name + "_view"
        model_name = model_name + "_" + view_option_str
        if reflection_parameterization:
            model_name = model_name + "_reflectParam"
        if view_ndf_clipping:
            model_name = model_name + "_ndfclip"
        if use_vndf:
            model_name = model_name + "_vndf"
        if fixed_cos_view_theta is not None:
            assert type(fixed_cos_view_theta) == float
            model_name = model_name + "_fixedcos{:.3f}".format(fixed_cos_view_theta)

    if adjust_level:
        model_name = model_name + "_ladj"

    model_name = model_name + "_" + optim_method

    if random_shuffle:
        model_name = model_name + "_randomdir"

    if allow_neg_weight:
        model_name = model_name + "_negweight"

    if ggx_ref_jac_weight.lower() != 'none':
        model_name = model_name + "_jacref" + ggx_ref_jac_weight.lower()

    if post_fix_for_dirs is not None:
        model_name = model_name + "_" + post_fix_for_dirs

    return model_name


def write_dict_to_txt(dictionary, filename, delimiter=' '):
    """
    Writes a dictionary to a plain text file with each key-value pair on a separate line.

    Parameters:
    - dictionary (dict): The dictionary to write. Keys should be strings, and values should be floats.
    - filename (str): The path to the file where the dictionary will be saved.
    - delimiter (str): The string used to separate keys and values. Default is a space.
    """
    try:
        with open(filename, 'w') as file:
            for key, value in dictionary.items():
                file.write(f"{key}{delimiter}{value}\n")
        print(f"Dictionary successfully saved to '{filename}'.")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

def read_txt_to_dict(filename, delimiter=' '):
    """
    Reads a plain text file and converts it back into a dictionary.

    Parameters:
    - filename (str): The path to the file to read.
    - delimiter (str): The string used to separate keys and values. Must match the delimiter used during writing.

    Returns:
    - dict: The reconstructed dictionary.
    """
    reconstructed_dict = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                key, value = line.strip().split(delimiter)
                reconstructed_dict[key] = float(value)  # Convert value back to float
        print(f"Dictionary successfully loaded from '{filename}'.")
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")
    except ValueError as ve:
        print(f"Value conversion error: {ve}")
    return reconstructed_dict

if __name__ == "__main__":
    face = 4
    u,v = 0.8,0.2
    location_global = uv_to_xyz((u, v), face)
    print(location_global)