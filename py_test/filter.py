"""
The second pass filtering
There are in total 7 levels of mipmap: 2^7 -> 2^1
"""
from cmath import polar

import numpy as np
import map_util
import mat_util
import coefficient
import interpolation
import image_read


def gen_tex_levels(n_level,n_high_res):
    texture_list = []
    cur_res = n_high_res
    for i in range(n_level):
        tex = np.zeros((6,cur_res,cur_res,3))
        texture_list.append(tex)
        cur_res = n_high_res / 2


def gen_face_uv(n_res):
    uv_table = np.zeros((n_res,n_res,2))
    uv_ascending_order = map_util.create_pixel_index(n_res,1)
    uv_ascending_order /= n_res
    # yv for u, xv for v
    xv, yv = np.meshgrid(np.flip(uv_ascending_order),uv_ascending_order,indexing='ij')
    uv_table[:,:,0] = yv
    uv_table[:,:,1] = xv
    return uv_table


def frame_axis_index(frame_idx):
    """
    There are three frames, the first one consider x the up-axis. the second one consider y the up-axis
    the third one consider z the up-axis.
    Note: all system should use the same convention, we choose right-hand here
    :param frame_idx: the index of the frame. 0 means x as up, 1 means y as up, 2 means z as up
    :return:
    """
    # the original z-axis in a traditional coordinate system
    up_axis = [0,1,2]
    # the original x-axis in a traditional coordinate system
    other_axis0 = [1,2,0]
    # the original y-axis in a traditional coordinate system
    other_axis1 = [2,0,1]

    return other_axis0[frame_idx],other_axis1[frame_idx],up_axis[frame_idx]


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
        face_xyz = map_util.uv_to_xyz_vectorized(face_uv, face_idx,False)
        faces_xyz[face_idx] = face_xyz
    return faces_xyz


def normalized(a, axis=-1, order=2):
    # https://stackoverflow.com/a/21032099
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)



def gen_frame_xyz(faces_xyz):
    """
    The frame xyz is used exclusively in sampling parameters
    The original direction of the texel is considered the Z axis, we note the normal of this face a
    The X axis is cross(a,z)  The y axis is cross(Z,X)
    :param faces_xyz: (6,res,res,3)
    :return:
    """
    Z = normalized(faces_xyz, axis=-1)
    polar_axis = np.zeros_like(Z)
    polar_axis[0:2,:,:,0] = 1
    polar_axis[2:4,:,:,1] = 1
    polar_axis[4:6,:,:,2] = 1

    X = np.cross(polar_axis, Z)
    Y = np.cross(Z,X)

    return X,Y,Z


def gen_frame_weight(facex_xyz, frame_idx):
    """
    Compute frame weight for each texel according to the paper, the up/bot face have little weight
    :param facex_xyz:
    :param frame_idx:
    :return:
    """
    new_x_idx, new_y_idx, new_z_idx = frame_axis_index(frame_idx)
    faces_xyz_abs = np.abs(facex_xyz)

    frame_weight = np.clip(4 * np.maximum(faces_xyz_abs[:,:,new_x_idx], faces_xyz_abs[:,:,new_y_idx]) - 3,0.0,1.0)

    return frame_weight


def gen_theta_phi(faces_xyz,frame_idx):
    """
    Generate a theta phi table of shape (6,res,res,2) according to the paper
    :param faces_xyz: original xyz direction for each texel
    :param frame_idx: the index of the frame, used to determine the new x,y,z axis
    :return: theta,phi,theta^2,phi^2
    """
    new_x_idx, new_y_idx, new_z_idx = frame_axis_index(frame_idx)

    #TODO: Why use abs(z) in original code?
    nx = faces_xyz[:,:,:,new_x_idx]
    ny = faces_xyz[:,:,:,new_y_idx]
    nz = faces_xyz[:,:,:,new_z_idx]
    max_xy = np.maximum(np.abs(nx),np.abs(ny))

    #normalize nx,ny, in 2/3 of the cases, one of nx and ny should be 1 without normalizing it
    nx = nx / max_xy
    ny = ny / max_xy

    # xy_flag = np.where(nx > ny, np.ones_like(nx), np.zeros_like(nx))
    #
    # #the y<x case
    # tmp1 = np.where(ny <= -0.999, nx, ny)
    # tmp2 = np.where(ny > 0.999, -nx, ny)
    #
    # theta = np.where()



    theta = np.zeros_like(nx)
    theta[(ny < nx) & (ny <= -0.999)] = nx[(ny < nx) & (ny <= -0.999)]
    theta[(ny < nx) & (ny > -0.999)] = ny[(ny < nx) & (ny > -0.999)]
    theta[(nx <= nx) & (ny >= 0.999)] = -nx[(nx <= nx) & (ny >= 0.999)]
    theta[(nx <= ny) & (ny < 0.999)] = -ny[(nx <= ny) & (ny < 0.999)]


    phi = np.zeros_like(nx)
    phi[nz <= -0.999] = -max_xy[nz <= -0.999]
    phi[nz >= 0.999] = max_xy[nz >= 0.999]
    phi[(nz > -0.999) & (nz < 0.999)] = nz[(nz > -0.999) & (nz < 0.999)]

    theta2 = theta * theta
    phi2 = phi * phi

    return theta,phi,theta2,phi2




def fetch_samples(tex_input,output_level):
    """

    :param tex_input: the all level cubemap input
    :param output_level: the output level(0-6) with 128 res for level 0 and 2 res for level 6
    :return:
    """
    interpolator = interpolation.trilinear_mipmap_interpolator(tex_input)


    n_tap = 8
    n_subtap = 4
    n_res = 128 >> output_level
    faces_xyz = texel_directions(n_res)
    X,Y,Z = gen_frame_xyz(faces_xyz)

    coefficient_dirx = coefficient.fetch_coefficient("quad",output_level,0)
    coefficient_diry = coefficient.fetch_coefficient("quad",output_level,1)
    coefficient_dirz = coefficient.fetch_coefficient("quad",output_level,2)
    coefficient_level = coefficient.fetch_coefficient("quad",output_level,3)
    coefficient_weight = coefficient.fetch_coefficient("quad",output_level,4)


    color = np.zeros((6,n_res,n_res,3))
    weight = np.zeros((6,n_res,n_res))


    # frame_idx 0 means up vector is x
    # The coefficient for this frame lies in [frame_idx * 8, (frame_idx + 1) * 8) for the last dimension
    for frame_idx in range(3):
        frame_weight = gen_frame_weight(faces_xyz, frame_idx)
        theta,phi,theta2,phi2 = gen_theta_phi(faces_xyz,frame_idx=frame_idx)

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




                #compute sample directioin
                sample_direction = sample_x + sample_y + sample_z
                #project it to cube
                max_dir = np.max(sample_direction,axis=-1)
                sample_direction /= np.stack((max_dir,max_dir,max_dir),axis=-1)
                #adjust level
                sample_level += 3/4 * np.log2( map_util.dot_vectorized_4D(sample_direction,sample_direction) )


                color_tmp = interpolator.interpolate_all(sample_direction,sample_level)

                color += color_tmp
                weight ++ sample_weight

    #devide by weight
    weight_stack = np.stack((weight,weight,weight),axis=-1)
    color = color / weight_stack

    return color






if __name__ == '__main__':
    mipmap_l0 = image_read.envmap_to_cubemap('exr_files/rosendal_plains_2_1k.exr',128)
    mipmaps = interpolation.downsample_full(mipmap_l0,7)
    fetch_samples(mipmaps,0)