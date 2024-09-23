import numpy as np
import material
from filter import texel_directions
import image_read
import specular
import map_util
import mat_util

import multiprocessing

from tqdm import tqdm
import numba

"""
Compute the reference \int l(h)d(h)dh by numerically integrate every texel we have
"""




def test(res):
    """
    Integrate over one face, the area should be 4pi / 6 - > 2*pi/3
    :return:
    """
    z = 1

    u = np.arange(0,res)
    u = u + 0.5
    u = u / res
    u = u * 2 - 1
    xv,yv = np.meshgrid(u,u,indexing='ij')



    zv = np.ones_like(xv)

    xyz = np.stack([xv,yv,zv],axis=-1)

    jacobian = map_util.jacobian_vertorized(xyz)

    #each area was 2 / 128 in area size

    result = np.sum(jacobian) / 64.0 / 64.0

    print("Done")

@numba.jit(nopython=True)
def ndf_isotropic(a,cos_theta):
    a_pow_of2 = a ** 2
    ndf = a_pow_of2 / (np.pi * np.pow(cos_theta * cos_theta * (a_pow_of2 - 1) + 1, 2))
    ndf = np.where(cos_theta > 0.0, ndf, 0.0)
    return ndf

@numba.jit(nopython=True)
def loop_compute(normal_direction,normalized_xyz_input,j_weighted_input,alpha,delta_s):
    cosine = np.dot(normalized_xyz_input, normal_direction)
    ndf = ndf_isotropic(alpha,cosine)
    integral_sum = np.stack((ndf, ndf, ndf), axis=-1) * j_weighted_input
    integral = np.sum(integral_sum, axis=0) * delta_s
    return integral


def outer_loop(normalized_xyz_output,normalized_xyz_input,j_weighted_input,alpha,delta_s,result,ref_res):
    for face_idx in range(6):
        for i in range(ref_res):
            for j in range(ref_res):
                # l = v = h
                normal_direction = normalized_xyz_output[face_idx, i, j]
                # Now, compute ndf for each direction, using input

                # The computation of cosine requires all normalized vectors
                integral = loop_compute(normal_direction, normalized_xyz_input, j_weighted_input, alpha,
                                        delta_s)
                # cosine = np.dot(normalized_xyz_input,normal_direction)
                # ndf = GGX.ndf_isotropic(cosine)
                #
                # integral_sum = np.stack((ndf,ndf,ndf),axis=-1) * j_weighted_input
                #
                # integral = np.sum(integral_sum, axis=(0,1,2)) * delta_s
                result[face_idx, i, j] = integral

    return result



def compute_reference(input_map, cubemap_res, ref_res, GGX_alpha, save_file_name):
    """
    lower res means rougher GGX
    :param input_map:
    :param cubemap_res: the resolution of environment map image(input)
    :param ref_res: the resolution of reference image(output)
    :param GGX_alpha:
    :return:
    """

    #Now loop through every texel
    int_result = np.zeros((6,ref_res,ref_res,input_map.shape[-1]))

    # generate xyz for each face
    xyz_output = texel_directions(ref_res)
    xyz_input = texel_directions(cubemap_res)
    #integral area
    delta_s = 4.0 / cubemap_res / cubemap_res

    jacobian = map_util.jacobian_vertorized(xyz_input)

    j_weighted_input = np.stack((jacobian,jacobian,jacobian),axis=-1) * input_map

    normalized_xyz_input = mat_util.normalized(xyz_input)
    normalized_xyz_output = mat_util.normalized(xyz_output)

    total_it = 6 * ref_res * ref_res

    normalized_xyz_input_tmp = np.reshape(normalized_xyz_input,(-1,3))
    j_weighted_input_tmp = np.reshape(j_weighted_input,(-1,3))

    with tqdm(total=total_it, desc="texel progress") as pbar:
        #loop through every texel
        for face_idx in range(6):
            for i in range(ref_res):
                for j in range(ref_res):
                    # l = v = h
                    normal_direction = normalized_xyz_output[face_idx,i,j]
                    # Now, compute ndf for each direction, using input

                    # The computation of cosine requires all normalized vectors
                    integral = loop_compute(normal_direction,normalized_xyz_input_tmp,j_weighted_input_tmp,GGX_alpha,delta_s)
                    # cosine = np.dot(normalized_xyz_input,normal_direction)
                    # ndf = GGX.ndf_isotropic(cosine)
                    #
                    # integral_sum = np.stack((ndf,ndf,ndf),axis=-1) * j_weighted_input
                    #
                    # integral = np.sum(integral_sum, axis=(0,1,2)) * delta_s
                    int_result[face_idx,i,j] = integral
                    pbar.update(1)

    #file_name = "res" + str(ref_res) + "alpha" + str(GGX_alpha) + ".npy"
    np.save("./refs/" + save_file_name,int_result)





if __name__ == '__main__':
    #test(128)

    file_list = ["08-21_Swiss_A.hdr","08-08_Sunset_D.hdr","04-29_Night_B.hdr","04-07_Fila_lnter.hdr"]

    for file_name in file_list:
        print("For HDR map ", file_name)

        mipmap_l0 = image_read.envmap_to_cubemap('exr_files/' + file_name, 128)
        level = specular.cubemap_level_params()

        for n_lev in range(len(level)):
            print("generating level ", n_lev," result")
            save_file_name = file_name[6:-4] + "_" + str(level[n_lev].res) + "_" + str(level[n_lev].roughness)[:6] + ".npy"
            arr = np.load("./refs/" + save_file_name)
            image_read.gen_cubemap_preview_image(arr, level[n_lev].res,filename="./refs/" + save_file_name[:-4] + ".hdr")
            #compute_reference(mipmap_l0,128,level[n_lev].res,level[n_lev].roughness, save_file_name)
