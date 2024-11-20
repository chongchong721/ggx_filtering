import numpy as np
import material
import image_read
import specular
import map_util
import mat_util

import multiprocessing

from tqdm import tqdm
import numba

import interpolation


import torch
import torch_util
from torch_util import texel_dir_128_torch


#Compute the reference \int l(h)d(h)dh by numerically integrate every texel we have





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


def loop_compute_view_dependent(normal_direction,normalized_xyz_input,j_weighted_input,alpha,delta_s, view_direction):
    pass



def loop_compute_nojit(normal_direction,normalized_xyz_input,j_weighted_input,alpha,delta_s):
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

def compute_reference_view_dependent(input_map,cubemap_res,ref_res,GGX_alpha,view_direction,save_file_name=None):
    #Now loop through every texel
    int_result = np.zeros((6,ref_res,ref_res,input_map.shape[-1]))
    # generate xyz for each face
    xyz_output = map_util.texel_directions(ref_res)
    xyz_input = map_util.texel_directions(cubemap_res)
    # integral area
    delta_s = 4.0 / cubemap_res / cubemap_res

    jacobian = map_util.jacobian_vertorized(xyz_input)

    j_weighted_input = np.stack((jacobian, jacobian, jacobian), axis=-1) * input_map

    normalized_xyz_input = mat_util.normalized(xyz_input)
    normalized_xyz_output = mat_util.normalized(xyz_output)

    total_it = 6 * ref_res * ref_res

    normalized_xyz_input_tmp = np.reshape(normalized_xyz_input, (-1, 3))
    j_weighted_input_tmp = np.reshape(j_weighted_input, (-1, 3))

    with tqdm(total=total_it, desc="texel progress") as pbar:
        # loop through every texel
        for face_idx in range(6):
            for i in range(ref_res):
                for j in range(ref_res):
                    # l = v = h
                    normal_direction = normalized_xyz_output[face_idx, i, j]
                    # Now, compute ndf for each direction, using input

                    # The computation of cosine requires all normalized vectors
                    # integral = loop_compute(normal_direction,normalized_xyz_input_tmp,j_weighted_input_tmp,GGX_alpha,delta_s)
                    integral = loop_compute_nojit(normal_direction, normalized_xyz_input_tmp, j_weighted_input_tmp,
                                                  GGX_alpha,
                                                  delta_s)
                    # cosine = np.dot(normalized_xyz_input,normal_direction)
                    # ndf = GGX.ndf_isotropic(cosine)
                    #
                    # integral_sum = np.stack((ndf,ndf,ndf),axis=-1) * j_weighted_input
                    #
                    # integral = np.sum(integral_sum, axis=(0,1,2)) * delta_s
                    int_result[face_idx, i, j] = integral
                    pbar.update(1)

    # file_name = "res" + str(ref_res) + "alpha" + str(GGX_alpha) + ".npy"
    if save_file_name is not None:
        np.save("./refs/" + save_file_name, int_result)

    return int_result

def compute_reference(input_map, cubemap_res, ref_res, GGX_alpha, save_file_name = None):
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
    xyz_output = map_util.texel_directions(ref_res)
    xyz_input = map_util.texel_directions(cubemap_res)
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
                    #integral = loop_compute(normal_direction,normalized_xyz_input_tmp,j_weighted_input_tmp,GGX_alpha,delta_s)
                    integral = loop_compute_nojit(normal_direction, normalized_xyz_input_tmp, j_weighted_input_tmp, GGX_alpha,
                                            delta_s)
                    # cosine = np.dot(normalized_xyz_input,normal_direction)
                    # ndf = GGX.ndf_isotropic(cosine)
                    #
                    # integral_sum = np.stack((ndf,ndf,ndf),axis=-1) * j_weighted_input
                    #
                    # integral = np.sum(integral_sum, axis=(0,1,2)) * delta_s
                    int_result[face_idx,i,j] = integral
                    pbar.update(1)

    #file_name = "res" + str(ref_res) + "alpha" + str(GGX_alpha) + ".npy"
    if save_file_name is not None:
        np.save("./refs/" + save_file_name,int_result)

    return int_result


def ndf_isotropic_torch_vectorized(alpha, cos_theta):
    """
    Only cosine(theta) is needed in isotropic case
    :param cosine theta: computed directly from vector dot product
    :return:
    """
    a_pow_of2 = alpha ** 2
    ndf = a_pow_of2 / (np.pi * torch.pow(cos_theta * cos_theta * (a_pow_of2-1) + 1,2))
    ndf = torch.where(cos_theta > 0.0, ndf, 0.0)
    return ndf


def importance_sample_view_dependent_location(ggx_alpha, n_sample_per_frame, n_sample_per_level, normal_directions, view_directions):
    """
    Importance sampling, for each sample, sample n_sample_per_frame * 3 importance samples based on GGX VNDF to initialize optimization pipeline
    :param ggx_alpha:
    :param n_sample_per_frame:
    :param n_sample_per_level:
    :param normal_directions:
    :param view_directions:
    :return:
    """
    rng = np.random.default_rng()
    ggx_sampler = material.VNDFSamplerGGX(ggx_alpha,ggx_alpha)

    local_view = material.rotate_vectors_to_up_batch(normal_directions,view_directions)

    half_vec_list = []
    for i in range(n_sample_per_level):
        view = local_view[i]
        half_vec_list_cur_view = []
        for j in range(n_sample_per_frame*3):
            h = ggx_sampler.sample_anisotropic_ggx_new(view)

            #compute

            half_vec_list_cur_view.append(h)

        half_vec_list.append(half_vec_list_cur_view)



def g1_isotropic_torch_vectorized(ggx_alpha,view, normal_directions, half_vectors):
    """

    :param view: [N,3]  should be normalized
    :param normal_directions: [N,3] should be normalized
    :param half_vectors: [N,6,128,128,3] should be normalized
    :return:
    """
    #heaviside
    #dot product of view and all normal_directions
    view_reshaped = view.view(view.shape[0],1,1,1,3)
    element_wise_sum = view_reshaped * half_vectors
    cosine = torch.sum(element_wise_sum,dim=-1)
    heaviside_result = torch.where(cosine > 0.0, 1.0, 0.0)

    #factor is in shape of [N]. Given one normal and one view, there is only one lambda
    factor = 1 / (1 + lambda_isotropic_torch_vectorized(ggx_alpha,view, normal_directions))

    return heaviside_result * factor.view(view.shape[0],1,1,1)


def lambda_isotropic_torch_vectorized(ggx_alpha, directions, normal_directions):
    """
    Given N view directions, compute Lambda of this view with different wh

    Lambda(w) -> compute theta_o
    :param ggx_alpha:
    :param directions: [N,3] should be normalized
    :param normal_directions: [N,3]
    :return:
    """
    cosine = torch.sum(directions * normal_directions,dim=-1)
    theta = torch.arccos(cosine)
    tangent = torch.tan(theta)

    alpha = 1 / ggx_alpha / tangent

    l = (-1 + torch.sqrt(1 + 1 / (alpha**2))) / 2
    return l


def vndf_isotropic_torch_vectorized(view_directions, normal_directions, directions, directions_map ,ggx_alpha):
    """
    :param view_directions: [N,3]
    :param normal_directions: [N,3]
    :param directions [N,6,128,128,3] -> l
    :return:
    """
    #[N,6,128,128,3]
    half_vectors = view_directions.view(view_directions.shape[0], 1, 1, 1, 3) + directions
    half_vectors = half_vectors / torch.linalg.norm(half_vectors, dim = -1, keepdim = True)

    g1 = g1_isotropic_torch_vectorized(ggx_alpha,view_directions,normal_directions,half_vectors)
    ndf = compute_ggx_ndf_ref_view_dependent_torch_vectorized(ggx_alpha,normal_directions,directions,directions_map,view_directions,False)
    cos = torch.abs(torch.sum(view_directions.view(view_directions.shape[0], 1, 1, 1, 3) * half_vectors,dim=-1))
    result = g1 * ndf * cos / torch.sum(view_directions * normal_directions,dim=-1).view(view_directions.shape[0], 1, 1, 1)

    zero_condition = torch.sum(normal_directions.view(view_directions.shape[0], 1, 1, 1, 3) * half_vectors, dim = -1) < 0.0

    result[zero_condition] = 0.0

    # #Test the first and second item vndf
    # for i in range(2):
    #     view_test = view_directions[i]
    #     normal_test = normal_directions[i]
    #     for j in range(6):
    #         for m in range(128):
    #             for n in range(128):
    #                 direction_l = directions[j,m,n]
    #
    #                 this_g1 = g1[i,j,m,n]
    #                 this_ndf = ndf[i,j,m,n]
    #                 this_cos = cos[i,j,m,n]
    #
    #
    #                 this_half_vector = half_vectors[i,j,m,n].detach().numpy()
    #
    #
    #                 view_test_np = view_test.detach().numpy()
    #                 normal_test_np = normal_test.detach().numpy()
    #                 direction_l_np = direction_l.detach().numpy()
    #
    #                 view_rotate_np = material.rotate_vector_to_up(normal_test_np,view_test_np)
    #                 l_rotate_np = material.rotate_vector_to_up(normal_test_np,direction_l_np)
    #                 this_half_rotate = material.rotate_vector_to_up(normal_test_np,this_half_vector)
    #
    #                 h = mat_util.get_half_vector(view_rotate_np.reshape((3,1)),l_rotate_np.reshape((3,1)))
    #
    #                 vndf_current = ggx.vndf(view_rotate_np.reshape((3,1)),h)
    #                 print(vndf_current)
    #                 vndf_test = result[i,j,m,n]
    #                 print(vndf_test.item())

    return result



def compute_ggx_vndf_ref_view_dependent_torch_vectorized(ggx_alpha, normal_directions, directions, directions_map, view_direction, apply_jacobian = False):
    """
    Compute VNDF(visible NDF)
    :param ggx_alpha:
    :param normal_directions:
    :param directions:
    :param directions_map:
    :param view_direction:
    :param apply_jacobian:
    :return:
    """
    vndf = vndf_isotropic_torch_vectorized(view_direction,normal_directions,directions,directions_map,ggx_alpha)
    if apply_jacobian:
        #which directions to use, half_vec or map direction? This will make huge difference when viewing angle close to grazing angle?
        j = torch_util.torch_jacobian_vertorized(directions_map)
        vndf = vndf * j

    vndf = torch_util.clip_below_horizon_part_view_dependent(normal_directions, vndf, texel_dir_128_torch)

    return vndf



def compute_ggx_ndf_ref_view_dependent_torch_vectorized(ggx_alpha, normal_directions, directions, directions_map, view_direction, apply_jacobian = False):
    """
    :param ggx_alpha:
    :param normal_directions: in shape of [N,3]
    :param directions: texel directions(pregen) in shape of [n_sample_per_level,6,res,res,3], this directions should be normalized
    :param directions_map: un-normalized directions(on the cubemap)
    :param view_direction: the view direction, should be normalized
    :param normal_directions: normalized normal directions
    :return:

    In this case normal_directions are n directions are l, view_direction is v

    When computing view dependent reference, directions can be thought as the light direction. Given n, we can compute
    the h for each texel, and we can get the NDF of h for each direction

    We should manually set the NDF that is below the horizon to be zero, should we?
    If we do this, after push back, we also need to clip the below horizon part.

    """
    if ggx_alpha > 1.0:
        raise NotImplementedError
    #normal_direction_normalized = normal_directions / torch.linalg.norm(normal_directions, dim=-1, keepdim=True)
    half_vec = torch_util.get_all_half_vector_torch_vectorized(view_direction,directions)
    cosine = torch.einsum('bl,bijkl->bijk', normal_directions, half_vec)
    ndf = ndf_isotropic_torch_vectorized(ggx_alpha,cosine)

    if apply_jacobian:
        #which directions to use, half_vec or map direction? This will make huge difference when viewing angle close to grazing angle?
        j = torch_util.torch_jacobian_vertorized(directions_map)
        ndf = ndf * j

    ndf = ndf.reshape(ndf.shape + (1,))

    ndf = torch_util.clip_below_horizon_part_view_dependent(normal_directions, ndf, directions)

    return ndf




def compute_ggx_ndf_reference_half_vector_torch_vectorized(res, ggx_alpha, normal_directions, directions, directions_map, apply_jacobian=False):
    """
    The half vector is computed as (normal + directions). Not
    :param res:
    :param ggx_alpha:
    :param normal_directions: this directions is normalized
    :param directions:
    :param directions_map:
    :param apply_jacobian:
    :return:
    """
    #normal_direction_normalized = normal_directions / torch.linalg.norm(normal_directions,dim = -1, keepdim = True)

    #We use normal directions to compute the have vector only because we are under the assumption n=v  (l + v)

    half_vec = torch_util.get_all_half_vector_torch_vectorized(normal_directions,directions)
    cosine = torch.einsum('bl,bijkl->bijk', normal_directions, half_vec)
    ndf = ndf_isotropic_torch_vectorized(ggx_alpha,cosine)

    if apply_jacobian:
        j = torch_util.torch_jacobian_vertorized(directions_map)
        ndf = ndf * j

    ndf = ndf.reshape(ndf.shape + (1,))

    return ndf




def compute_ggx_ndf_reference_torch_vectorized(res, ggx_alpha, normal_directions, directions, directions_map, apply_jacobian=False):
    """
    :param res:
    :param ggx_alpha:
    :param normal_directions: in shape of [N,3]
    :param directions: texel directions(pregen) in shape of [n_sample_per_level,6,res,res,3]?
    :param directions_map: un-normalized directions(on the cubemap)
    :return:
    """
    normal_direction_normalized = normal_directions / torch.linalg.norm(normal_directions,dim = -1, keepdim = True)
    cosine = torch.einsum('bl,bijkl->bijk', normal_direction_normalized, directions)
    ndf = ndf_isotropic_torch_vectorized(ggx_alpha,cosine)

    if apply_jacobian:
        j = torch_util.torch_jacobian_vertorized(directions_map)
        ndf = ndf * j


    return ndf



def compute_ggx_ndf_reference(res, ggx_alpha, normal_direction, apply_jacobian=False):
    """
    Compute the reference GGX in cube map form. normal_direction is the assumed GGX normal(where there is highest pdf)
    :param res:
    :param ggx_alpha:
    :param normal_direction:
    :return:
    """
    ggx = material.GGX(ggx_alpha,ggx_alpha)
    assert normal_direction.size == 3
    normal_direction_normalized = normal_direction / np.linalg.norm(normal_direction)
    directions = map_util.texel_directions(res)
    normalized_directions = mat_util.normalized(directions,axis=-1)

    cosine = np.dot(normalized_directions,normal_direction_normalized.flatten())
    ndf = ggx.ndf_isotropic(cosine)

    if apply_jacobian:
        j = map_util.jacobian_vertorized(directions)
        ndf = ndf * j


    return ndf.reshape(ndf.shape+(1,))



def synthetic_onepoint_input(direction,res,value=100.0):
    array = np.zeros((6,res,res,3))
    uv,face = map_util.xyz_to_uv(direction)
    u,v = uv
    u_idx = int(u / (1 / res))
    v_idx = int(v / (1 / res))

    v_idx_top = v_idx + 1
    u_idx_right = u_idx + 1



    u_loc_right = u_idx_right * (1 / res)
    v_loc_top = v_idx_top * (1 / res)
    u_loc_left = u_idx * (1 / res)
    v_loc_bot = v_idx * (1 / res)

    distance = u_loc_right - u_loc_left

    v_arr_idx_bot = (res - 1) - v_idx
    v_arr_idx_top = (res - 1) - v_idx_top

    p_top_left = value * (u_loc_right - u) / distance * (v - v_loc_bot) / distance
    p_top_right = value * (u - u_loc_left) / distance * (v - v_loc_bot) / distance
    p_bot_left = value * (u_loc_right - u) / distance * (v_loc_top - v) / distance
    p_bot_right = value * (u - u_loc_left)  / distance * (v_loc_top - v) / distance

    array[face,v_arr_idx_top , u_idx] = p_top_left
    array[face,v_arr_idx_top , u_idx_right] = p_top_right
    array[face,v_arr_idx_bot , u_idx] = p_bot_left
    array[face,v_arr_idx_bot , u_idx_right] = p_bot_right

    return array

def get_synthetic_mipmap(direction,res):
    mipmap_l0 = synthetic_onepoint_input(direction, res)
    mipmaps = interpolation.downsample_full(mipmap_l0,7)
    return mipmaps




def generate_custom_reference(file_name, high_res, ggx_alpha, ref_res):
    env_map = image_read.envmap_to_cubemap('./exr_files/'+file_name,high_res)
    save_name = "custom_" + file_name[6:-4] + "_" + str(ref_res) + "_" + "{:.3f}".format(ggx_alpha)
    save_name_np = save_name + ".npy"
    save_name_hdr = save_name + ".hdr"
    array_ref = compute_reference(env_map, high_res, ref_res,ggx_alpha, save_name_np)
    image_read.gen_cubemap_preview_image(array_ref,ref_res,filename = "./refs/" + save_name_hdr)


if __name__ == '__main__':
    u=0.8
    v=0.2
    direction = map_util.uv_to_xyz((u,v),4)
    #synthetic_onepoint_input(direction,128)





    info = specular.cubemap_level_params(18)

    a = np.array([[0.0,0.0,1.0]]).T
    normal_vector = np.array([[1.0, 0.2, 0.1]])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    normal_vector = normal_vector.T
    view_vector_test = mat_util.reflect(-a, normal_vector)

    # test = compute_ggx_vndf_ref_view_dependent_torch_vectorized(0.1, torch.from_numpy(b).reshape((1,3)), torch_util.texel_dir_128_torch,torch_util.texel_dir_128_torch_map,
    #                                                            torch.from_numpy(c).reshape((1,3)),True)

    X,Y,Z = torch_util.torch_gen_anisotropic_frame_xyz(torch.from_numpy(normal_vector).reshape((1, 3)), torch.from_numpy(view_vector_test).reshape((1, 3)))

    X_proj = torch_util.project_vector_to_surface(vector=X,surface_normal=torch.Tensor([[0.0,0.0,1.0]]))

    u,v,face_idx = torch_util.torch_xyz_to_uv_vectorized(torch.from_numpy(a).reshape((1,3)))

    if X_proj[0][0] != 0.0:
        k = X_proj[0][1] / X_proj[0][0]
        c = v - k * u
    else:
        k = torch.inf
        raise NotImplementedError

    u_at_v_at_0 = (v-c) / k
    v_at_u_at_0 = c

    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, 100)
    y = k * x + c
    plt.figure(figsize=(6, 6))  # Optional: set the figure size

    plt.grid(True)
    plt.legend()

    # Set equal aspect ratio
    plt.axis('equal')  # Ensures that the scale is the same on both axes
    plt.plot(x, y, label=f'y = {k}x + {c}')  # Plot the line
    plt.show()


    test = compute_ggx_ndf_ref_view_dependent_torch_vectorized(0.1, torch.from_numpy(normal_vector).reshape((1, 3)), torch_util.texel_dir_128_torch, torch_util.texel_dir_128_torch_map,
                                                               torch.from_numpy(view_vector_test).reshape((1, 3)), True)




    test = test[0]

    test = torch.stack((test,test,test),dim = -1)

    test = test.detach().numpy()

    image_read.gen_cubemap_preview_image(test,128,filename="./test_clip.exr")


    generate_custom_reference("08-21_Swiss_A.hdr",128,info[4].roughness,8)

    #test(128)

    file_list = ["08-21_Swiss_A.hdr","08-08_Sunset_D.hdr","04-29_Night_B.hdr","04-07_Fila_lnter.hdr"]

    #post_fix = ""
    post_fix = "_16"
    #post_fix = "_16_025"  #pow(,.25f)


    for file_name in file_list:
        print("For HDR map ", file_name)

        mipmap_l0 = image_read.envmap_to_cubemap('exr_files/' + file_name, 128)
        level = specular.cubemap_level_params(18,True)

        for n_lev in range(len(level)):
            print("generating level ", n_lev," result")
            save_file_name = file_name[6:-4] + "_" + str(level[n_lev].res) + "_" + str(level[n_lev].roughness)[:6]+ post_fix + ".npy"
            arr = np.load("./refs/" + save_file_name)
            image_read.gen_cubemap_preview_image(arr, level[n_lev].res,filename="./refs/" + save_file_name[:-4] + ".hdr")
            #compute_reference(mipmap_l0,128,level[n_lev].res,level[n_lev].roughness, save_file_name)
