import torch_util_all_locations
import torch_util
import torch
from reference import compute_ggx_distribution_reference,compute_ggx_distribution_reference_torch_vectorized
import numpy as np
import mat_util
import map_util
from optmization_torch import precompute_opt_info,test_multiple_texel_full_optimization_vectorized,test_multiple_texel_full_optimization
import specular
import logging

def check_vectorized():
    n_sample_per_level = 5
    ref_list_global = []

    ggx_alpha = 0.1

    device = torch.device('cpu')

    rng = np.random.default_rng(12345)
    all_locations = map_util.sample_location(n_sample_per_level, rng)


    params = np.random.random((5,3,24))
    params[4] = np.random.random((3,24)) - 0.1

    for i in range(n_sample_per_level):
        location = all_locations[i, :]
        ggx_ref = compute_ggx_distribution_reference(128, ggx_alpha, location)
        ggx_ref = torch.from_numpy(ggx_ref).to(device)
        ggx_ref /= torch.sum(ggx_ref)
        ref_list_global.append(ggx_ref)

    all_locations = torch.from_numpy(all_locations).to(device)

    params = torch.from_numpy(params).to(device)

    weight_per_frame_global, xyz_per_frame_global, theta_phi_per_frame_global = precompute_opt_info(all_locations,
                                                                                                    n_sample_per_level)

    tmp_pushed_back_result = test_multiple_texel_full_optimization_vectorized(8,
                                                                              n_sample_per_level, ref_list_global,
                                                                              weight_per_frame_global, xyz_per_frame_global,
                                                                              theta_phi_per_frame_global, params,
                                                                              False, True, True,
                                                                              device)


    _,loop_pushed_back_result = test_multiple_texel_full_optimization(None,8,n_sample_per_level, ref_list_global, weight_per_frame_global, xyz_per_frame_global,theta_phi_per_frame_global, params,False,True,True)

    for i in range(n_sample_per_level):
        test1 = tmp_pushed_back_result[i]
        test2 = loop_pushed_back_result[i]

        diff = test1 - test2

        print(torch.unique(diff==0))



def find_closest_alpha():
    n_sample_per_level = 200
    n_sample_per_frame = 32
    constant = False
    adjust_level = True
    allow_neg_weight = True
    ggx_ref_jac_weight = False

    model = torch_util.SimpleModel(n_sample_per_frame)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    rng = torch.Generator(device=device)
    # rng.manual_seed(12345)
    all_locations = torch_util.sample_location(n_sample_per_level, rng)

    tex_directions_res_map = map_util.texel_directions(128).astype(np.float32)
    tex_directions_res = mat_util.normalized(tex_directions_res_map)
    # stack this for N times, is this necessary?
    tex_directions_res_map = np.tile(tex_directions_res_map, (n_sample_per_level, 1, 1, 1, 1))
    tex_directions_res = np.tile(tex_directions_res, (n_sample_per_level, 1, 1, 1, 1))
    tex_directions_res_map = torch.from_numpy(tex_directions_res_map).to(device)
    tex_directions_res = torch.from_numpy(tex_directions_res).to(device)

    # test
    all_locations = torch_util.sample_location(n_sample_per_level, rng)
    # new parameter
    weight_per_frame, xyz_per_frame, theta_phi_per_frame = precompute_opt_info(all_locations,
                                                                               n_sample_per_level)

    info = specular.cubemap_level_params(18)

    logger = logging.getLogger(__name__)

    log_name = "level_alpha_ref_nojacref.log"

    logging.basicConfig(filename=log_name, filemode='a', level=logging.INFO,
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    logger.info("\n\n")





    for level in range(7):

        name = "./refs/quad32level" + str(level)
        model.load_state_dict(torch.load(name))



        params = model()
        tmp_pushed_back_result = test_multiple_texel_full_optimization_vectorized(n_sample_per_frame,
                                                                                  n_sample_per_level, None,
                                                                                  weight_per_frame, xyz_per_frame,
                                                                                  theta_phi_per_frame, params,
                                                                                  constant, adjust_level, allow_neg_weight,
                                                                                  device)
        # normalize pushed_back result
        tmp_pushed_back_sum = torch.sum(tmp_pushed_back_result, dim=[1, 2, 3, 4], keepdim=True)
        tmp_pushed_back_result /= (tmp_pushed_back_sum + 1e-7)


        level_alpha = info[level].roughness


        ref_list = compute_ggx_distribution_reference_torch_vectorized(128, level_alpha, all_locations,
                                                                       tex_directions_res,
                                                                       tex_directions_res_map, ggx_ref_jac_weight)
        ref_list = ref_list.reshape(ref_list.shape + (1,))
        ref_list = ref_list / torch.sum(ref_list, dim=(1, 2, 3, 4), keepdim=True)
        diff = torch.abs(ref_list - tmp_pushed_back_result)
        diff = torch.sum(diff, dim=[1, 2, 3, 4])
        mean_error = torch.mean(diff)


        logger.info("****************************")
        logger.info("Level {}  GGX alpha from specular{}  Error using this alpha{}".format(level,level_alpha,mean_error.item()))


        alpha_range_min = level_alpha - (0.1 * 2/7 * level)
        alpha_range_max = level_alpha + (0.1 * 2/7 * level)

        if alpha_range_min < 0.0:
            alpha_range_min = 0.001

        if alpha_range_max > 1.0:
            alpha_range_max = 1.0

        alpha_list = np.linspace(alpha_range_min, alpha_range_max, 200)

        min_error = 2.0
        best_idx = -1

        logger.info("Begin to find lowest error alpha within range [{},{}]".format(alpha_range_min, alpha_range_max))

        for i in range(len(alpha_list)):
            # new reference
            ref_list = compute_ggx_distribution_reference_torch_vectorized(128, alpha_list[i], all_locations,
                                                                           tex_directions_res,
                                                                           tex_directions_res_map, ggx_ref_jac_weight)
            ref_list = ref_list.reshape(ref_list.shape + (1,))
            ref_list = ref_list / torch.sum(ref_list, dim=(1, 2, 3, 4), keepdim=True)
            diff = torch.abs(ref_list - tmp_pushed_back_result)
            diff = torch.sum(diff, dim=[1, 2, 3, 4])
            mean_error = torch.mean(diff)

            if mean_error.item() < min_error:
                best_idx = i
                min_error = mean_error.item()

        logger.info("Best GGX alpha{}    Best Error{}]\n\n".format(alpha_list[best_idx],min_error))







if __name__ == '__main__':
    find_closest_alpha()
    #check_vectorized()