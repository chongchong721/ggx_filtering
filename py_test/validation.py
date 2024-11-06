import torch_util_all_locations
import torch_util
import torch
from reference import compute_ggx_distribution_reference
import numpy as np
import mat_util
import map_util
from optmization_torch import precompute_opt_info,test_multiple_texel_full_optimization_vectorized,test_multiple_texel_full_optimization
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


if __name__ == '__main__':
    check_vectorized()