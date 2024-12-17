import optimization_ggx_torch
import argparse
import os
import mat_util
import torch_util
import torch_util_all_locations
import logging
import map_util
import material
import numpy as np
import torch
import reference
from LBFGS import FullBatchLBFGS,LBFGS
import image_read

merl_directory = '../merl_database/'


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
    parser.add_argument("-n", "--ndir",type=int)
    parser.add_argument("-f","--nsampleframe",type=int)

    parser.add_argument("-vreflect", "--view-reflection-parameterization",type=str2bool, nargs='?',
                        const=True,
                        default=False,
                        help='Whether to generate theta phi according to reflected directions'
                        )
    parser.add_argument("-s", "--shuffle", type=str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help='Whether to use random directions for each iteration'
                        )
    parser.add_argument('-clip', '--clip-below-horizon',type=str2bool,nargs='?',
        const=True,
        default=False)
    parser.add_argument(
        '-w', '--negweight',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Allow negative weight or not'
    )
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
    parser.add_argument(
        '-j', '--jacref',
        type=str,
        default='None',
        help='apply what jacobian to reference'
    )

    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer (default: 0.01)')

    parser.add_argument('-material', '--merl-material-name', type=str,default='None')

    args = parser.parse_args()

    optimizer = args.optimizer
    ndir = args.ndir
    constant = args.constant
    adjust_level = args.adjustlevel
    optimizer = optimizer.lower()
    random_shuffle = args.shuffle
    allow_neg_weight = args.negweight
    n_sample_per_frame = args.nsampleframe
    ref_jac_weight = args.jacref
    lr = args.learning_rate
    clip = args.clip_below_horizon


    merl_material_name = args.merl_material_name

    if len(merl_material_name) < 7:
        merl_material_name = merl_material_name + '.binary'

    if merl_material_name[-7:] != '.binary':
        merl_material_name = merl_material_name + '.binary'


    if not os.path.exists(merl_directory + merl_material_name):
        raise ValueError("MERL material not found")

    return ndir,n_sample_per_frame,constant, adjust_level, allow_neg_weight, random_shuffle, optimizer ,lr, merl_material_name, ref_jac_weight, clip



def optimize_multiple_locations_merl(merl_material_name,n_sample_per_level, constant, n_sample_per_frame, adjust_level = False,
                                vectorize = True, optimizer_type = "adam", random_shuffle = False, allow_neg_weight = False,
                                ref_jac_weight = 'None', learning_rate = 1e-4, ndf_clipping = False):
    """
    To speed up, a lot of things can be precomputed, including the relative XYZ,the frame weight_g
    we don't have to compute this in every iteration
    :param n_sample_per_level:
    :return:
    """
    merl_ndf = material.powit_merl_ndf(merl_directory + merl_material_name)

    merl_material_name_noext = merl_material_name.split('.')[0]


    visualize_loss = True

    device = optimization_ggx_torch.get_device()


    view_model_dict = torch_util.create_view_model_dict()

    logger = logging.getLogger(__name__)

    log_name = map_util.log_merl_filename(merl_name=merl_material_name_noext, n_sample_per_frame=n_sample_per_frame,
                                          n_sample_per_level=n_sample_per_level, constant=constant, adjust_level=adjust_level,
                                          optim_method=optimizer_type, random_shuffle=random_shuffle, allow_neg_weight=allow_neg_weight,
                                          ref_jac_weight=ref_jac_weight, ndf_clipping=ndf_clipping)

    log_name = './logs_merl/' + log_name

    logging.basicConfig(filename=log_name, filemode='a', level=logging.INFO,
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    logger.info("\n\n")

    if not random_shuffle:
        raise NotImplementedError
    else:
        rng = torch.Generator(device=device)
        rng.seed()
        #rng.manual_seed(12345)

        all_locations_cube,all_locations = torch_util.sample_location(n_sample_per_level,rng)


    model_name = map_util.model_merl_filename(merl_name=merl_material_name_noext, n_sample_per_frame=n_sample_per_frame,
                                          n_sample_per_level=n_sample_per_level, constant=constant, adjust_level=adjust_level,
                                          optim_method=optimizer_type, random_shuffle=random_shuffle, allow_neg_weight=allow_neg_weight,
                                          ref_jac_weight=ref_jac_weight, ndf_clipping=ndf_clipping)



    if not constant:
        model = torch_util.QuadModel(n_sample_per_frame)
    else:
        model = torch_util.ConstantModel(n_sample_per_frame)


    if os.path.exists("./model_merl/" + model_name):
        logger.info("Read model from dict")
        model.load_state_dict(torch.load("./model_merl/" + model_name,map_location=device))


    model.to(device)

    tex_directions_res_map = map_util.texel_directions(128).astype(np.float32)
    tex_directions_res = mat_util.normalized(tex_directions_res_map)
    # stack this for N times, is this necessary?
    tex_directions_res_map = np.tile(tex_directions_res_map, (n_sample_per_level, 1, 1, 1, 1))
    tex_directions_res = np.tile(tex_directions_res, (n_sample_per_level, 1, 1, 1, 1))
    tex_directions_res_map = torch.from_numpy(tex_directions_res_map).to(device)
    tex_directions_res = torch.from_numpy(tex_directions_res).to(device)


    mipmaps = reference.get_synthetic_mipmap(np.array([0, 0, 1]), 128)





    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    elif optimizer_type == "bfgs":
        #optimizer = optim.LBFGS(model.parameters(), lr = 0.7, line_search_fn="strong_wolfe", max_iter = 50, tolerance_grad = 1e-8, tolerance_change = 1e-10)
        optimizer = FullBatchLBFGS(model.parameters(), lr = learning_rate, history_size=120,line_search="Wolfe")
    else:
        raise ValueError("Unknown optimizer type")
    n_epoch = 1000000



    def closure():
        optimizer.zero_grad()
        params = model()

        # start_time = time.time()

        if random_shuffle:
            all_locations_cube,all_locations =  torch_util.sample_location(n_sample_per_level,rng)

            all_precomputed_info = optimization_ggx_torch.precompute_opt_info(all_locations_cube,n_sample_per_level)
            ref_list = reference.compute_merl_ndf_reference_half_vector_torch_vectorized(
                128,merl_ndf,all_locations,tex_directions_res,tex_directions_res_map,ref_jac_weight
            )

            ref_list = ref_list / torch.sum(ref_list,dim=(1,2,3,4),keepdim=True)


            # ref_list_test = compute_ggx_distribution_reference_torch_vectorized(128, ggx_alpha, all_locations,
            #                                                                tex_directions_res, tex_directions_res_map,
            #                                                                False)
            #
            # ref_list_test = ref_list_test.reshape(ref_list_test.shape + (1,))
            # ref_list_test = ref_list_test / torch.sum(ref_list_test,dim=(1,2,3,4),keepdim=True)
            #
            # ref_list_diff = ref_list - ref_list_test
            #
            # diff = torch.abs(ref_list_diff)
            # diff = torch.sum(diff,dim=[1,2,3,4])
            # mean_error = torch.mean(diff)
        else:
            raise NotImplementedError
        if vectorize:

            tmp_pushed_back_result = optimization_ggx_torch.multiple_texel_full_optimization_vectorized_vec_prepare(n_sample_per_frame,
                                                                                 n_sample_per_level, ref_list,
                                                                                 all_precomputed_info, params,
                                                                                 constant, adjust_level, allow_neg_weight, device)


            # tmp_pushed_back_result = multiple_texel_full_optimization_vectorized(n_sample_per_frame,
            #                                                                      n_sample_per_level, ref_list,
            #                                                                      weight_per_frame, xyz_per_frame,
            #                                                                      theta_phi_per_frame, params,
            #                                                                      constant, adjust_level, allow_neg_weight, device)


            tmp_pushed_back_sum = torch.sum(tmp_pushed_back_result, dim=[1, 2, 3, 4], keepdim=True)
            tmp_pushed_back_result /= (tmp_pushed_back_sum + 1e-7)
            diff = torch.abs(ref_list - tmp_pushed_back_result)
            diff = torch.sum(diff, dim=[1, 2, 3, 4])
            mean_error = torch.mean(diff)
        else:
            # if view_dependent:
            #     raise NotImplementedError
            # error_list, result_list = multiple_texel_full_optimization(all_locations, n_sample_per_frame,
            #                                                            n_sample_per_level, ref_list,
            #                                                            weight_per_frame, xyz_per_frame,
            #                                                            theta_phi_per_frame, coef_table=params,
            #                                                            constant=constant,
            #                                                            adjust_level=adjust_level,
            #                                                            device=device)
            #
            # tmp = torch.stack(error_list)
            # mean_error = tmp.mean()
            raise NotImplementedError

        #If we are using the LBFGS from github. Thenwe should not call backward
        #mean_error.backward()
        return mean_error


    if optimizer_type == "adam":
        for i in range(n_epoch):
            loss = closure()
            loss.backward()
            if not torch.isnan(loss):
                optimizer.step()
            else:
                logger.info("NaN loss detected")
                logger.info("Current parameters are", model().detach().numpy())
                # pushed_back_result = multiple_texel_full_optimization_vectorized(n_sample_per_frame,
                #                                                                  n_sample_per_level, ref_list_global,
                #                                                                  weight_per_frame_global, xyz_per_frame_global,
                #                                                                  theta_phi_per_frame_global, model(),
                #                                                                  constant, adjust_level, allow_neg_weight, device)
                # # normalize pushed_back result
                # pushed_back_sum = torch.sum(pushed_back_result, dim=[1, 2, 3, 4], keepdim=True)
                # logger.info("is pushed_back_result NaN? ", torch.isnan(pushed_back_result).any())
                # logger.info("is pushed_back_sum 0?", pushed_back_sum == 0.0)


            logger.info("[it{}]:loss is{}".format(i, loss.item()))

            if i % 500 == 0:
                # normalize result
                # result /= torch.sum(result)
                # visualization.visualize_optim_result(ggx_ref, result)
                logger.info(f"saving model")
                optimization_ggx_torch.save_model(model, "./model_merl/" + model_name)
                # logger.info(f"[it{i}]Loss: {mean_error.item()}")
                # if not view_dependent:
                #     synthetic_filter_showcase(model().cpu().detach().numpy(), constant, adjust_level, ggx_alpha,
                #                               n_sample_per_frame, 2, n_sample_per_level, optimizer_type, random_shuffle,
                #                               allow_neg_weight, ggx_ref_jac_weight, mipmaps, view_dependent,view_option_str,"it" + str(i))



    else:
        for i in range(n_epoch):
            options = {
                "closure" : closure,
            }

            obj,grad,lr,backtracks,clos_evals,grad_evals,desc_dir,fail = optimizer.step(options=options)
            logger.info("[it{}]:loss is{}".format(i, obj.item()))
            if(torch.isnan(obj)):
                logger.info("NaN loss detected in LBFGS, Save last param and terminate!")
                optimization_ggx_torch.save_model(model, "./model_merl/" + model_name + "_nan")
            if i % 50 == 0:
                logger.info(f"saving model")
                optimization_ggx_torch.save_model(model, "./model_merl/" + model_name)
                # if not view_dependent:
                #     synthetic_filter_showcase(model().cpu().detach().numpy(), constant, adjust_level, ggx_alpha,
                #                           n_sample_per_frame, 2, n_sample_per_level, optimizer_type, random_shuffle,
                #                           allow_neg_weight, ggx_ref_jac_weight ,mipmaps, view_dependent,view_option_str,"it" + str(i))


    logger.info(f"MAX n_iter {n_epoch} reached")



def visualize_merl_ref_ndf(merl_material_name):
    pass





if __name__ == '__main__':
    ndir_g, n_sample_per_frame_g, constant_g, adjust_level_g, allow_neg_weight_g, random_shuffle_g, optimizer_g, lr_g, merl_material_name_g, ref_jac_weight_g, clip_g = process_cmd()

    print("Computing MERL material: {}.\n"
          "Adjust Level with Jacobian:{}\n"
          "Using constant params:{}\n"
          "Optimizer:{} - LR{}\n"
          "random shuffle:{}\n"
          "Allow negative weight:{},\n"
          "Use Jacobian weighted ndf as reference:{}\n"
          "Clipping below horizong ndf:{}\n".format(merl_material_name_g,adjust_level_g,constant_g,optimizer_g,lr_g,
                                                    random_shuffle_g,allow_neg_weight_g,ref_jac_weight_g,clip_g))


    optimize_multiple_locations_merl(
        merl_material_name=merl_material_name_g,n_sample_per_level=ndir_g, constant=constant_g,n_sample_per_frame=n_sample_per_frame_g,
        adjust_level=adjust_level_g, vectorize=True, optimizer_type=optimizer_g,random_shuffle=random_shuffle_g,
        allow_neg_weight=allow_neg_weight_g,ref_jac_weight=ref_jac_weight_g,learning_rate=lr_g,ndf_clipping=clip_g
    )