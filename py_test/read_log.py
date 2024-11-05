from optmization_torch import process_cmd
from map_util import log_filename

import numpy as np
import matplotlib.pyplot as plt




def read_log(path):
    loss_list = []
    with open(path, 'r') as f:
        for line in f:
            idx = line.find("is")
            if idx != -1:
                loss = float(line[idx+2:])
                loss_list.append(loss)

    return loss_list


def plot_graph(filename,loss_list):
    plt.figure()
    plt.plot(loss_list,label='loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.title(filename)
    plt.show()


if __name__ == '__main__':
    n_sample_per_frame,n_sample_per_level, ggx_info, flag_constant, flag_adjust_level, optimizer_string, random_shuffle, allow_neg_weight = process_cmd()

    import specular
    info = specular.cubemap_level_params(18)
    if isinstance(ggx_info, int):

        ggx_alpha = info[ggx_info].roughness
    else:
        ggx_alpha = ggx_info

    filename = log_filename(ggx_alpha,n_sample_per_frame,n_sample_per_level,flag_constant,flag_adjust_level,optimizer_string, random_shuffle, allow_neg_weight)

    n_sample_list = [1000,1000,1000,200,200,200]
    constant_list = [False,False,False,True,False,False]
    opt_list = ["adam","bfgs","bfgs","bfgs","adam","bfgs"]
    random_dir_list = [False,False,True,True,True,True]



    for i in range(6):
        plt.figure()

        n_sample_per_level, ggx_alpha, flag_constant, flag_adjust_level, optimizer_string, random_shuffle, allow_neg_weight = n_sample_list[i], 0.1, constant_list[i], True, opt_list[i], random_dir_list[i], True
        filename1 = log_filename(ggx_alpha, 8,n_sample_per_level, flag_constant, flag_adjust_level, optimizer_string,
                                random_shuffle, allow_neg_weight)

        filename1_full = "./plots/other_logs/" + filename1

        arr1 = read_log(filename1_full)
        plt.plot(arr1,label=filename1)
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')

        plt.title("0.1_compare?")
        plt.show()



    #plot_graph(filename, read_log(filename))