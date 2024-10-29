from optmization_torch import log_filename,process_cmd

import numpy as np
import matplotlib.pyplot as plt




def read_log(path):
    loss_list = []
    with open(filename, 'r') as f:
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
    n_sample_per_level, ggx_info, flag_constant, flag_adjust_level, optimizer_string = process_cmd()

    if isinstance(ggx_info, int):
        import specular
        info = specular.cubemap_level_params(18)
        ggx_alpha = info[ggx_info].roughness
    else:
        ggx_alpha = ggx_info

    filename = map_util.log_filename(ggx_alpha,n_sample_per_level,flag_constant,flag_adjust_level,optimizer_string)



    plot_graph(filename, read_log(filename))