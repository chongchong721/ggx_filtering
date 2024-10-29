import numpy as np


class float4:
    def __init__(self, v0,v1,v2,v3):
        self.array = np.array([v0,v1,v2,v3])

    def __getitem__(self, key):
        return self.array[key]



def fetch_coefficient(mode:str,level_idx,parameter_idx):
    """

    :param mode:'const' or  'quad'
    :param level_idx: number of level
    :param parameter_idx: nth parameter
    :return:
    if we are fetching 32*3 coeffcient, we get a float4 array in shape of [3,N] where N is the number of samples to take for each texel
    if fetching 8*3, we get a float array in shape of [3,N]
    """
    if mode == 'const':
        raise NotImplementedError
    elif mode == 'quad':
        return coefficient_quad32[level_idx,parameter_idx]
    else:
        raise NotImplementedError


def get_coeff_table(const:bool, n_sample_per_frame):
    if const:
        if n_sample_per_frame == 8:
            table = np.load("./refs/const8.npy")
        elif n_sample_per_frame == 16:
            table = np.load("./refs/const16.npy")
        elif n_sample_per_frame == 32:
            table = np.load("./refs/const32.npy")
        else:
            raise NotImplementedError
    else:
        if n_sample_per_frame == 32:
            table = np.load("./refs/quad32.npy")
        else:
            raise NotImplementedError

    return table



def expand_float4(coeff:np.ndarray):
    dimension = coeff.ndim

    n_sample_per_frame = int(coeff.shape[-1] * 4 / 3)

    new_table = np.zeros((coeff.shape[:-1] + (n_sample_per_frame * 3,)))


    if dimension == 3: #constant
        for i in range(coeff.shape[0]):
            for j in range(coeff.shape[1]):
                for k in range(coeff.shape[2]):
                    new_table[i][j][4*k] = coeff[i][j][k][0]
                    new_table[i][j][4*k+1] = coeff[i][j][k][1]
                    new_table[i][j][4*k+2] = coeff[i][j][k][2]
                    new_table[i][j][4*k+3] = coeff[i][j][k][3]

    elif dimension == 4: #quad
        for i in range(coeff.shape[0]):
            for j in range(coeff.shape[1]):
                for k in range(coeff.shape[2]):
                    for l in range(coeff.shape[3]):
                        new_table[i][j][k][4*l] = coeff[i][j][k][l][0]
                        new_table[i][j][k][4*l+1] = coeff[i][j][k][l][1]
                        new_table[i][j][k][4*l+2] = coeff[i][j][k][l][2]
                        new_table[i][j][k][4*l+3] = coeff[i][j][k][l][3]

    else:
        raise NotImplementedError

    return new_table



if __name__ == "__main__":
    # quad32_np = expand_float4(coefficient_quad32)
    # const8_np = expand_float4(coefficient_const8)
    # const16_np = expand_float4(coeffs_const16)
    # const32_np = expand_float4(coeffs_const32)
    #
    # np.save("./refs/quad32.npy",quad32_np)
    # np.save("./refs/const8.npy",const8_np)
    # np.save("./refs/const16.npy",const16_np)
    # np.save("./refs/const32.npy",const32_np)
    #
    #
    # print(coefficient_const8.shape)
    # print(coefficient_quad32.shape)

    print("Done")