import numpy as np


def roughness_from_gloss(gloss):
    GGX_MAX_SPEC_POWER = 18
    exponent = np.pow(2, gloss * GGX_MAX_SPEC_POWER)
    return np.pow(2.0/(1 + exponent),0.25)


class level_params:
    base_resolution = 128
    def __init__(self, level):
        self.level = level
        self.gloss = (6.0 - level) / (6.0 - 0.0)
        self.roughness = roughness_from_gloss(self.gloss)
        self.res = self.base_resolution >> level

def cubemap_level_params():
    levels = []
    for i in range(7):
        levels.append(level_params(i))

    return levels





if __name__ == '__main__':
    levels = cubemap_level_params()
    for l in levels:
        print(l.roughness)