import numpy as np
from scipy.stats import expon


def roughness_from_gloss(gloss, max_spec_power = 16, sqrt = True):
    """
    To match ref and filtered result for level 1, though level 3 still does not match
    GGX_MAX_SPEC_POWER should be 16 and the final roughness computation is a square root(not ^0.25
    :param gloss:
    :return:
    """
    assert max_spec_power == 16 or max_spec_power == 18
    GGX_MAX_SPEC_POWER = max_spec_power
    exponent = np.pow(2, gloss * GGX_MAX_SPEC_POWER)
    #return np.pow(2.0/(1 + exponent),0.25)
    #?? See Material Advances in Call of Duty:WWII
    if sqrt:
        power = 0.5
    else:
        power = 0.25
    if GGX_MAX_SPEC_POWER == 16:
        return np.power(2.0/(2.0 + exponent),power)
    elif GGX_MAX_SPEC_POWER == 18:
        return np.power(2.0/(1.0 + exponent),power)


class level_params:
    base_resolution = 128
    def __init__(self, level, max_spec_power = 16, sqrt = True):
        self.level = level
        self.gloss = (6.0 - level) / (6.0 - 0.0)
        self.roughness = roughness_from_gloss(self.gloss, max_spec_power, sqrt)
        self.res = self.base_resolution >> level

def cubemap_level_params(max_power = 16, sqrt = True):
    levels = []
    for i in range(7):
        levels.append(level_params(i, max_power, sqrt))

    return levels





if __name__ == '__main__':
    levels = cubemap_level_params(18)
    for l in levels:
        print(l.roughness)