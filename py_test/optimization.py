import numpy as np
import map_util
import mat_util
from datetime import datetime
from filter import gen_frame_xyz,gen_frame_weight,texel_directions,frame_axis_index,gen_theta_phi
from interpolation import gen_extended_uv_table,get_edge_information
from reference import compute_ggx_distribution_reference

import scipy


"""
Optimization routine

All the mipmaps here have only one channel, which is the weight/contribution

"""

chan_count = 1


def initialize_mipmaps(n_level):
    """
    initialize an all zero mipmap chain
    :param n_level:
    :return:
    """
    mipmaps = []
    for i in range(n_level):
        res = level_to_res(i, n_level)
        tmp = np.zeros((6,res,res,chan_count))
        mipmaps.append(tmp)
    return mipmaps



def level_to_res(level, n_level):
    """
    Get this level's mipmap resolution. Assume the lowest resolution is 2*2
    :param level: the level of mipmap, in shape of [N,]
    :param n_level: how many levels in mipmap
    :return: the resolution
    """

    return np.left_shift(2,int(n_level-1-level))



def process_trilinear_samples(location, level, n_level, initial_weight):
    """
    TODO, add initial portion
    :param location: xyz location in [N,3]
    :param level: mipmap level [N,
    :param n_level number of mipmap level
    :param initial_weight: [N,]
    :return:uv coordinates in [N,2]. high_res portion in [N,] low_res portion in[N,], face_idx in [N,]
    """

    #first clamp the level
    level = np.clip(level, 0, n_level - 1)

    high_level = np.floor(level).astype(np.int32)
    low_level = np.ceil(level).astype(np.int32)

    high_res = np.left_shift(2,(n_level- 1 - high_level).astype(np.int32))
    low_res = np.left_shift(2,(n_level- 1 - low_level).astype(np.int32))

    high_res_portion = 1 - (level - high_level)
    low_res_portion = 1 - high_res_portion

    high_res_portion *= initial_weight
    low_res_portion *= initial_weight

    if chan_count > 1:
        high_res_portion = np.tile(high_res_portion[:, np.newaxis], (1, chan_count))
        low_res_portion = np.tile(low_res_portion[:, np.newaxis], (1, chan_count))


    u,v,face_idx = map_util.xyz_to_uv_vectorized(location)
    uv = np.stack((u,v),axis=-1)
    face_idx = face_idx.astype(np.int32)

    # return {"uv":uv,
    #         "high_res_portion":high_res_portion,"low_res_portion":low_res_portion,
    #         "face_idx":face_idx,
    #         "high_res":high_res,"low_res":low_res,
    #         "high_level":high_level,"low_level":low_level}

    #concatenate high and low information
    uv = np.vstack((uv, uv))
    face_idx = np.concatenate((face_idx, face_idx))
    res = np.concatenate((high_res,low_res))
    level = np.concatenate((high_level,low_level))
    portion = np.concatenate((high_res_portion,low_res_portion))

    return {"uv":uv,
            "portion":portion,
            "face_idx":face_idx,
            "res":res,
            "level":level}



def process_bilinear_samples(trilerp_sample_info:{},mipmaps:[]):
    """

    :param trilerp_sample_info: dictionary returned from process_trilinear_samples
    :param mipmaps: all zero mipmaps
    :return:
    """
    n_level = len(mipmaps)

    uv = trilerp_sample_info['uv']
    portion = trilerp_sample_info['portion']
    face_idx = trilerp_sample_info['face_idx']
    res = trilerp_sample_info['res']
    level = trilerp_sample_info['level']

    #given a uv and its resolution/level, we find the four points that contribute to this bilerped sample
    #the location is expressed using index

    #generate per level uv mesh(with boundary extension)
    uv_grid_list = []
    for level_idx in range(n_level):
        cur_res = 2 << (n_level - level_idx - 1)
        uv_grid_list.append(gen_extended_uv_table(cur_res))

    #loop each level(this design is mainly for numpy parallelization
    for level_idx in range(n_level):
        cur_res = 2 << (n_level - level_idx - 1)
        condition = (level == level_idx)
        cur_uv_grid = uv_grid_list[level_idx]

        cur_uv_ascending_order = cur_uv_grid[0,:,0]
        cur_uv_descending_order = np.flip(cur_uv_ascending_order)


        cur_u = (uv[:,0])[condition]
        cur_v = (uv[:,1])[condition]

        if cur_u.size == 0:
            """
            We do not find any samples located in this level, skip it
            """
            continue

        u_location = np.searchsorted(cur_uv_ascending_order,cur_u,side='left') # j in np array order

        v_location_inv = np.searchsorted(cur_uv_ascending_order,cur_v,side='left') # -i in np array order
        v_location = (cur_res + 2) - 1 - v_location_inv   # extended_res - 1 - v_location   # Visually, this v_location is the upper v(where v has a higher value)

        # there should be no location = 0 or location = extended_res, if so, somewhere is wong
        assert not np.any(u_location == 0)
        assert not np.any(v_location == 0)
        assert not np.any(u_location == cur_res + 2)
        assert not np.any(v_location == cur_res + 2)

        # The following two lines uses uv_grid ordering, where the order of v is inversed

        # note that u[location] and v[location] will never equal u/v if we use searchsorted(side='left')

        #For u, we found the location u[u_location-1] < cur_u <= u[u_location]
        #for v, we found the location v[location] >= cur_v > v[location+1]
        #in the corner(extreme) case, our index is at upper-right side of a uv grid.
        u_location_left = u_location - 1
        #v_location_bot = v_location - 1 which one is correct?
        v_location_bot = v_location + 1

        u_right = cur_uv_ascending_order[u_location]
        u_left = cur_uv_ascending_order[u_location_left]
        v_up = cur_uv_descending_order[v_location]
        v_bot = cur_uv_descending_order[v_location_bot]

        cur_uv = np.stack((cur_u,cur_v),axis=-1)
        cur_portion = portion[condition]
        cur_face = face_idx[condition]

        p0,p1,p2,p3 = bilerp_inverse(cur_uv,cur_portion,u_right,u_left, v_up, v_bot)

        #assign all p0,p1,p2,p3
        #we create a temporary extended mipmap for now, and we will add them back later
        extended_mipmap_cur_level = np.zeros((6,cur_res+2,cur_res+2,mipmaps[level_idx].shape[-1]))


        #p0
        np.add.at(extended_mipmap_cur_level,(cur_face,v_location,u_location_left),p0.reshape(-1,chan_count))
        np.add.at(extended_mipmap_cur_level,(cur_face,v_location,u_location),p1.reshape(-1,chan_count))
        np.add.at(extended_mipmap_cur_level,(cur_face,v_location_bot,u_location_left),p2.reshape(-1,chan_count))
        np.add.at(extended_mipmap_cur_level,(cur_face,v_location_bot,u_location),p3.reshape(-1,chan_count))


        # now move the extended boundaries to where they belong
        original_mipmap_current_level = process_extended_face(extended_mipmap_cur_level)

        mipmaps[level_idx] = original_mipmap_current_level

    return mipmaps

def process_extended_face(extended_mipmap):
    """
    move the value in the extended boundaries to the right place
    :param extended_mipmap: [6,N,N,channel]
    :return: original_mipmap
    """

    def move_edge(tmp_val,tmp_idx,tmp_edge,tmp_reverse):
        if tmp_reverse:
            tmp_val = np.flip(tmp_val)

        if tmp_edge == 'L':
            original_map[tmp_idx][:,0,:] += tmp_val
        elif tmp_edge == 'R':
            original_map[tmp_idx][:,-1,:] += tmp_val
        elif tmp_edge == 'U':
            original_map[tmp_idx][0,:,:] += tmp_val
        elif tmp_edge == 'D':
            original_map[tmp_idx][-1,:,:] += tmp_val
        else:
            raise NotImplementedError



    #for now we ignore the corner
    extended_res = extended_mipmap.shape[1]
    original_res = extended_res - 2
    original_map = np.zeros((6,original_res,original_res,extended_mipmap.shape[-1]))

    for face_idx in range(6):
        idx_info, edge_side_info, reverse_info = get_edge_information(face_idx)

        idx_info = list(idx_info)
        edge_side_info = list(edge_side_info)
        reverse_info = list(reverse_info)


        cur_face = extended_mipmap[face_idx]

        for i in range(len(idx_info)):
            idx = idx_info[i]
            edge = edge_side_info[i]
            reverse = reverse_info[i]

            if i == 0:
                val = cur_face[1:-1,0,:]
            elif i == 1:
                val = cur_face[1:-1,-1,:]
            elif i == 2:
                val = cur_face[0,1:-1,:]
            else:
                val = cur_face[-1,1:-1,:]

            move_edge(val,idx,edge,reverse)

        #copy the center
        original_map[face_idx] += cur_face[1:-1,1:-1,:]



    #process the corner separately.
    """
    There are eight corners, each corner has 3 neighboring faces
    this part is hard-coded
    """

    corner_face_idx = [[4,1,2], #corner of face 4,1,2
                       [4,0,2],
                       [4,1,3],
                       [4,0,3],
                       [5,1,2],
                       [5,0,2],
                       [5,1,3],
                       [5,0,3],]
    corner_pixel_i = [[0,0,-1], # for face 4, i is 0, for face 1, i is 0, for face 2, i is -1
                      [0,0,-1],
                      [-1,-1,0],
                      [-1,-1,0],
                      [0,0,0],
                      [0,0,0],
                      [-1,-1,-1],
                      [-1,-1,-1],]
    corner_pixel_j = [[0,-1,0],
                      [-1,0,-1],
                      [0,-1,0],
                      [-1,0,-1],
                      [-1,0,0],
                      [0,-1,-1],
                      [-1,0,0],
                      [0,-1,-1],]

    for i in range(len(corner_pixel_i)):
        corner_face_list = corner_face_idx[i]
        pixel_location_i = corner_pixel_i[i]
        pixel_location_j = corner_pixel_j[i]

        val = np.zeros(extended_mipmap.shape[-1])

        for j in range(len(corner_face_list)):
            face_idx = corner_face_list[j]
            location_i = pixel_location_i[j]
            location_j = pixel_location_j[j]
            val += extended_mipmap[face_idx,location_i,location_j]

        # one third of the whole contribution will be given to one corner
        val /= 3

        for j in range(len(corner_face_list)):
            face_idx = corner_face_list[j]
            location_i = pixel_location_i[j]
            location_j = pixel_location_j[j]
            original_map[face_idx,location_i,location_j] += val

    return original_map








def bilerp_inverse(location, portion ,u_right,u_left,v_up,v_bot):
    """

    :param location: the location: [N,2]
    :param portion: the portion of this sample
    :param u_right: the right u value
    :param u_left: the left u value
    :param v_up: the up v value
    :param v_bot: the bot v value
    :return:
    """

    u = location[:,0]
    v = location[:,1]

    u_len = u_right - u_left
    v_len = v_up - v_bot

    u0_portion = portion * (u_right - u) / u_len * (v - v_bot) / v_len
    u1_portion = portion * (u - u_left) / u_len * (v - v_bot) / v_len
    u2_portion = portion * (u_right - u) / u_len * (v_up - v) / v_len
    u3_portion = portion * (u - u_left) / u_len * (v_up - v) / v_len

    return u0_portion,u1_portion,u2_portion,u3_portion



def push_back(mipmaps):
    """
    push all lower level values to the higher level
    :param mipmaps: a mipmap that is initialized to zero(if push back func is not called, all weight is zero)
    :return: the highest level of mipmap only, since all the values have been pushed back to this level
    """
    for level_idx in reversed(range(len(mipmaps))):
        if level_idx == 0:
            break

        cur_level = mipmaps[level_idx]
        res = level_to_res(level_idx,7)
        extended_upper_res = res * 2 + 2
        extended_upper_level = np.zeros((6,extended_upper_res,extended_upper_res,cur_level.shape[-1]))


        #instead of looping through each element, we'd better loop through each kernel item
        """
        For four corner 1/64 contribution
        The pattern is:
        
        x - x - x - x - - - 
        - - - - - - - - - -
        x - x - x - x - - - 
        - - - - - - - - - -
        x - x - x - x - - - 
        - - - - - - - - - -
        x - x - x - x - - - 
        - - - - - - - - - -
        - - - - - - - - - -
        - - - - - - - - - -
        
        this is for upper-left pattern, the upper-right is simply moving to the right for 2 unit
        the bottom left is moving this down for 2 unit, the bottom right is moving this down and right both for 2 units
        
        """


        start_idx = [0,3]

        cur_level_unit = cur_level / 64.0

        for row_start in start_idx:
            # Note that, the row_end here needs to be included(while python slice exclude row end
            # row_end = row_start + 2 * (extended_upper_res - 1)
            row_end = row_start + 2 * (res -1) + 2
            for col_start in start_idx:
                col_end = col_start + 2 * (res - 1) + 2
                extended_upper_level[:,row_start:row_end:2,col_start:col_end:2] += cur_level_unit


        """
        For the four 3/64 contributions at left and right side
        """
        cur_level_unit = cur_level_unit * 3

        start_idx_row = [1,2]
        start_idx_col = [0,3]
        for row_start in start_idx_row:
            row_end = row_start + 2 * (res - 1) + 2
            for col_start in start_idx_col:
                col_end = col_start + 2 * (res - 1) + 2
                extended_upper_level[:,row_start:row_end:2, col_start:col_end:2] += cur_level_unit

        """
        For the four 3/64 contributions at up and bot side
        """
        start_idx_row,start_idx_col = start_idx_col,start_idx_row
        for row_start in start_idx_row:
            row_end = row_start + 2 * (res - 1) + 2
            for col_start in start_idx_col:
                col_end = col_start + 2 * (res - 1) + 2
                extended_upper_level[:,row_start:row_end:2, col_start:col_end:2] += cur_level_unit

        """
        For the centric 9/64 contributions in shape of 2 * 2 
        """
        cur_level_unit = cur_level_unit * 3
        tile = np.repeat(np.repeat(cur_level_unit,2,axis=1),2,axis=2)
        extended_upper_level[:,1:-1,1:-1] += tile


        """
        Add the pushed result to the level
        """
        upper_level_original = process_extended_face(extended_upper_level)
        mipmaps[level_idx-1] += upper_level_original

    return mipmaps[0]



def compute_contribution(location,level,initial_weight,n_level):
    """
    compute the contribution of all trilinear samples
    :param location: the location: [N,3] in xyz form
    :param level: level of each trilinear sample [N,]
    :param initial_weight: the initial weight of each sample
    :param n_level: number of mipmap levels in total
    :return: a 6*res*res map
    """
    zero_map = initialize_mipmaps(n_level)
    trilinear_samples = process_trilinear_samples(location, level, n_level, initial_weight)
    bilinear_samples = process_bilinear_samples(trilinear_samples, zero_map)
    result = push_back(bilinear_samples)

    return result


def L1_error_one_texel(ggx_kernel, contribution_map):
    """
    Comparison should be made between *normalized* map
    :param ggx_kernel:
    :param contribution_map:
    :return:
    """

    l1_loss = np.abs(ggx_kernel - contribution_map / np.sum(contribution_map))
    return l1_loss



def test_optimize(ggx_alpha,res,texel_direction,n_sample_per_frame):
    ggx_ref = compute_ggx_distribution_reference(res,ggx_alpha, texel_direction)

    #normalize ggx kernel
    ggx_ref /= np.sum(ggx_ref)

    rng = np.random.default_rng(int(datetime.now().timestamp()))
    #initialize some random parameter very close to zero?
    param = rng.random(n_sample_per_frame * 3 * 5 * 3)
    #param /= 50.0

    result = scipy.optimize.minimize(error_func,param,args=(texel_direction,n_sample_per_frame,ggx_ref),method='BFGS', options={'disp':True,'gtol':1e-4})


    print("done")


def error_func(x, texel_direction, n_sample_per_frame, ggx_ref):
    """
        A wrapper to compute error as scipy required
    :param x:
    :param texel_direction:
    :param n_sample_per_frame:
    :param ggx_ref:
    :return:
    """
    x = x.reshape((5,3,n_sample_per_frame*3))
    error = test_one_texel_full_optimization(texel_direction, n_sample_per_frame, ggx_ref, x)
    return error





def test_one_texel_full_optimization(texel_direction, n_sample_per_frame, ggx_ref, coef_table = None):
    """
    How does frame weight work on the optimization part?
    :param texel_direction: the one texel we are testing on optimization
    :param n_sample_per_frame:
    :param ggx_ref: ggx reference kernel of size (6,res,res,1)
    :return:
    """
    if coef_table is None:
        rng = np.random.default_rng(int(datetime.now().timestamp()))

        #a single coefficient table in shape [5,3,nsample_per_frame * 3]

        coefficient_table = rng.random((5,3,n_sample_per_frame*3))
    else:
        coefficient_table = coef_table


    #frame_weight:
    frame_weight_list = []
    theta_phi_list = []
    unnormalized_texel_direction = texel_direction / np.max(np.abs(texel_direction))
    for i in range(3):
        frame_weight_list.append(gen_frame_weight(unnormalized_texel_direction,i))
        theta_phi_list.append(gen_theta_phi(unnormalized_texel_direction,i))

    all_directions = []
    all_levels = []
    all_weights = []

    for i in range(3):
        X,Y,Z = gen_frame_xyz(texel_direction,i)

        _,_,theta2,phi2 = theta_phi_list[i]

        coeff_start = i * n_sample_per_frame
        coeff_end = coeff_start + n_sample_per_frame

        coeff_x_table = coefficient_table[0,:,coeff_start:coeff_end]
        coeff_y_table = coefficient_table[1,:,coeff_start:coeff_end]
        coeff_z_table = coefficient_table[2,:,coeff_start:coeff_end]
        coeff_level_table = coefficient_table[3,:,coeff_start:coeff_end]
        coeff_weight_table = coefficient_table[4,:,coeff_start:coeff_end]

        coeff_x = coeff_x_table[0] + coeff_x_table[1] * theta2 + coeff_x_table[2] * phi2
        coeff_y = coeff_y_table[0] + coeff_y_table[1] * theta2 + coeff_y_table[2] * phi2
        coeff_z = coeff_z_table[0] + coeff_z_table[1] * theta2 + coeff_z_table[2] * phi2

        coeff_x = np.stack((coeff_x, coeff_x, coeff_x), axis=-1)
        coeff_y = np.stack((coeff_y, coeff_y, coeff_y), axis=-1)
        coeff_z = np.stack((coeff_z, coeff_z, coeff_z), axis=-1)

        level = coeff_level_table[0] + coeff_level_table[1] * theta2 + coeff_level_table[2] * phi2
        weight = coeff_weight_table[0] + coeff_weight_table[1] * theta2 + coeff_weight_table[2] * phi2

        sample_direction = coeff_x * X + coeff_y * Y + coeff_z * Z
        max_dir = np.max(np.abs(sample_direction), axis=-1)
        sample_direction /= np.stack((max_dir, max_dir, max_dir), axis=-1)

        weight_cur_frame = weight * frame_weight_list[i]

        all_weights.append(weight_cur_frame)
        all_levels.append(level)
        all_directions.append(sample_direction)


    sample_direction = np.concatenate((all_directions[0],all_directions[1],all_directions[2]))
    sample_level = np.concatenate((all_levels[0],all_levels[1],all_levels[2]))
    sample_weight = np.concatenate((all_weights[0],all_weights[1],all_weights[2]))


    result = compute_contribution(sample_direction,sample_level,sample_weight,7)

    e_arr = L1_error_one_texel(ggx_ref,result)

    e = np.sum(e_arr)
    #e = np.average(e_arr)

    return e














if __name__ == "__main__":




    #dummy location

    #test if reverse work as desired

    face = 4
    u = 0.8
    v = 0.2
    location_global= map_util.uv_to_xyz((u,v),face)
    location_global = location_global.reshape((1,-1))
    u = 0.2
    location_global2 = map_util.uv_to_xyz((u,v),face).reshape((1,-1))

    location_global = np.concatenate((location_global,location_global2))

    #BFGS optimization test
    #test_optimize(0.01,128,location_global[0:1,:],8)

    ggx = compute_ggx_distribution_reference(128,0.01,location_global[0:1,:])
    #
    test_one_texel_full_optimization(location_global[0:1,:],8,ggx)
    #
    # level_global = np.array([5.4,3.2])
    # n_level_global = 7
    # initial_weight_global = np.array([1.0,1.0])
    #
    # t = initialize_mipmaps(n_level_global)
    # sample_info = process_trilinear_samples(location_global, level_global, n_level_global,initial_weight_global)
    # t = process_bilinear_samples(sample_info,t)
    # final_image = push_back(t)