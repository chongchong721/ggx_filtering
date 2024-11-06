import torch
from interpolation import get_edge_information
import numpy as np
import torch_util

chan_count = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# generate per level uv mesh(with boundary extension)
uv_grid_list = []
for tmp_global_level_idx in range(7):
    cur_res = 2 << (7 - tmp_global_level_idx - 1)
    uv_grid_list.append(torch_util.torch_gen_extended_uv_table(cur_res))


def initialize_mipmaps_all(n_level,n_sample_per_level):
    mipmaps = []
    for i in range(n_level):
        res = torch_util.level_to_res(i, n_level)
        tmp = torch.zeros((n_sample_per_level,6,res,res,chan_count),device=device)
        mipmaps.append(tmp)
    return mipmaps




def move_edge_all_locations(tmp_val,tmp_idx,tmp_edge,tmp_reverse, original_map):
    """

    :param tmp_val: in shape of [n_sample_per_level,res,chan_count]
    :param tmp_idx:
    :param tmp_edge:
    :param tmp_reverse:
    :return:
    """
    if tmp_reverse:
        tmp_val = torch.flip(tmp_val,dims = [1])

    if tmp_edge == 'L':
        original_map[:,tmp_idx,:, 0, :] += tmp_val
    elif tmp_edge == 'R':
        original_map[:,tmp_idx,:, -1, :] += tmp_val
    elif tmp_edge == 'U':
        original_map[:,tmp_idx,0, :, :] += tmp_val
    elif tmp_edge == 'D':
        original_map[:,tmp_idx,-1, :, :] += tmp_val
    else:
        raise NotImplementedError





def process_extended_face_all_locations(extended_mipmap):
    n_sample_per_level = extended_mipmap.shape[0]
    extended_res = extended_mipmap.shape[2]
    original_res = extended_res - 2

    original_map = torch.zeros((n_sample_per_level,6,original_res,original_res,chan_count),device=device)




    for face_idx in range(6):
        idx_info, edge_side_info, reverse_info = get_edge_information(face_idx)

        idx_info = list(idx_info)
        edge_side_info = list(edge_side_info)
        reverse_info = list(reverse_info)

        cur_face = extended_mipmap[:,face_idx,...]

        for i in range(len(idx_info)):
            idx = idx_info[i]
            edge = edge_side_info[i]
            reverse = reverse_info[i]

            if i == 0:
                val = cur_face[:,1:-1, 0, :]
            elif i == 1:
                val = cur_face[:,1:-1, -1, :]
            elif i == 2:
                val = cur_face[:,0, 1:-1, :]
            else:
                val = cur_face[:,-1, 1:-1, :]

            move_edge_all_locations(val, idx, edge, reverse, original_map)

        original_map[:,face_idx] += cur_face[:,1:-1, 1:-1, :]

        # process the corner separately.
        """
        There are eight corners, each corner has 3 neighboring faces
        this part is hard-coded
        """

    corner_face_idx = [[4, 1, 2],  # corner of face 4,1,2
                       [4, 0, 2],
                       [4, 1, 3],
                       [4, 0, 3],
                       [5, 1, 2],
                       [5, 0, 2],
                       [5, 1, 3],
                       [5, 0, 3], ]
    corner_pixel_i = [[0, 0, -1],  # for face 4, i is 0, for face 1, i is 0, for face 2, i is -1
                      [0, 0, -1],
                      [-1, -1, 0],
                      [-1, -1, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [-1, -1, -1],
                      [-1, -1, -1], ]
    corner_pixel_j = [[0, -1, 0],
                      [-1, 0, -1],
                      [0, -1, 0],
                      [-1, 0, -1],
                      [-1, 0, 0],
                      [0, -1, -1],
                      [-1, 0, 0],
                      [0, -1, -1], ]

    for i in range(len(corner_pixel_i)):
        corner_face_list = corner_face_idx[i]
        pixel_location_i = corner_pixel_i[i]
        pixel_location_j = corner_pixel_j[i]

        val = torch.zeros(n_sample_per_level,extended_mipmap.shape[-1], device=device)

        for j in range(len(corner_face_list)):
            face_idx = corner_face_list[j]
            location_i = pixel_location_i[j]
            location_j = pixel_location_j[j]
            val += extended_mipmap[:,face_idx, location_i, location_j]

        # one third of the whole contribution will be given to one corner
        val /= 3

        for j in range(len(corner_face_list)):
            face_idx = corner_face_list[j]
            location_i = pixel_location_i[j]
            location_j = pixel_location_j[j]
            original_map[:,face_idx, location_i, location_j] += val

    return original_map

def process_trilinear_samples_all_locations(all_locations, all_levels, all_initial_weights ,n_level, n_sample_per_level, sample_idx):
    """

    :param all_locations:
    :param all_levels:
    :param all_initial_weights
    :param n_level:
    :param n_sample_per_level:
    :param sample_idx:
    :return:
    """
    high_level = torch.floor(all_levels)
    low_level = torch.ceil(all_levels)

    high_res_portion = 1 - (all_levels - high_level)
    low_res_portion = 1 - high_res_portion

    high_res_portion *= all_initial_weights
    low_res_portion *= all_initial_weights

    u, v, face_idx = torch_util.torch_xyz_to_uv_vectorized(all_locations)
    uv = torch.stack((u, v), dim=-1)
    face_idx = face_idx.int()

    uv = torch.vstack((uv, uv))
    face_idx = torch.concatenate((face_idx, face_idx))
    level = torch.concatenate((high_level, low_level))
    portion = torch.concatenate((high_res_portion, low_res_portion))

    sample_idx = sample_idx.repeat(2)


    return {"uv": uv,
            "portion": portion,
            "face_idx": face_idx,
            "level": level,
            "sample_idx": sample_idx}



def process_bilinear_samples_all_locations(trilerp_sample_info: {}, mipmaps: [], n_sample_per_level):
    n_level = len(mipmaps)

    uv = trilerp_sample_info['uv']
    portion = trilerp_sample_info['portion']
    face_idx = trilerp_sample_info['face_idx']
    level = trilerp_sample_info['level']
    sample_idx = trilerp_sample_info['sample_idx']



    for level_idx in range(n_level):
        cur_res = 2 << (n_level - level_idx - 1)
        cur_res_inv = 1 / cur_res
        condition = (level == level_idx)
        cur_uv_grid = uv_grid_list[level_idx]

        cur_uv_ascending_order = cur_uv_grid[0, :, 0].contiguous()
        cur_uv_descending_order = torch.flip(cur_uv_ascending_order,[0])

        cur_u = (uv[:, 0])[condition]
        cur_v = (uv[:, 1])[condition]

        cur_sample_idx = sample_idx[condition]

        if cur_u.size()[0] == 0:
            """
            We do not find any samples located in this level, skip it
            """
            continue

        # u_location = torch.searchsorted(cur_uv_ascending_order, cur_u, right=False)  # j in np array order
        #
        # v_location_inv = torch.searchsorted(cur_uv_ascending_order, cur_v, right=False)  # -i in np array order
        # v_location = (cur_res + 2) - 1 - v_location_inv  # extended_res - 1 - v_location   # Visually, this v_location is the upper v(where v has a higher value)

        u_location = torch.ceil((cur_u + 0.5 * cur_res_inv) / cur_res_inv).int()
        v_location_inv = torch.ceil((cur_v + 0.5 * cur_res_inv) / cur_res_inv).int()
        v_location = (cur_res + 2) - 1 - v_location_inv

        # The following two lines uses uv_grid ordering, where the order of v is inversed

        # note that u[location] and v[location] will never equal u/v if we use searchsorted(side='left')

        # For u, we found the location u[u_location-1] < cur_u <= u[u_location]
        # for v, we found the location v[location] >= cur_v > v[location+1]
        # in the corner(extreme) case, our index is at upper-right side of a uv grid.
        u_location_left = u_location - 1
        # v_location_bot = v_location - 1 which one is correct?
        v_location_bot = v_location + 1

        u_right = cur_uv_ascending_order[u_location]
        u_left = cur_uv_ascending_order[u_location_left]
        v_up = cur_uv_descending_order[v_location]
        v_bot = cur_uv_descending_order[v_location_bot]

        cur_uv = torch.stack((cur_u, cur_v), dim=-1)
        cur_portion = portion[condition]
        cur_face = face_idx[condition]

        p0, p1, p2, p3 = torch_util.bilerp_inverse(cur_uv, cur_portion, u_right, u_left, v_up, v_bot)

        # assign all p0,p1,p2,p3
        # we create a temporary extended mipmap for now, and we will add them back later
        extended_mipmap_cur_level = torch.zeros((n_sample_per_level, 6, cur_res + 2, cur_res + 2, mipmaps[level_idx].shape[-1]),dtype=p0.dtype, device=device)

        extended_mipmap_cur_level.index_put_((cur_sample_idx,cur_face,v_location,u_location_left),p0.reshape((-1,chan_count)),accumulate=True)
        extended_mipmap_cur_level.index_put_((cur_sample_idx,cur_face,v_location,u_location),p1.reshape((-1,chan_count)),accumulate=True)
        extended_mipmap_cur_level.index_put_((cur_sample_idx,cur_face,v_location_bot,u_location_left),p2.reshape((-1,chan_count)),accumulate=True)
        extended_mipmap_cur_level.index_put_((cur_sample_idx,cur_face,v_location_bot,u_location),p3.reshape((-1,chan_count)),accumulate=True)


        # now move the extended boundaries to where they belong
        original_mipmap_current_level = process_extended_face_all_locations(extended_mipmap_cur_level)

        mipmaps[level_idx] = original_mipmap_current_level

    return mipmaps



def push_back_all_lcoations(mipmaps, j_inv = False):
    n_sample_per_level = mipmaps[0].shape[0]
    for level_idx in reversed(range(len(mipmaps))):
        if level_idx == 0:
            break

        #cur_level in shape of (n_sample_per_level,6,res,res,1)
        cur_level = mipmaps[level_idx]
        res = torch_util.level_to_res(level_idx, 7)
        extended_upper_res = res * 2 + 2
        extended_upper_level = torch.zeros((n_sample_per_level,6, extended_upper_res, extended_upper_res, cur_level.shape[-1]),
                                           device=device)

        """
        xyz_bilerp_upper has a resolution of 2res * 2res, the neighboring 4 are the jacobian used for the bilerp samples
        """
        xyz_bilerp_upper = torch_util.downsample_xyz_pattern_full(extended_upper_res)
        if not j_inv:
            j = torch_util.torch_jacobian_vertorized(xyz_bilerp_upper)
        else:
            j = 1 / torch_util.torch_jacobian_vertorized(xyz_bilerp_upper)
        j_sum = j.view(6, res, 2, res, 2).sum(dim=(2, 4))

        j_pattern_upper_left = j[:, 0::2, 0::2]
        j_pattern_upper_right = j[:, 0::2, 1::2]
        j_pattern_bot_left = j[:, 1::2, 0::2]
        j_pattern_bot_right = j[:, 1::2, 1::2]

        j_selection = torch.zeros((2, 2) + j_pattern_upper_left.shape, device=device)
        j_selection[0, 0] = j_pattern_upper_left
        j_selection[0, 1] = j_pattern_upper_right
        j_selection[1, 0] = j_pattern_bot_left
        j_selection[1, 1] = j_pattern_bot_right

        cur_level_unit_selection = torch.zeros(j_selection.shape[0:2] + (n_sample_per_level,) + j_selection.shape[2:]  + (1,), device=device)

        #automatic broadcast on the first dimension? Is this true?
        cur_level_unit_selection[0, 0] = cur_level * (1 / 128.0 + 1 / 32.0 * j_pattern_upper_left / j_sum).unsqueeze(-1)
        cur_level_unit_selection[0, 1] = cur_level * (1 / 128.0 + 1 / 32.0 * j_pattern_upper_right / j_sum).unsqueeze(-1)
        cur_level_unit_selection[1, 0] = cur_level * (1 / 128.0 + 1 / 32.0 * j_pattern_bot_left / j_sum).unsqueeze(-1)
        cur_level_unit_selection[1, 1] = cur_level * (1 / 128.0 + 1 / 32.0 * j_pattern_bot_right / j_sum).unsqueeze(-1)

        start_idx = [0, 3]
        for row_start in start_idx:
            """
            0,0 is upper-left(J[0,0])
            0,3 is upper-right(J[0,1])
            3,0 is bottom-left(J[1,0])
            3,3 is bottom-right(J[1,1])
            """

            # Note that, the row_end here needs to be included(while python slice exclude row end
            # row_end = row_start + 2 * (extended_upper_res - 1)
            row_end = row_start + 2 * (res - 1) + 2
            for col_start in start_idx:
                col_end = col_start + 2 * (res - 1) + 2
                extended_upper_level[:,:, row_start:row_end:2, col_start:col_end:2] += cur_level_unit_selection[int(row_start/2),int(col_start/2)]

        """
                For the four 3/64 contributions at left and right side
                """
        # cur_level_unit = cur_level_unit * 3

        start_idx_row = [1, 2]
        start_idx_col = [0, 3]

        """
        1,0 is top-left(J[0,0]
        1,3 is top-right(J[0,1])
        2,0 is bot-left(J[1,0])
        2,3 is bot-right(J[1,1])
        """

        for row_start in start_idx_row:
            row_end = row_start + 2 * (res - 1) + 2
            for col_start in start_idx_col:
                col_end = col_start + 2 * (res - 1) + 2
                extended_upper_level[:,:, row_start:row_end:2, col_start:col_end:2] += cur_level_unit_selection[
                                                                                         int(row_start / 2), int(
                                                                                             col_start / 2)] * 3

        """
        For the four 3/64 contributions at up and bot side
        """

        """
        0,1 is upper-left
        0,2 is upper-right
        3,1 is bot-left
        3,2 is bot-right
        """

        start_idx_row, start_idx_col = start_idx_col, start_idx_row
        for row_start in start_idx_row:
            row_end = row_start + 2 * (res - 1) + 2
            for col_start in start_idx_col:
                col_end = col_start + 2 * (res - 1) + 2
                extended_upper_level[:,:, row_start:row_end:2, col_start:col_end:2] += cur_level_unit_selection[
                                                                                         int(row_start / 2), int(
                                                                                             col_start / 2)] * 3

        """
        For the centric 9/64 contributions in shape of 2 * 2 
        """
        # cur_level_unit = cur_level_unit * 3

        #j_sum_ext in shape of (6,ext_res,ext_res,1)
        j_sum_ext = torch.repeat_interleave(torch.repeat_interleave(j_sum, 2, dim=1), 2, dim=2)
        tile = torch.repeat_interleave(torch.repeat_interleave(cur_level, 2, dim=2), 2, dim=3)
        #automatic broadcast?
        cur_level_unit = 9 * tile * (1 / 128.0 + 1 / 32.0 * j / j_sum_ext).unsqueeze(-1)
        extended_upper_level[:,:, 1:-1, 1:-1] += cur_level_unit

        """
        Add the pushed result to the level
        """
        upper_level_original = process_extended_face_all_locations(extended_upper_level)
        mipmaps[level_idx - 1] += upper_level_original
    return mipmaps[0]


def compute_contribution_full(all_locations,all_levels, all_weights, sample_idx, n_level, n_sample_per_level):
    """
    There should be n_sample_per_level * 3 * n_sample_per_frame samples in total
    :param all_locations:
    :param all_levels:
    :param all_weights:
    :param n_level: 7
    :param sample_idx: in shape of n_sample_per_level * 3 * n_sample_per_frame
    :return:
    """
    zero_map = initialize_mipmaps_all(n_level, n_sample_per_level)
    trilinear_samples = process_trilinear_samples_all_locations(all_locations, all_levels, all_weights, n_level, n_sample_per_level, sample_idx)
    bilinear_samples = process_bilinear_samples_all_locations(trilinear_samples, zero_map, n_sample_per_level)
    result = push_back_all_lcoations(bilinear_samples,False)

    return result