import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator

import map_util
import skimage


"""
The quadratic b-spline interpolation to generate downsampled cubemap

A 4x4 kernel of

---------------------------
|1 3 3 1|
|3 9 9 3|   * (1/64)
|3 9 9 3|
|1 3 3 1|
---------------------------

Then weighted by a tuned Jacobian 1/2 * （1 + J(x,y,z))

samples in other faces need to be retrieved when computing the boundary samples

"""






"""
For a face. Assume the face resolution is N
This means the face have a length of N and each sample is located in the center of the pixel

So, the leftmost sample will have a coordinate of 0.5, and the 


"""


def bilerp(array,location):
    """

    :param array: 2 * 2 vector
    :param location: location in ij order, must be within (0,1)
    :return:
    """

    # lerp i
    ti = location[1]
    s1 = array[0,0] * (1-ti) + array[0,1] * ti
    s2 = array[1,0] * (1-ti) + array[1,1] * ti

    # lerp s1,s2
    tj = location[0]
    s = s1 * (1-tj) + s2 * tj

    return s


def fetch_boundary(face, edge_side:str):
    """
    fetch the boundary line of a face,
    the function returns one of the right/left/up/down boundary of the face
    :param face: the face to fetch the line
    :param edge_side: indicating which boundary line to fetch
    :return: vector in shape of (N,3)?
    """
    assert face.shape[0] == face.shape[1]

    if edge_side == 'L':
        # return the first column
        return face[:,0,:]
    elif edge_side == 'R':
        # return the last column
        return face[:,-1,:]
    elif edge_side == 'U':
        # return the first row
        return face[0,:,:]
    elif edge_side == 'D':
        # return the last row
        return face[-1,:,:]
    else:
        raise NotImplementedError




def get_edge_information(face_idx):
    """
    return the information of edges, which faces are they belonging to, which side are the edges, are they reversed or not

    How we generate this is documented in function extend_face(faces,face_idx)


    :param face_idx:
    :return:
    """

    # +x -x +y -y +z -z
    left_idxes =  \
        [4,5,1,1,1,0]
    right_idxes = \
        [5,4,0,0,0,1]
    up_idxes =    \
        [2,2,5,4,2,2]
    down_idxes =  \
        [3,3,4,5,3,3]

    # choosing left/right/up/down
    left_edge_select_sides =  \
        ['R', 'R', 'U', 'D', 'R', 'R']
    right_edge_select_sides = \
        ['L', 'L', 'U', 'D', 'L', 'L']
    up_edge_select_sides = \
        ['R', 'L', 'U', 'D', 'D', 'U']
    down_edge_select_sides = \
        ['R', 'L', 'U', 'D', 'U', 'D']

    left_reverse_flags = \
        [False,False,False,True,False,False]
    right_reverse_flags = \
        [False,False,True,False,False,False]
    up_reverse_flags = \
        [True,False,True,False,False,True]
    down_reverse_flags = \
        [False,True,False,True,False,True]

    left_idx,right_idx,up_idx,down_idx = (
        left_idxes[face_idx],right_idxes[face_idx],up_idxes[face_idx],down_idxes[face_idx])
    left_edge_side,right_edge_side,up_edge_side,down_edge_side = (
        left_edge_select_sides[face_idx],right_edge_select_sides[face_idx],up_edge_select_sides[face_idx],down_edge_select_sides[face_idx])
    left_reverse_flag,right_reverse_flag,up_reverse_flag,down_reverse_flag = (
        left_reverse_flags[face_idx],right_reverse_flags[face_idx],up_reverse_flags[face_idx],down_reverse_flags[face_idx])


    return (left_idx,right_idx,up_idx,down_idx),(left_edge_side,right_edge_side,up_edge_side,down_edge_side),(left_reverse_flag,right_reverse_flag,up_reverse_flag,down_reverse_flag)



def gen_boundary_uv_for_interp(edge_side,reverse_flag,cubemap_res):
    """
    How to generate this is documented in downsample
    :param edge_side: L(eft) R(ight) U(p) D(own)
    :param reverse_flag: True or False
    :param cubemap_res:
    :return:
    """
    uv = np.zeros((cubemap_res,2))
    uv_ascending_order = map_util.create_pixel_index(cubemap_res,1)

    uv_ascending_order = uv_ascending_order / cubemap_res

    uv_min = uv_ascending_order.min()
    uv_max = uv_ascending_order.max()

    epsilon = 1.0 - uv_max
    assert 1.0 - uv_max == uv_min

    if edge_side == "L":
        # u is 0 while v goes from 1 to 0
        uv[:,0] = np.full(cubemap_res,uv_min - 2 * epsilon)
        uv[:,1] = np.flip(uv_ascending_order)
    elif edge_side == "R":
        # u is 1 while v gose from 1 to 0
        uv[:,0] = np.full(cubemap_res,uv_max + 2 * epsilon)
        uv[:,1] = np.flip(uv_ascending_order)
    elif edge_side == "U":
        # v is 1 while u goes from 0 to 1
        uv[:,0] = uv_ascending_order
        uv[:,1] = np.full(cubemap_res,uv_max + 2 * epsilon)
    elif edge_side == "D":
        # v is 0 while u goes from 0 to 1
        uv[:,0] = uv_ascending_order
        uv[:,1] = np.full(cubemap_res,uv_min - 2 * epsilon)
    else:
        raise NotImplementedError

    if reverse_flag:
        uv = np.flip(uv,axis=0)

    return uv




def gen_boundary_uv(edge_side,reverse_flag,cubemap_res):

    uv = np.zeros((cubemap_res,2))
    uv_ascending_order = map_util.create_pixel_index(cubemap_res,1)
    uv_ascending_order = uv_ascending_order / cubemap_res
    uv_min = uv_ascending_order.min()
    uv_max = uv_ascending_order.max()

    if edge_side == "L":
        # u is 0 while v goes from 1 to 0
        uv[:,0] = np.full(cubemap_res,uv_min)
        uv[:,1] = np.flip(uv_ascending_order)
    elif edge_side == "R":
        # u is 1 while v gose from 1 to 0
        uv[:,0] = np.full(cubemap_res,uv_max)
        uv[:,1] = np.flip(uv_ascending_order)
    elif edge_side == "U":
        # v is 1 while u goes from 0 to 1
        uv[:,0] = uv_ascending_order
        uv[:,1] = np.full(cubemap_res,uv_max)
    elif edge_side == "D":
        # v is 0 while u goes from 0 to 1
        uv[:,0] = uv_ascending_order
        uv[:,1] = np.full(cubemap_res,uv_min)
    else:
        raise NotImplementedError

    if reverse_flag:
        uv = np.flip(uv,axis=0)

    return uv



def return_extended_uv(cubemap_res,face_idx):
    """
    When computing the Jacobian
    we need to know the uv coordinates of the extended faces so that we can compute the xyz
    This function generate four arrays of size (N,2)
        - left
        - right
        - up
        - down
    Arrays are in numpy ij order(same order as the face)
    :param faces:
    :param face_idx:
    :return:
    """
    idx_info,edge_side_info,reverse_info = get_edge_information(face_idx)
    left_idx,right_idx,up_idx,down_idx = idx_info
    left_edge_side,right_edge_side,up_edge_side,down_edge_side = edge_side_info
    left_reverse_flag,right_reverse_flag,up_reverse_flag,down_reverse_flag = reverse_info




    #generate left one
    left_uv = gen_boundary_uv(left_edge_side,left_reverse_flag,cubemap_res)
    right_uv = gen_boundary_uv(right_edge_side,right_reverse_flag,cubemap_res)
    up_uv = gen_boundary_uv(up_edge_side,up_reverse_flag,cubemap_res)
    down_uv = gen_boundary_uv(down_edge_side,down_reverse_flag,cubemap_res)

    return left_uv,right_uv,up_uv,down_uv



def extend_face(faces,face_idx):
    """
    When the kernel is convolving near the boundary, we need values from other faces.
    This function copy some points from other faces and form a larger face so that we can do the
    convolution easily

    The faces are in this order https://en.wikipedia.org/wiki/Cube_mapping#Memory_addressing

    How to extend? Since the next level mipmap is shrinking the height/width by 2, so we only need to
    add one column/row to this face. However, we do not know the four corner of this extended face since
    this corner does not exist in a folded cube.

    Let's list each face's neighbor edges, in the order of left right up down

    R|L|U|D +- x|y|z means the right/left/up/down edge of  the -= x/y/z face

    --------- left --------- right --------- up --------- down
    +x       R +z(4)        L -z(5)        R +y(2)       R -y(3)
    -x       R -z(5)        L +z(4)        L +y(2)       L -y(3)
    +y       U -x(1)        U +x(0)        U -z(5)       U +z(4)
    -y       D -x(1)        D +x(0)        D +z(4)       D -z(5)
    +z       R -x(1)        L +x(0)        D +y(2)       U -y(3)
    -z       R +x(0)        L -x(1)        U +y(2)       D -y(3)



    After fetching, some of the boundary lines are in reverse order(numpy array is from left to right from up to bot)
    This issue needs to be fixed. Here are the lists( RV mark means the order is reversed


    --------- left --------- right --------- up --------- down
    +x                                       RV
    -x                                                     RV
    +y                        RV             RV
    -y         RV                                          RV
    +z
    -z                                       RV            RV



    :param faces: faces are all six faces in shape of (6,N,N,3)
    :param face_idx: The index of the face we want to extend, we need to use the index to determine how to extend the
     face.
    :return:
    """

    channel_count = faces.shape[3]


    idx_info,edge_side_info,reverse_info = get_edge_information(face_idx)


    left_idx,right_idx,up_idx,down_idx = idx_info
    left_edge_side,right_edge_side,up_edge_side,down_edge_side = edge_side_info
    left_reverse_flag,right_reverse_flag,up_reverse_flag,down_reverse_flag = reverse_info

    # fetch the left boundary to be extended
    left_bound = fetch_boundary(faces[left_idx],left_edge_side)
    if left_reverse_flag:
        left_bound = np.flip(left_bound,axis=0)
    # fetch the right boundary
    right_bound = fetch_boundary(faces[right_idx],right_edge_side)
    if right_reverse_flag:
        right_bound = np.flip(right_bound,axis=0)
    # up
    up_bound = fetch_boundary(faces[up_idx],up_edge_side)
    if up_reverse_flag:
        up_bound = np.flip(up_bound,axis=0)
    # down
    down_bound = fetch_boundary(faces[down_idx],down_edge_side)
    if down_reverse_flag:
        down_bound = np.flip(down_bound,axis=0)

    # compute the 4 corners According to 8.13.1 of OpenGL specification
    # https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf

    # corners in shape of (3,)
    up_left_corner = (up_bound[0] + left_bound[0] + faces[face_idx][0,0]) / 3
    up_right_corner = (up_bound[-1] + right_bound[0] + faces[face_idx][0,-1]) / 3
    down_left_corner = (down_bound[0] + left_bound[-1] + faces[face_idx][-1,0]) / 3
    down_right_corner = (down_bound[-1] + right_bound[-1] + faces[face_idx][-1,-1]) / 3

    #TODO: check the above works

    face_res = faces[face_idx].shape[0]

    face_extended = np.zeros((face_res+2,face_res+2,channel_count))

    face_extended[1:-1,0,:] = left_bound
    face_extended[1:-1,-1,:] = right_bound
    face_extended[0,1:-1,:] = up_bound
    face_extended[-1,1:-1,:] = down_bound

    face_extended[0,0,:] = up_left_corner
    face_extended[0,-1,:] = up_right_corner
    face_extended[-1,0,:] = down_left_corner
    face_extended[-1,-1,:] = down_right_corner


    face_extended[1:-1,1:-1,:] = faces[face_idx]

    return face_extended





def recurrence():
    pattern = np.array([[1/8],[3/8],[3/8],[1/8]])

    kernel = np.matmul(pattern,pattern.T)


    print(kernel)


def downsample(faces_extended,face_idx):
    """
    Downsample the face using bspline interpolation

    A 2*2 bilinear interpolation with special weight is first used (to account for the 1,3,3,9 interpolation weight)
    and then a 2D box filter is used to average the four bilinear samples

    To perform the first step, an intermediate downsampled face should be generated, All the samples in this face lie
    within the original face(without extension). The resolution of this downsampled face should be ((N+2)/2,(N+2)/2,3)

    :param faces_extended: faces extended in shape of (6,N+2,N+2,3)
    :param face_idx: the index of the face to be downsampled
    :return:
    """
    assert faces_extended.shape[1] == faces_extended.shape[2]

    channel_count = faces_extended.shape[3]

    face_extended_res = faces_extended.shape[1]
    face_res = face_extended_res - 2
    face_extended = faces_extended[face_idx]
    # fetch bilinear interpolation samples
    interpolators = []

    row_range = np.arange(0,face_extended_res)
    col_range = np.arange(0,face_extended_res)

    for chan_idx in range(channel_count):
        interpolator = scipy.interpolate.RegularGridInterpolator((row_range,col_range),face_extended[:,:,chan_idx])
        interpolators.append(interpolator)


    # actual point to interpolate
    """
    For each 2*2 kernel, p shoule be at the 1/4 location at either upper-left,upper-right,lower-left or lower-right
    The example is at lower-right
    x(0,0)      x(0,1)
    

            p(3/4,3/4)
    x(1,0)      x(1,1)
    """


    #generate all the points needed in the above comment
    point_idx_pattern1 = np.arange(3/4,face_extended_res - 1 - 1/4, 2)
    point_idx_pattern2 = np.arange(2+1/4,face_extended_res - 1 - 3/4 + 2, 2)

    point_idx = np.insert(point_idx_pattern2, np.arange(0,point_idx_pattern2.size),point_idx_pattern1)

    point_xs,point_ys = np.meshgrid(point_idx,point_idx,indexing='ij')

    points = np.zeros((point_xs.shape[0],point_xs.shape[1],channel_count))

    for chan_idx in range(channel_count):
        interpolator = interpolators[chan_idx]
        points[:,:,chan_idx] = interpolator((point_xs,point_ys))

        #testcode
        #cur_face = faces_extended[face_idx,:,:,chan_idx]
        #test = bilerp(cur_face[0:2, 0:2], [0.75, 0.75])


    """
    Now there are cube_res * cube_res points
    Each of the downsampled point uses its nearst 4 neighbor and take the average
    
    e.g. downsample[0,0] = np.mean(points[0:2,0:2])
         downsample[0,1] = np.mean(points[0:2,2:4])
         downsample[1,0] = np.mean(points[2:4,0:2])
         downsample[1,1] = np.mean(points[2:4,2:4])
         
    However, before we can do this, we need to add the Jacobian weighting for the four bilinear samples
    
    The four bilinear points have its own interpolated u,v so that we are able to compute the Jacobian
    
    What about the first/last row/col that is out of bound?
    
    N * N * 3 vector storing each point's xyz coordinates
    
    """


    """
    
    For the boundary interpolated sample:
    e.g.  face 4(+z)   The upper-left sample, it has three samples that are not in the face, with u,v(other faces) 
    (?,?)                 |               (ε,ε) (face 2)
                          |
                          |   
    -------------------------------------------------
                          |    p
                          |
    (1-ε,1-ε) (face 1)    |               (ε,1-ε)
    
    The interpolated sample lies at the p location,
    the UV of this sample is guaranteed to be within the surface. It can be computed analytically?(accurately?).
    It should be something like: ε - 2ε * 1/4 = ε/2(the distance between two samples are 2ε
    
    Thus, we could set 
    - left extended boundary:  u to be -ε, v as usual
    - right extended boundary: u to be 1 + ε, v sa usual
    - up extended boundary:    v to be 1 + ε, u as usual
    - down extended boundary:  v to be -ε, u as usual
    
    up-left corner uv(-ε,1+ε)
    up-right corner uv(1+ε,1+ε)
    bottom-left corner uv(-ε,-ε)
    bottom-right corner uv(1+ε,-ε)
    """
    idx_info, edge_side_info, reverse_info = get_edge_information(face_idx)
    left_edge_side, right_edge_side, up_edge_side, down_edge_side = edge_side_info
    left_reverse_flag, right_reverse_flag, up_reverse_flag, down_reverse_flag = reverse_info
    left_idx,right_idx,up_idx,down_idx = idx_info

    #uv originate from bottom-left

    left_uv = gen_boundary_uv_for_interp('L',False,face_res)
    right_uv = gen_boundary_uv_for_interp('R',False,face_res)
    up_uv = gen_boundary_uv_for_interp('U',False,face_res)
    down_uv = gen_boundary_uv_for_interp('D',False,face_res)




    # generate face uv
    uv_table = np.zeros((face_extended_res,face_extended_res,2))
    uv_ascending_order = map_util.create_pixel_index(face_res,1)
    uv_ascending_order /= face_res
    epsilon = uv_ascending_order.min()


    # yv for u, xv for v
    xv, yv = np.meshgrid(np.flip(uv_ascending_order),uv_ascending_order,indexing='ij')
    uv_table[1:-1,1:-1,0] = yv
    uv_table[1:-1,1:-1,1] = xv

    uv_table[1:-1,0,:] = left_uv
    uv_table[1:-1,-1,:] = right_uv
    uv_table[0,1:-1,:] = up_uv
    uv_table[-1,1:-1,:] = down_uv

    #corner
    uv_table[0,0,:] = np.array([-epsilon,1+epsilon])
    uv_table[0,-1,:] = np.array([1+epsilon,1+epsilon])
    uv_table[-1,0,:] = np.array([-epsilon,-epsilon])
    uv_table[-1,-1,:] = np.array([1+epsilon,-epsilon])

    # now interpolate uv
    uv_interpolators = []
    for idx in range(2):
        interpolator = scipy.interpolate.RegularGridInterpolator((np.arange(0,face_extended_res),np.arange(0,face_extended_res)),uv_table[:,:,idx])
        uv_interpolators.append(interpolator)

    uv_interpolated = np.zeros((point_xs.shape[0],point_xs.shape[1],2))
    for idx in range(2):
        interpolator = uv_interpolators[idx]
        uv_interpolated[:,:,idx] = interpolator((point_xs,point_ys))

    xyz_interpolated = map_util.uv_to_xyz_vectorized(uv_interpolated, face_idx,normalize_flag=False)

    """
    The Jacobian weight needs to be normalized to ensure the color is not scaled.
    We need to do sth like
    sample[0] * J[0]/(J[0]+J[1]+J[2]+J[3])
    
    So we need to compute a Jacobian sum of 2x2 block here
    
    The final weight for each pixel i would be:
    2*w[i] / (w[0] + w[1] + w[2] + w[3]) + 1/2
    
    Thus when we take the average of four samples, the color would be
    
    1/4 *(
        s[0] * ( 2 * w[0] / (w[0] + w[1] + w[2] + w[3]) + 1/2 ) + 
        s[1] * ( 2 * w[1] / (w[0] + w[1] + w[2] + w[3]) + 1/2 ) + 
        s[2] * ( 2 * w[2] / (w[0] + w[1] + w[2] + w[3]) + 1/2 ) +      
        s[3] * ( 2 * w[3] / (w[0] + w[1] + w[2] + w[3]) + 1/2 ) + 
    )
    = 1/4 * (
        2 * (s[0] * w[0] + s[1] * w[1] + s[2] * w[2] + s[3] * w[3]) / (w[0] + w[1] + w[2] + w[3]) +
        1/2 * (s[0] + s[1] + s[2] + s[3])
    ) 
    = 1/2 * (s[0] * w[0] + s[1] * w[1] + s[2] * w[2] + s[3] * w[3]) / (w[0] + w[1] + w[2] + w[3])
     +1/2 * (s[0] + s[1] + s[2] + s[3])
     
     This means half of the contribution comes from directly taking the average and half the contribution comes from
     the Jacobian weighted average
    """

    jacobian = map_util.jacobian_vertorized(xyz_interpolated)
    jac_inverse = False
    if jac_inverse:
        ####
        inverse_jacobian = 1 / jacobian
        jacobian = inverse_jacobian
        ####
    jacobian_sum = skimage.measure.block_reduce(jacobian,(2,2),np.sum)
    #copy(up sample) jacobian
    jacobian_sum = np.repeat(np.repeat(jacobian_sum,2,axis=0),2,axis=1)

    weight_tuned = 2 * jacobian / jacobian_sum + 0.5

    if channel_count == 3:
        weighted_points = points * np.stack((weight_tuned,weight_tuned,weight_tuned),axis=2)
    else:
        weighted_points = points * weight_tuned.reshape((weight_tuned.shape[0],weight_tuned.shape[1],1))


    # Downsample the weighted points
    # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.block_reduce
    downsampled_image = skimage.measure.block_reduce(weighted_points,(2,2,1),np.mean)


    return downsampled_image



    # #test
    # this_face_left_uv = gen_boundary_uv('L',False,face_res)
    # this_face_left_xyz = map_util.uv_to_xyz_vectorized(this_face_left_uv,face_idx)
    #
    # test_xyz = map_util.uv_to_xyz((-0.01,0.3),face_idx)
    # test_xyz2 = map_util.uv_to_xyz((0.99,0.3),left_idx)
    #
    # print("test")


def downsample_all_faces(faces):
    """

    :param faces: original faces without extending
    :return:
    """

    res = faces.shape[1]
    chan_count = faces.shape[-1]

    #extending face
    faces_extended = np.zeros((6, res + 2, res + 2, chan_count))

    new_faces = np.zeros((6,res>>1,res>>1,chan_count))

    for face_idx in range(6):
        face = extend_face(faces,face_idx)
        faces_extended[face_idx] = face

    for face_idx in range(6):
        new_face = downsample(faces_extended,face_idx)
        new_faces[face_idx] = new_face

    return new_faces



def downsample_full(original_map, n_mipmap_level):
    """
    Takes the original highres cubemap, make a level of downsampled cubemap
    :param original_map:
    :param n_mipmap_level:
    :return:
    """
    mipmaps = []
    high_res = original_map.shape[1]
    #assert high_res == 128
    previous_level_map = original_map
    for i in range(n_mipmap_level):
        if i == 0:
            mipmaps.append(original_map)
        else:
            new_level_map = downsample_all_faces(previous_level_map)
            mipmaps.append(new_level_map)
            previous_level_map = new_level_map

    return mipmaps




#TODO: implement trilinear interpolation for mipmap?
class trilinear_mipmap_interpolator:
    """
    We store a list of length channel_count
    list[0] is a list consist of n_level of bilinear interpolator

    It seems like we need to make a per-face interpolator to account for out-of-bound situation

    So we would have in total
    3 channels
    For each channel, we have 6 faces, n-levels. 6n faces each face with a interpolator
    """
    def __init__(self, mipmap_list:list):
        self.n_level = len(mipmap_list)
        self.channel_count = mipmap_list[0].shape[-1]
        self.bilinear_interpolator_list = []
        self.extended_mipmap_list = []

        #create extended_face mipmap list

        for level_idx in range(self.n_level):
            level_res = mipmap_list[level_idx].shape[1]
            current_extended_faces = np.zeros((6,level_res+2,level_res+2,self.channel_count))
            # Now generate interpolator for this face
            for face_idx in range(6):
                current_extended_face = extend_face(mipmap_list[level_idx], face_idx)
                current_extended_faces[face_idx] = current_extended_face
            self.extended_mipmap_list.append(current_extended_faces)


        self.per_channel_interpolator_list = [[] for _ in range(self.channel_count)]

        # dimension for each interpolator (face_idx, n_res, n_res)
        for level_idx in range(self.n_level):
            level_res = mipmap_list[level_idx].shape[1]
            left_uv = gen_boundary_uv_for_interp('L', False, level_res)
            right_uv = gen_boundary_uv_for_interp('R', False, level_res)
            up_uv = gen_boundary_uv_for_interp('U', False, level_res)
            down_uv = gen_boundary_uv_for_interp('D', False, level_res)

            # generate face uv
            uv_table = np.zeros((level_res + 2, level_res + 2, 2))
            uv_ascending_order = map_util.create_pixel_index(level_res, 1)
            uv_ascending_order /= level_res
            epsilon = uv_ascending_order.min()

            # yv for u, xv for v
            xv, yv = np.meshgrid(np.flip(uv_ascending_order), uv_ascending_order, indexing='ij')
            uv_table[1:-1, 1:-1, 0] = yv
            uv_table[1:-1, 1:-1, 1] = xv

            uv_table[1:-1, 0, :] = left_uv
            uv_table[1:-1, -1, :] = right_uv
            uv_table[0, 1:-1, :] = up_uv
            uv_table[-1, 1:-1, :] = down_uv

            # corner
            uv_table[0, 0, :] = np.array([-epsilon, 1 + epsilon])
            uv_table[0, -1, :] = np.array([1 + epsilon, 1 + epsilon])
            uv_table[-1, 0, :] = np.array([-epsilon, -epsilon])
            uv_table[-1, -1, :] = np.array([1 + epsilon, -epsilon])

            for chan_idx in range(self.channel_count):
                # This interpolator initialization should be in numpy order, not uv order
                #TODO: check this is correct
                current_level_chan_interpolator = RegularGridInterpolator((np.arange(0,6),uv_table[:,0,1],uv_table[0,:,0]),self.extended_mipmap_list[level_idx][...,chan_idx])
                self.per_channel_interpolator_list[chan_idx].append(current_level_chan_interpolator)


    def clamp_level(self,level):
        """

        :param level:[N,]
        :return:
        """
        max_level = self.n_level - 1.0
        min_level = 0.0

        return np.clip(level,min_level,max_level)




    def interpolate_all(self,xyz,level):
        """

        :param xyz: in shape of [N,3]
        :param level: in the shape of (xyz.shape[:-1]), incicating the mipmap level
        :return:
        """
        original_shape = xyz.shape
        xyz = xyz.reshape((-1,3))
        level = level.flatten()

        level = self.clamp_level(level)


        import map_util
        u,v,face_idx = map_util.xyz_to_uv_vectorized(xyz)

        per_channel_result = [[] for _ in range(self.channel_count)]


        for chan_idx in range(self.channel_count):
            for level_idx in range(self.n_level):
                chan_lev_result = self.per_channel_interpolator_list[chan_idx][level_idx]((face_idx,v,u))
                per_channel_result[chan_idx].append(chan_lev_result)

        per_channel_array = []
        # Now we have n_level of interpolation result
        # Try to convert it to numpy array?
        for chan_idx in range(self.channel_count):
            tmp = per_channel_result[chan_idx]
            tmp = np.array(tmp)
            per_channel_array.append(tmp)

        # Now per_channel_array[i] contains n-levels of interpolated result for all directions

        high_res_level = np.floor(level).astype(int)
        low_res_level = np.ceil(level).astype(int)


        # Do trilinear interpolation

        per_channel_high_res_result = []
        per_channel_low_res_result = []


        tmp = np.arange(0,xyz.shape[0])


        for chan_idx in range(self.channel_count):
            per_channel_high_res_result.append(per_channel_array[chan_idx][high_res_level,tmp])
            per_channel_low_res_result.append(per_channel_array[chan_idx][low_res_level,tmp])


        lerp_t = level - high_res_level

        final_color = []

        for chan_idx in range(self.channel_count):

            col_this_channel = lerp(per_channel_high_res_result[chan_idx], per_channel_low_res_result[chan_idx], lerp_t)
            final_color.append(col_this_channel)


        #reshape

        final_color = np.array(final_color).T

        final_color = final_color.reshape((original_shape[:-1] + (3,)))

        print("Done")

        return final_color



def lerp(array_1,array_2, t):
    """
    lerp used to compute the final trilinear step

    compute array_1 * (1-t) + array_2 * t

    :param array_1: [N]
    :param array_2: [N]
    :param t: [N]
    :return:
    """
    tmp1 = array_1 * (1.0 - t)
    tmp2 = array_2 * t

    return tmp1 + tmp2



if __name__ == '__main__':
    recurrence()