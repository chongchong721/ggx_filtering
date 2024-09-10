from operator import index

import numpy as np
import scipy
import map_util



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
    tj = location[1]
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


def gen_boundary_uv(edge_side,reverse_flag,cubemap_res,non_zero=True):

    if non_zero:
        OneMinusEpsilon = 0.999999940395355225
        Epsilon = 1.0 - OneMinusEpsilon
    else:
        OneMinusEpsilon = 1.0
        Epsilon = 0.0
    uv = np.zeros((cubemap_res,2))
    uv_ascending_order = np.linspace(0, 1.0, cubemap_res, endpoint=True)
    uv_ascending_order[0] = Epsilon
    uv_ascending_order[-1] = OneMinusEpsilon

    if edge_side == "L":
        # u is 0 while v goes from 1 to 0
        uv[:,0] = np.full(cubemap_res,Epsilon)
        uv[:,1] = np.flip(uv_ascending_order)
    elif edge_side == "R":
        # u is 1 while v gose from 1 to 0
        uv[:,0] = np.full(cubemap_res,OneMinusEpsilon)
        uv[:,1] = np.flip(uv_ascending_order)
    elif edge_side == "U":
        # v is 1 while u goes from 0 to 1
        uv[:,0] = uv_ascending_order
        uv[:,1] = np.full(cubemap_res,OneMinusEpsilon)
    elif edge_side == "D":
        # v is 0 while u goes from 0 to 1
        uv[:,0] = uv_ascending_order
        uv[:,1] = np.full(cubemap_res,Epsilon)
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

    # merge the 4 corners
    #TODO: is this the right way?

    # corners in shape of (3,)
    up_left_corner = (up_bound[0] + left_bound[0]) / 2
    up_right_corner = (up_bound[-1] + right_bound[0]) / 2
    down_left_corner = (down_bound[0] + left_bound[-1]) / 2
    down_right_corner = (down_bound[-1] + right_bound[-1]) / 2

    #TODO: check the above works

    face_res = faces[face_idx].shape[0]

    face_extended = np.zeros((face_res+2,face_res+2,3))

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
    face_extended_res = faces_extended.shape[1]
    face_res = face_extended_res - 2
    face_extended = faces_extended[face_idx]
    # fetch bilinear interpolation samples
    interpolators = []

    row_range = np.arange(0,face_extended_res)
    col_range = np.arange(0,face_extended_res)

    for chan_idx in range(3):
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

    points = np.zeros((point_xs.shape[0],point_xs.shape[1],3))

    for chan_idx in range(3):
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

    uv_table = np.zeros((face_extended_res,face_extended_res,2))
    """
    For out of bound points(the boundary), we probably need to interpolate xyz, but how?
    
    spherical linear interpolation?
    or interpolate uv? -> this seems more reasonable since we are applying Jacobian afterwards?
    
    In the current implementation, the direction of the boundary is almost the same as the second row/col
    since the boundary uses u/v of Epsilon and the original face boundary uses u/v of OneMinusEpsilon. They
    have little difference, so now just set those place's u=0 or v=0 or u=1 or v=1
    
    """
    idx_info, edge_side_info, reverse_info = get_edge_information(face_idx)
    left_edge_side, right_edge_side, up_edge_side, down_edge_side = edge_side_info
    left_reverse_flag, right_reverse_flag, up_reverse_flag, down_reverse_flag = reverse_info
    left_idx,right_idx,up_idx,down_idx = idx_info

    left_uv = gen_boundary_uv(left_edge_side,left_reverse_flag,face_res,False)
    right_uv = gen_boundary_uv(right_edge_side,right_reverse_flag,face_res,False)
    up_uv = gen_boundary_uv(up_edge_side,up_reverse_flag,face_res,False)
    down_uv = gen_boundary_uv(down_edge_side,down_reverse_flag,face_res,False)





    left_xyz = map_util.uv_to_xyz_vectorized(left_uv,left_idx)
    right_xyz = map_util.uv_to_xyz_vectorized(right_uv,right_idx)
    up_xyz = map_util.uv_to_xyz_vectorized(up_uv,up_idx)
    down_xyz = map_util.uv_to_xyz_vectorized(down_uv,down_idx)




    #test
    this_face_left_uv = gen_boundary_uv('L',False,face_res)
    this_face_left_xyz = map_util.uv_to_xyz_vectorized(this_face_left_uv,face_idx)

    test_xyz = map_util.uv_to_xyz((-0.01,0.3),face_idx)
    test_xyz2 = map_util.uv_to_xyz((0.99,0.3),left_idx)

    print("test")


if __name__ == '__main__':
    recurrence()