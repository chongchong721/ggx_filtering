import numpy as np


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

    return face_extended





def recurrence():
    pattern = np.array([[1/8],[3/8],[3/8],[1/8]])

    kernel = np.matmul(pattern,pattern.T)


    print(kernel)








if __name__ == '__main__':
    recurrence()