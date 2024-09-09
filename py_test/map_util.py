import numpy as np


"""
currently long lat assume +Y is up

which means tho = ar
"""

def xyz_to_latlon(xyz:np.ndarray):
    """
    :param xyz: normalized vector xyz
    :return:
    """

    x,y,z = xyz[0],xyz[1],xyz[2]

    theta = np.arccos(y)


    phi = np.arctan2(z,x)

    # move phi(longitude to positive)
    # if phi < 0.0:
    #     phi += 2.0*np.pi

    return theta,phi


def xyz_to_latlon_vectorized(xyz:np.ndarray):
    """

    :param xyz: vector in shape of (N,3)
    :return: theta,phi pair, each a vector of (N,)
    """
    x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]
    theta = np.arccos(y)
    phi = np.arctan2(z,x)

    return theta,phi


def latlon_to_xyz(lat,lon):
    """
    :param lon: longitude
    :param lat: latitude
    :return: normalized direction vector
    """
    y = np.cos(lat)
    x = np.sin(lat)*np.cos(lon)
    z = np.sin(lat)*np.sin(lon)

    return np.array([x,y,z])


def xyz_to_uv(xyz:np.ndarray):
    """
    :param xyz: normalized vector xyz
    :return: (u,v),face index
    """
    x,y,z = xyz[0],xyz[1],xyz[2]
    abs_x = np.abs(x)
    abs_y = np.abs(y)
    abs_z = np.abs(z)

    is_x_positive = True if x > 0 else False
    is_y_positive = True if y > 0 else False
    is_z_positive = True if z > 0 else False


    if is_x_positive and abs_x >= abs_y and abs_x >= abs_z:
        # positive x face
        max_axis = abs_x
        uc = -z
        vc = y
        index = 0
    elif not is_x_positive and abs_x >= abs_y and abs_x >= abs_z:
        # negative x face
        max_axis = abs_x
        uc = z
        vc = y
        index = 1
    elif is_y_positive and abs_y >= abs_z and abs_y >= abs_x:
        # positive y face
        max_axis = abs_y
        uc = x
        vc = -z
        index = 2
    elif not is_y_positive and abs_y >= abs_z and abs_y >= abs_x:
        # negative y face
        max_axis = abs_y
        uc = x
        vc = z
        index = 3
    elif is_z_positive and abs_z >= abs_y and abs_z >= abs_x:
        # positive z face
        max_axis = abs_z
        uc = x
        vc = y
        index = 4
    elif not is_z_positive and abs_z >= abs_y and abs_z >= abs_x:
        # negative z face
        max_axis = abs_z
        uc = -x
        vc = y
        index = 5
    else:
        raise NotImplementedError

    # convert range to [0,1]
    u = 0.5 * (uc / max_axis + 1.0)
    v = 0.5 * (vc / max_axis + 1.0)

    return (u,v),index


def uv_to_xyz_vectorized(uv,idx):
    """

    :param uv: array in shape of (N,2)
    :param idx:
    :return: vectorized direction in shape of (N,3)
    """
    u = uv[:,0]
    v = uv[:,1]
    uc = 2.0 * u - 1.0
    vc = 2.0 * v - 1.0

    if idx == 0:
        x = 1.0
        y = vc
        z = -uc
    elif idx == 1:
        x = -1.0
        y = vc
        z = uc
    elif idx == 2:
        x = uc
        y = 1.0
        z = -vc
    elif idx == 3:
        x = uc
        y = -1.0
        z = vc
    elif idx == 4:
        x = uc
        y = vc
        z = 1.0
    elif idx == 5:
        x = -uc
        y = vc
        z = -1.0
    else:
        raise NotImplementedError

    # normalize xyz
    vec = np.zeros((uv.shape[0],3))
    vec[:,0] = x
    vec[:,1] = y
    vec[:,2] = z

    norm = np.linalg.norm(vec,axis=1)
    norm = np.tile(norm, (3, 1)).T
    vec = vec / norm

    return vec


def uv_to_xyz(uv, idx):
    """

    :param uv: u,v coordinate
    :param idx: face index
    :return: normalized 3D vector
    """
    u,v = uv
    uc = 2.0 * u - 1.0
    vc = 2.0 * v - 1.0

    if idx == 0:
        x = 1.0
        y = vc
        z = -uc
    elif idx == 1:
        x = -1.0
        y = vc
        z = uc
    elif idx == 2:
        x = uc
        y = 1.0
        z = -vc
    elif idx == 3:
        x = uc
        y = -1.0
        z = vc
    elif idx == 4:
        x = uc
        y = vc
        z = 1.0
    elif idx == 5:
        x = -uc
        y = vc
        z = -1.0
    else:
        raise NotImplementedError

    # normalize xyz
    vec = np.array([x,y,z])
    vec = vec / np.linalg.norm(vec)

    return vec