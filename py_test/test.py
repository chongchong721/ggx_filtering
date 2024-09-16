import map_util
import numpy as np
import image_read
import interpolation

def test_xyz_to_uv():
    
    res = 32
    uv_table = np.zeros((res, res, 2))
    uv_ascending_order = map_util.create_pixel_index(res, 1)
    uv_ascending_order /= res

    # yv for u, xv for v
    xv, yv = np.meshgrid(np.flip(uv_ascending_order), uv_ascending_order, indexing='ij')
    uv_table[:, :, 0] = yv
    uv_table[:, :, 1] = xv

    face_idx = 0

    xyz = map_util.uv_to_xyz_vectorized(uv_table,face_idx)
    result = map_util.xyz_to_uv_vectorized(xyz)

    print("Done")


def test_trilerp():
    mipmap_l0 = image_read.envmap_to_cubemap('exr_files/rosendal_plains_2_1k.exr',128)
    mipmaps = interpolation.downsample_full(mipmap_l0,7)

    a = interpolation.trilinear_mipmap_interpolator(mipmaps)

    res = 32
    uv_table = np.zeros((res, res, 2))
    uv_ascending_order = map_util.create_pixel_index(res, 1)
    uv_ascending_order /= res

    # yv for u, xv for v
    xv, yv = np.meshgrid(np.flip(uv_ascending_order), uv_ascending_order, indexing='ij')
    uv_table[:, :, 0] = yv
    uv_table[:, :, 1] = xv

    face_idx = 0

    xyz = map_util.uv_to_xyz_vectorized(uv_table,face_idx)
    random_level = np.random.random(xyz.shape[:-1]) * 6

    a.interpolate_all(xyz,random_level)




if __name__ == '__main__':
    test_trilerp()
    test_xyz_to_uv()