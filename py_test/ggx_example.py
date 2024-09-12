import image_read
import interpolation
import material
import mat_util
import map_util
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
"""
To test this, we first generate a high-resolution GGX kernel cubemap
then downsample it using the method we implemented
"""


"""
To better visualize this,
we consider wh to be at, face +z(4) u=0 v=1
"""


def create_color_map(face_ndf,face_xyz):
    matplotlib.use('TkAgg')
    x = face_xyz[:,:,:,0].flatten()
    y = face_xyz[:,:,:,1].flatten()
    z = face_xyz[:,:,:,2].flatten()

    value = face_ndf.flatten()

    value_normalized = (value - np.min(value)) / (np.max(value) - np.min(value))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.scatter(x, y, z, c=value_normalized,cmap='magma')

    plt.show()

    print("Done")



def ref_kernel(alpha, wh_direction:np.ndarray):
    GGX = material.GGX(alpha,alpha)

    wh_direction = mat_util.reshape_direction(wh_direction)
    ref_res = 128
    uv_table = np.zeros((ref_res,ref_res,2))
    uv_ascending = np.linspace(0, 1.0, ref_res, endpoint=True)

    # yv for u, xv for v
    xv, yv = np.meshgrid(np.flip(uv_ascending),uv_ascending,indexing='ij')
    uv_table[:,:,0] = yv
    uv_table[:,:,1] = xv

    face_ndf = np.zeros((6,ref_res,ref_res))
    face_xyz = np.zeros((6,ref_res,ref_res,3))

    for face_idx in range(6):
        current_face_xyz = map_util.uv_to_xyz_vectorized(uv_table,face_idx,True)
        cosine_theta = np.dot(current_face_xyz, wh_direction.flatten())
        current_face_ndf = GGX.ndf_isotropic(cosine_theta)
        face_ndf[face_idx] = current_face_ndf
        face_xyz[face_idx] = map_util.uv_to_xyz_vectorized(uv_table,face_idx,False)

    face_ndf = face_ndf.reshape((6,ref_res,ref_res,1))

    return face_ndf, face_xyz



def ref_xyz(res):
    uv_table = np.zeros((res, res, 2))
    uv_ascending = np.linspace(0, 1.0, res, endpoint=True)
    # yv for u, xv for v
    xv, yv = np.meshgrid(np.flip(uv_ascending), uv_ascending, indexing='ij')
    uv_table[:, :, 0] = yv
    uv_table[:, :, 1] = xv

    face_xyz = np.zeros((6, res, res, 3))

    for face_idx in range(6):
        current_face_xyz = map_util.uv_to_xyz_vectorized(uv_table, face_idx, True)
        face_xyz[face_idx] = map_util.uv_to_xyz_vectorized(uv_table, face_idx, False)

    return face_xyz








if __name__ == "__main__":
    wh_direction = np.array([-1,1,1])
    wh_direction = wh_direction / np.linalg.norm(wh_direction)

    face_ndf,face_xyz = ref_kernel(0.7,wh_direction)
    faces_extended = np.zeros((6,130,130,1))
    faces_downsampled = np.zeros((6,64,64,1))


    for i in range(6):
        face_extended = interpolation.extend_face(face_ndf,i)
        faces_extended[i] = face_extended
    for i in range(6):
        face_downsampled = interpolation.downsample(faces_extended,i)
        faces_downsampled[i] = face_downsampled

    face_downsampled_xyz = ref_xyz(64)


    create_color_map(face_ndf,face_xyz)
    create_color_map(faces_downsampled,face_downsampled_xyz)

    print(wh_direction)