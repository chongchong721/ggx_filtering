import os
from stringprep import map_table_b2
import OpenEXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import map_util
import scipy
import Imath
import imageio
import interpolation


OneMinusEpsilon = 0.999999940395355225
Epsilon = 1.0 - OneMinusEpsilon

# OpenCV image x go left-to-right y go top-to-bottom

def hdri_read(exr_file_path):
    img_exr = cv2.imread(exr_file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    #img_exr = cv2.cvtColor(img_exr, cv2.COLOR_RGB2BGR)
    #print(img.shape)
    return img_exr


def write_exr(ldr_img, filename = None):
    """

    :param ldr_img: ldr_img is assumed to be in BGR format
    :param filename:
    :return:
    """
    ldr_img = cv2.cvtColor(ldr_img, cv2.COLOR_BGR2RGB)


    if filename is None:
        imageio.imwrite("imageio_test.exr", ldr_img)
    else:
        imageio.imwrite(filename, ldr_img)




def read_exr_image(file_path):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(file_path)

    # Extract image size
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Define the channel type for EXR
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read RGB channels
    r_channel = np.frombuffer(exr_file.channel('R', FLOAT), dtype=np.float32).reshape(height, width)
    g_channel = np.frombuffer(exr_file.channel('G', FLOAT), dtype=np.float32).reshape(height, width)
    b_channel = np.frombuffer(exr_file.channel('B', FLOAT), dtype=np.float32).reshape(height, width)

    # Stack the channels together to create an image
    exr_image = cv2.merge([b_channel, g_channel, r_channel])

    return exr_image



def grid_interpolator(image):
    """
    We assume pixel[0,:] has a longitude of -pi
              pixel[-1,:] has a longitude of +pi
              pixel[:,0] has a latitude of 0
              pixel[:,-1] has a latitude of +pi


    generate three bilinear interpolators(three channels) class using scipy.RegularGridInterpolator
    :param image:image
    :return:
    """

    # Note, latitude is from [0,np.pi]
    # longitude is from [-np.pi, np.pi]. Need to move to positive
    lat_res,lon_res = get_img_size(image)
    # latitudes = np.arange(0, lat_res)
    # longitudes = np.arange(0, lon_res)


    #test code generate lat-lon in the right way
    latitudes_test = map_util.create_pixel_index(lat_res,1)
    longitudes_test = map_util.create_pixel_index(lon_res,1)

    interpolators = []

    for channel in range(3):
        img_chan = image[:,:,channel]
        interpolator = scipy.interpolate.RegularGridInterpolator((latitudes_test,longitudes_test),img_chan,bounds_error=False,fill_value=None)
        interpolators.append(interpolator)

    return interpolators





def get_img_size(img):
    """

    :param img: image array
    :return: height,width
    """
    shape = img.shape

    height = shape[0]
    width = shape[1]

    return height, width



def loop_through_lat_lon(img):
    lat_res, lon_res = get_img_size(img)

    latitudes = np.linspace(0,np.pi, lat_res)
    longitudes = np.linspace(0, np.pi * 2, lon_res)

    for lat_idx in range(lat_res):
        for lon_idx in range(lon_res):
            lat = latitudes[lat_idx]
            lon = longitudes[lon_idx]

            xyz = map_util.latlon_to_xyz(lat,lon)
            uv,idx = map_util.xyz_to_uv(xyz)
            u,v = uv




def loop_through_certrain_face(img,cubemap_res, interpolators,face_idx):
    lat_res, lon_res = get_img_size(img)

    face = np.zeros((cubemap_res, cubemap_res, 3))

    cubemap_uv = np.linspace(0, 1.0, cubemap_res, endpoint=True)
    cubemap_uv[0] = Epsilon
    cubemap_uv[-1] = OneMinusEpsilon

    #test code generate uv in the right way?
    cubemap_uv_test = map_util.create_pixel_index(cubemap_res,1)
    cubemap_uv_test /= cubemap_res

    print("Computing face ", face_idx)


    u_idx = np.repeat(np.arange(0,cubemap_res), cubemap_res)
    v_idx = np.tile(np.arange(0,cubemap_res), cubemap_res)
    face_idx_row = (cubemap_res - 1) - v_idx
    face_idx_col = u_idx

    uvs = np.zeros((cubemap_res*cubemap_res,2))

    uvs[:,0] = np.repeat(cubemap_uv_test,cubemap_res)
    uvs[:,1] = np.tile(cubemap_uv_test,cubemap_res)


    xyz = map_util.uv_to_xyz_vectorized(uvs,face_idx,True)
    lat,lon = map_util.xyz_to_latlon_vectorized(xyz)
    lon += np.pi

    # the lat_idx and lon_idx in interpolator always have coordiante of x.5, with the max idx of lat/lon_res - 0.5
    # Thus the index here should be lat / np.pi * lat_res
    lat_idx = lat / (np.pi) * lat_res
    lon_idx = lon / (np.pi * 2) * lon_res

    for chan_idx in range(3):
        interpolator = interpolators[chan_idx]
        face[face_idx_row,face_idx_col,chan_idx] = interpolator((lat_idx,lon_idx))

    return face



def loop_through_cube_face_vectorized(img,cubemap_res,interpolators):
    cubemap = np.zeros((6, cubemap_res, cubemap_res, 3))

    for face_idx in range(6):
        face = loop_through_certrain_face(img,cubemap_res,interpolators,face_idx)
        cubemap[face_idx] = face

    return cubemap


def loop_through_cube_face(img, cubemap_res, interpolators):
    """

    :param img: the lat-lon image
    :param cubemap_res: the resolution of a face of the cubemap
    :return: a new image size of [6,cubemap_res,cubemap_res,3]
    """
    lat_res, lon_res = get_img_size(img)

    cubemap = np.zeros((6,cubemap_res,cubemap_res,3))

    cubemap_uv = np.linspace(0,1.0,cubemap_res,endpoint=True)
    cubemap_uv[0] = Epsilon
    cubemap_uv[-1] = OneMinusEpsilon
    for face_idx in range(6):
        print("Computing face ", face_idx)
        face = cubemap[face_idx]
        for u_idx in range(cubemap_res):
            u = cubemap_uv[u_idx]
            for v_idx in range(cubemap_res):
                v = cubemap_uv[v_idx]

                xyz = map_util.uv_to_xyz((u,v),face_idx)
                #print("direction{:.2},{:.2},{:.2}".format(xyz[0],xyz[1],xyz[2]))

                lat,lon = map_util.xyz_to_latlon(xyz)

                # since longitude (arctan2) is in [-np.pi , np.pi ], we need to + np.pi
                lon += np.pi


                # res-1 is to prevent out of bound sample since we assume pixel 0 and pixel -1 is the boundary
                lat_idx = lat / (np.pi) * (lat_res-1)
                lon_idx = lon / (np.pi * 2) * (lon_res-1)


                #u starts from left to right
                #v starts from bottom to up
                #need to convert it to ij index

                face_idx_row = (cubemap_res-1) - v_idx
                face_idx_col = u_idx


                for chan_idx in range(3):
                    interpolator = interpolators[chan_idx]
                    face[face_idx_row,face_idx_col,chan_idx] = interpolator((lat_idx,lon_idx))


    return cubemap


def gen_cubemap_preview_image(cubemap,cubemap_res, preview_path = None,filename = None):

    #https://en.wikipedia.org/wiki/Cube_mapping#Memory_addressing
    if preview_path is None:
        cubemap_preview = np.zeros((cubemap_res * 3, cubemap_res * 4, 3))


        # positive x face
        cubemap_preview[cubemap_res:cubemap_res*2,cubemap_res*2:cubemap_res*3,:] = cubemap[0]

        # negative x face
        cubemap_preview[cubemap_res:cubemap_res*2,0:cubemap_res,:] = cubemap[1]

        # positive y face
        cubemap_preview[0:cubemap_res,cubemap_res:cubemap_res*2,:] = cubemap[2]

        # negative y face
        cubemap_preview[cubemap_res*2:cubemap_res*3,cubemap_res:cubemap_res*2,:] = cubemap[3]

        # positive z face
        cubemap_preview[cubemap_res:cubemap_res * 2, cubemap_res:cubemap_res * 2, :] = cubemap[4]

        # negative z face
        cubemap_preview[cubemap_res:cubemap_res * 2, cubemap_res * 3:cubemap_res * 4, :] = cubemap[5]
        cubemap_preview = cubemap_preview.astype(np.float32)
    else:
        cubemap_preview = np.load(preview_path).astype(np.float32)


    write_exr(cubemap_preview,filename)

    # tonemap1 = cv2.createTonemap(gamma = 2.2)
    #
    # tonemapped_img = tonemap1.process(cubemap_preview)
    #
    # result = np.clip(tonemapped_img * 255.0, 0, 255).astype('uint8')
    #
    # cv2.imwrite("test.jpg",result)



def test_tonemap():
    img = hdri_read('exr_files/rosendal_plains_2_1k.exr')
    tonemapDurand = cv2.createTonemap(2.2)
    ldrDurand = tonemapDurand.process(img.copy())

    im2_8bit = np.clip(ldrDurand * 255, 0, 255).astype('uint8')

    new_filename =  "tonemapped_map.jpg"
    cv2.imwrite(new_filename, im2_8bit)
    write_exr(img)



def envmap_to_cubemap(path, cubemap_res):
    envmap_img = hdri_read(path)
    interpolators = grid_interpolator(envmap_img)
    cubemap = loop_through_cube_face_vectorized(envmap_img, cubemap_res, interpolators)
    return cubemap



def replace_tiff():
    folder_path = "./plots/quad_0.062_200_randomdir_ladj_adam_randomdir_negweight"

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path[-3:] == 'exr':
            continue
        new_path = file_path + "_0.exr"
        tmp_img = imageio.imread(file_path,format='TIFF')
        imageio.imwrite(new_path, tmp_img)


def rename():
    folder_path = "./plots/quad_0.062_200_randomdir_ladj_adam_randomdir_negweight"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path[-3:] == 'exr':
            if file_path[-6:-4] == "_0":
                pass
            else:
                new_path = file_path[:-4] + "_1.exr"
                os.rename(file_path, new_path)



if __name__ == '__main__':
    rename()
    #read_exr_image('exr_files/rosendal_plains_2_1k.exr')
    #test_tonemap()
    #gen_cubemap_preview_image(None,512,"tmp.npy")

    face_resolution = 512
    img = hdri_read('exr_files/08-21_Swiss_A.hdr')
    interpolators_list = grid_interpolator(img)
    cubemap_img = loop_through_cube_face_vectorized(img, face_resolution, interpolators_list)
    #gen_cubemap_preview_image(cubemap_img,face_resolution,None,"swiss_lon2pi.exr")
    faces_extended = np.zeros((6,face_resolution+2,face_resolution+2,3))

    for face_idx in range(6):
        face_extended = interpolation.extend_face(cubemap_img,face_idx)
        faces_extended[face_idx] = face_extended


    interpolation.downsample(faces_extended, 0)

