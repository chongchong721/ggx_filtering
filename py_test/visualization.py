import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def normalize_minmax(array):
    """
    Normalize a NumPy array to the range [0, 1].
    """
    min_val = array.min()
    max_val = array.max()
    # To avoid division by zero
    if max_val - min_val == 0:
        return np.zeros_like(array)
    return (array - min_val) / (max_val - min_val)

def joint_normalize(array, min_val, max_val):
    """
    Normalize a NumPy array to the range [0, 1] based on provided min and max values.
    """
    # Avoid division by zero
    if max_val - min_val == 0:
        return np.zeros_like(array)
    return (array - min_val) / (max_val - min_val)


def plot_single_cubemap(fig, parent_gs, data, image_type, face_to_position):
    """
    Plots a single cubemap within a parent GridSpec.

    Parameters:
    - fig: matplotlib figure
    - parent_gs: GridSpec for the cubemap
    - data: dict with 'images', 'vmin', 'vmax', 'cmap'
    - image_type: str, title of the cubemap
    - face_to_position: dict, mapping face names to grid positions
    """
    # Iterate over each face and plot
    for face, pos in face_to_position.items():
        row, col = pos
        # Add subplot for each face
        ax = fig.add_subplot(parent_gs[row, col])
        # Get image data
        img = data['images'][face]
        # Display image
        im = ax.imshow(img, cmap=data['cmap'], vmin=data['vmin'], vmax=data['vmax'])
        # Remove axis for clarity
        ax.axis('off')
        # Add face label
        #ax.set_title(face, fontsize=8, pad=5)

    # Calculate overall bounding box of the cubemap
    # positions = parent_gs.get_grid_positions(fig)
    # x0 = min([pos.x0 for pos in positions])
    # x1 = max([pos.x1 for pos in positions])
    # y0 = min([pos.y0 for pos in positions])
    # y1 = max([pos.y1 for pos in positions])
    #
    # # Add colorbar to the right of the cubemap
    # cbar_ax = fig.add_axes([x1 + 0.005, y0, 0.01, y1 - y0])
    # fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    #
    # # Add title above the cubemap
    # fig.text(x0 + (x1 - x0)/2, y1 + 0.02, image_type, ha='center', fontsize=16)




def cubemap_to_2d_array(cubemap):
    """
    array of 3*N,4*N, locations without map is set to zero
    :param cubemap:
    :return:
    """
    chan_count = cubemap.shape[-1]
    cubemap_res = cubemap.shape[1]
    arr = np.zeros((3*cubemap_res, 4*cubemap_res, chan_count))
    # positive x face
    arr[cubemap_res:cubemap_res * 2, cubemap_res * 2:cubemap_res * 3, :] = cubemap[0]

    # negative x face
    arr[cubemap_res:cubemap_res * 2, 0:cubemap_res, :] = cubemap[1]

    # positive y face
    arr[0:cubemap_res, cubemap_res:cubemap_res * 2, :] = cubemap[2]

    # negative y face
    arr[cubemap_res * 2:cubemap_res * 3, cubemap_res:cubemap_res * 2, :] = cubemap[3]

    # positive z face
    arr[cubemap_res:cubemap_res * 2, cubemap_res:cubemap_res * 2, :] = cubemap[4]

    # negative z face
    arr[cubemap_res:cubemap_res * 2, cubemap_res * 3:cubemap_res * 4, :] = cubemap[5]

    return arr


def create_alpha_array(cubemap_res,chan_count):
    """

    :return:
    """
    alpha = np.zeros((3*cubemap_res,4*cubemap_res))
    alpha[cubemap_res:cubemap_res * 2, cubemap_res * 2:cubemap_res * 3] = 1
    # negative x face
    alpha[cubemap_res:cubemap_res * 2, 0:cubemap_res] = 1
    # positive y face
    alpha[0:cubemap_res, cubemap_res:cubemap_res * 2] = 1
    # negative y face
    alpha[cubemap_res * 2:cubemap_res * 3, cubemap_res:cubemap_res * 2] = 1
    # positive z face
    alpha[cubemap_res:cubemap_res * 2, cubemap_res:cubemap_res * 2] = 1
    # negative z face
    alpha[cubemap_res:cubemap_res * 2, cubemap_res * 3:cubemap_res * 4] = 1

    return alpha


def visualize_optim_result(ref:torch.Tensor,result:torch.Tensor):
    """
    create three image that shows ref, diff, pushed_back_img

    ref and result should be in the same scale

    :param ref:
    :param result:
    :return:
    """
    ref = ref.detach().numpy()
    result = result.detach().numpy()
    difference = ref - result

    alpha_arr = create_alpha_array(result.shape[1],result.shape[-1])

    # Find global min and max across both reference and result
    joint_min = min(ref.min(), result.min())
    joint_max = max(ref.max(), result.max())

    # For difference, determine symmetric min and max based on absolute values
    diff_max_abs = max(abs(difference.min()), difference.max())

    # Normalize difference to range [-1, 1]
    difference_norm = difference / diff_max_abs



    difference_2d = cubemap_to_2d_array(difference_norm)
    ref_2d = cubemap_to_2d_array(ref)
    result_2d = cubemap_to_2d_array(result)







    image_types = ['Reference', 'Difference', 'Result']
    image_data = {
        'Reference': {
            'images': ref_2d,
            'vmin': joint_min,
            'vmax': joint_max,
            'cmap': 'viridis',
        },
        'Difference': {
            'images': difference_2d,
            'vmin': -diff_max_abs,
            'vmax': diff_max_abs,
            'cmap': 'seismic',
        },
        'Result': {
            'images': result_2d,
            'vmin': joint_min,
            'vmax': joint_max,
            'cmap': 'viridis',
        }
    }

    fig = plt.figure(figsize=(18,6))
    main_gs = GridSpec(1, 3, figure=fig, wspace=0.3, hspace=0.1)

    # Iterate over image types and plot their cubemaps
    for idx, image_type in enumerate(image_types):
        data = image_data[image_type]

        # Add subplot for each face
        ax = fig.add_subplot(main_gs[0,idx])
        # Get image data
        img = data['images']
        # Display image
        im = ax.imshow(img, cmap=data['cmap'], vmin=data['vmin'], vmax=data['vmax'], alpha=alpha_arr)
        # Remove axis for clarity
        ax.axis('off')

    # Add an overall title
    fig.suptitle('Cubemap Visualization - Reference, Difference, and Result', fontsize=20)

    # Display the plot
    plt.show()



