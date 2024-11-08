import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

from map_util import texel_directions

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


def draw_cube(ax, size=1):
    # Define the 8 vertices of the cube
    r = size
    vertices = np.array([[-r, -r, -r],
                         [r, -r, -r],
                         [r, r, -r],
                         [-r, r, -r],
                         [-r, -r, r],
                         [r, -r, r],
                         [r, r, r],
                         [-r, r, r]])

    # Define the 12 edges connecting the vertices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    # Plot the edges
    for edge in edges:
        points = vertices[list(edge)]
        ax.plot3D(*zip(*points), color="black")


# All vectors are row vector
def dot(v1, v2):
    result = np.dot(v1.T, v2).item()

    # result = v1[0][0] * v2[0][0] + v1[1][0] * v1[1][0] + v2[1][0] * v2[1][0]

    if result == 0.0:
        result = 0.00001
    return result

def reflect(wi, wm):
    wo = wi - 2 * dot(wi, wm) * wm
    return wo


def reflect_vector(v, n):
    n = np.array(n)
    v = np.array(v)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("Normal vector cannot be zero.")
    n_unit = n / n_norm
    reflection = v - 2 * np.dot(v, n_unit) * n_unit
    return reflection


def plot_vectors_with_cube(normal_vector:np.ndarray):
    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw the cube
    draw_cube(ax, size=1)

    # Original vector
    v = np.array([0, 0, 1])
    ax.quiver(0, 0, 0, v[0], v[1], v[2], color='blue', linewidth=2, label='Original Vector (0,0,1)')

    # Normal vector
    n = normal_vector
    ax.quiver(0, 0, 0, n[0], n[1], n[2], color='green', linewidth=2, label=f'Normal Vector')

    # Reflected vector
    try:
        r = reflect(-v,n)
        ax.quiver(0, 0, 0, r[0], r[1], r[2], color='red', linewidth=2, label='Reflected Vector')
    except ValueError as e:
        print(e)
        return

    # Set the axes limits
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


def ndf_isotropic_vectorized(alpha, cos_theta):
    """
    Only cosine(theta) is needed in isotropic case
    :param cosine theta: computed directly from vector dot product
    :return:
    """
    a_pow_of2 = alpha ** 2
    ndf = a_pow_of2 / (np.pi * np.pow(cos_theta * cos_theta * (a_pow_of2-1) + 1,2))
    ndf = np.where(cos_theta > 0.0, ndf, 0.0)
    return ndf



def plot_cube_with_function(normal_vector):
    """
    Plot a cube with edges from -1 to 1, vectors, and visualize a scalar function on the cube's surface.

    Parameters:
    - normal_vector: The normal vector to use for reflection and the scalar function.
    """
    # Create a new figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define the scalar function f(x, y, z) = dot(n, [x, y, z])
    n = np.array(normal_vector)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("Normal vector cannot be zero.")
    n_unit = n / n_norm

    # Define original vector
    original_v = np.array([0, 0, 1])

    # Define cube faces parametrically
    size = 1
    resolution = 50  # Number of points per axis on each face
    u = np.linspace(-size, size, resolution)
    v_grid = np.linspace(-size, size, resolution)
    U, V = np.meshgrid(u, v_grid)


    dir = texel_directions(resolution)

    h = (dir / np.linalg.norm(dir,axis=-1,keepdims=True) + normal_vector)
    h = h / np.linalg.norm(h,axis=-1,keepdims=True)

    cosine = np.dot(h,normal_vector)

    cosine = np.flip(cosine,axis=2)


    # List of faces with constant coordinate
    # Each face is defined by a constant x, y, or z
    faces = []
    colors = []

    # Front face (z = size)
    Z = size
    X = U
    Y = V
    faces.append((X, Y, Z * np.ones_like(X)))
    colors.append(cosine[4])

    # Back face (z = -size)
    Z = -size
    X = U
    Y = V
    faces.append((X, Y, Z * np.ones_like(X)))
    colors.append(cosine[5])

    # Left face (x = -size)
    X = -size
    Y = U
    Z = V
    faces.append((X * np.ones_like(Y), Y, Z))
    colors.append(cosine[1])

    # Right face (x = size)
    X = size
    Y = U
    Z = V
    faces.append((X * np.ones_like(Y), Y, Z))
    colors.append(cosine[0])

    # Top face (y = size)
    Y = size
    X = U
    Z = V
    faces.append((X, Y * np.ones_like(X), Z))
    colors.append(cosine[2])

    # Bottom face (y = -size)
    Y = -size
    X = U
    Z = V
    faces.append((X, Y * np.ones_like(X), Z))
    colors.append(cosine[3])

    # Find global min and max for normalization
    all_f = np.concatenate([cosine[4].ravel(), cosine[5].ravel(),
                            cosine[1].ravel(), cosine[0].ravel(),
                            cosine[2].ravel(), cosine[3].ravel()])
    f_min = np.min(all_f)
    f_max = np.max(all_f)
    norm = plt.Normalize(vmin=f_min, vmax=f_max)
    cmap = plt.cm.viridis

    # Plot each face
    for i in range(6):
        X, Y, Z = faces[i]
        f = colors[i]
        # Normalize function values for color mapping
        face_colors = cmap(norm(f))
        # Plot the surface
        ax.plot_surface(X, Y, Z, facecolors=face_colors, shade=False, linewidth=0, antialiased=False)

    # Add a colorbar
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(all_f)
    plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='f(x, y, z) = n · [x, y, z]')

    # Plot the vectors
    # Original vector
    ax.quiver(0, 0, 0, original_v[0], original_v[1], original_v[2],
              color='blue', linewidth=2, arrow_length_ratio=0.1, label='Original Vector (0,0,1)')

    # Normal vector
    ax.quiver(0, 0, 0, n_unit[0], n_unit[1], n_unit[2],
              color='green', linewidth=2, arrow_length_ratio=0.1, label=f'Normal Vector {tuple(normal_vector)}')

    # Reflected vector
    try:
        reflected_v = reflect_vector(original_v, normal_vector)
        ax.quiver(0, 0, 0, reflected_v[0], reflected_v[1], reflected_v[2],
                  color='red', linewidth=2, arrow_length_ratio=0.1, label='Reflected Vector')
    except ValueError as e:
        print(e)

    # Set the axes limits
    max_val = size * 2
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    # Set labels
    ax.set_xlabel('X axis', fontsize=12)
    ax.set_ylabel('Y axis', fontsize=12)
    ax.set_zlabel('Z axis', fontsize=12)

    # Create proxy artists for the legend
    from matplotlib.lines import Line2D
    proxy = [Line2D([0], [0], color='blue', lw=2),
             Line2D([0], [0], color='green', lw=2),
             Line2D([0], [0], color='red', lw=2)]
    ax.legend(proxy, ['Original Vector', 'Normal Vector', 'Reflected Vector'])

    # Enhance the view
    ax.view_init(elev=20, azim=30)

    # Show the grid
    ax.grid(True)

    plt.show()





if __name__ == "__main__":
    # Example normal vector
    normal = np.array([0,0,-1])
    normal = normal / np.linalg.norm(normal)
    plot_cube_with_function(normal)
    # Plot the cube and vectors
    plot_vectors_with_cube(normal)