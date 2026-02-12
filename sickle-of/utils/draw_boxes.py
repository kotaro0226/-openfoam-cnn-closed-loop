import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_wireframe_rectangular_prism(ax, origin, x_size, y_size, z_size, color='black'):
    """Draws a wireframe rectangular prism (box) given an origin and dimensions."""
    x, y, z = origin

    # Define the 8 vertices
    vertices = np.array([
        [x,         y,         z        ],
        [x + x_size, y,         z        ],
        [x + x_size, y + y_size, z       ],
        [x,         y + y_size, z        ],
        [x,         y,         z + z_size],
        [x + x_size, y,         z + z_size],
        [x + x_size, y + y_size, z + z_size],
        [x,         y + y_size, z + z_size]
    ])

    # Define the 12 edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    # Draw each edge
    for edge in edges:
        ax.plot3D(*zip(*vertices[list(edge)]), color=color)

def draw_wireframe_cube(ax, origin, size, color='black'):
    """Draws a wireframe cube (all edges = size)."""
    draw_wireframe_rectangular_prism(ax, origin, size, size, size, color=color)

# ----------------------------------------------------------------------------
# 1) Define your domain so that the x-dimension is 256 (left-to-right),
#    and the y,z dimensions are both 512.
# ----------------------------------------------------------------------------
LARGE_X, LARGE_Y, LARGE_Z = 256, 512, 512  # Now x=256, y=512, z=512
SMALL_CUBE_SIZE = 32                       # Subcubes: 32×32×32

# ----------------------------------------------------------------------------
# 2) Scaling for plotting
# ----------------------------------------------------------------------------
# We'll scale so the largest real dimension (512) becomes ~10 in the final plot
scale_factor = 10.0 / max(LARGE_X, LARGE_Y, LARGE_Z)  # 10 / 512 = 0.01953...
norm_x = LARGE_X * scale_factor  # ~5
norm_y = LARGE_Y * scale_factor  # ~10
norm_z = LARGE_Z * scale_factor  # ~10
norm_small = SMALL_CUBE_SIZE * scale_factor

# ----------------------------------------------------------------------------
# 3) Generate grid positions for all possible subcubes
# ----------------------------------------------------------------------------
nx = LARGE_X // SMALL_CUBE_SIZE  # 256/32 = 8
ny = LARGE_Y // SMALL_CUBE_SIZE  # 512/32 = 16
nz = LARGE_Z // SMALL_CUBE_SIZE  # 512/32 = 16

all_positions = []
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            real_x = i * SMALL_CUBE_SIZE
            real_y = j * SMALL_CUBE_SIZE
            real_z = k * SMALL_CUBE_SIZE

            # Scale them for plotting
            sx = real_x * scale_factor
            sy = real_y * scale_factor
            sz = real_z * scale_factor
            all_positions.append((sx, sy, sz))

# ----------------------------------------------------------------------------
# 4) Randomly select 128 cubes to show
# ----------------------------------------------------------------------------
random.seed(42)  # For reproducibility
selected_positions = random.sample(all_positions, 64)

# ----------------------------------------------------------------------------
# 5) Plot everything
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Optionally remove ticks and grids for clarity
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Draw the large rectangular box with x=256, y=512, z=512 (scaled)
draw_wireframe_rectangular_prism(
    ax,
    origin=(0, 0, 0),
    x_size=norm_x,
    y_size=norm_y,
    z_size=norm_z
)

# Draw the selected smaller cubes (still 32x32x32, scaled)
for pos in selected_positions:
    draw_wireframe_cube(ax, origin=pos, size=norm_small)

# Keep the aspect ratio true to the scaled dimensions
ax.set_box_aspect([norm_x, norm_y, norm_z])

# Optional: adjust the view so x-axis is indeed left-to-right
ax.view_init(elev=20, azim=-45)
plt.savefig('boxes.png')
