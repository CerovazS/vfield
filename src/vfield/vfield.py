import matplotlib.pyplot as plt
import numpy as np
import lic
from scipy.interpolate import griddata
from PIL import Image, ImageDraw
from scipy.interpolate import griddata


def plot_vector_field(grid, displacement):
    """Plot a vector field using matplotlib."""

    x = grid[:, 0].numpy()
    y = grid[:, 1].numpy()
    u = displacement[:, 0].numpy()
    v = displacement[:, 1].numpy()

    # color the vector field based on the magnitude of the displacement
    magnitude = np.sqrt(u**2 + v**2)
    plt.quiver(x, y, u, v, magnitude, cmap='viridis',
            angles='xy', scale_units='xy', scale=1, alpha=0.5)

    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')


def render_lic(grid, displacement, resolution=1024, length=30, normalize=True) -> np.ndarray:
    """
    Create a Line Integral Convolution (LIC) visualization of a vector field.

    Args:
        grid: tensor of shape (N, 2) representing the grid points
        displacement: tensor of shape (N, 2) representing the vector field at each grid point
        resolution: resolution of the output image (default: 1024)
        length: length of the LIC lines (default: 30)
        normalize: whether to normalize the vector field (default: True)
    
    Returns:
        lic_img: numpy array of shape (resolution, resolution) representing the LIC image
    """
    grid_np = grid.numpy()
    displacement_np = displacement.numpy()

    a = grid_np[:, 0].min()
    b = grid_np[:, 0].max()
    c = grid_np[:, 1].min()
    d = grid_np[:, 1].max()

    # create high-resolution grid
    x_lic, y_lic = np.mgrid[a:b:resolution*1j, c:d:resolution*1j]
    
    # analytic test field: a clockwise vortex centred at (0,0)
    # u = -y_lic
    # v = x_lic

    # create points for interpolation (flatten the high-res grid)
    points_lic = np.column_stack((x_lic.ravel(), y_lic.ravel()))
    
    # interpolate displacement components
    u_interp = griddata(grid_np, displacement_np[:, 0], points_lic, 
                        method='linear', fill_value=0.0)
    v_interp = griddata(grid_np, displacement_np[:, 1], points_lic, 
                        method='linear', fill_value=0.0)
    
    # reshape back to grid
    u = u_interp.reshape(resolution, resolution)
    v = v_interp.reshape(resolution, resolution)

    if normalize:
        mag = np.hypot(u, v)
        u = u / (mag + 1e-9)
        v = v / (mag + 1e-9)

    # render the LIC
    seed = lic.gen_seed((resolution, resolution), noise="white")
    lic_img = lic.lic(u, v, seed, length=length)

    lic_img = np.rot90(lic_img, k=3)  # rotate 90 degrees ccw
    lic_img = np.fliplr(lic_img)      # flip left-right
    
    return lic_img


def render_flow_field(grid, displacement, W=1000, H=1000, particles=2_000, steps=100, step_size=0.002, bg_color=0) -> Image:
    """
    Create flow field visualization using explicit particle integration.
    
    Args:
        W, H: canvas size
        particles: number of particles
        steps: integration steps per particle
        step_size: step size as fraction of image width
        bg_color: background color for the image (default: 0)

    Returns:
        img: PIL Image object with the flow field visualization
    """
    # Pre-compute interpolation on a dense grid for speed
    grid_np = grid.numpy()
    displacement_np = displacement.numpy()
    
    min_x = grid_np[:, 0].min()
    max_x = grid_np[:, 0].max()
    min_y = grid_np[:, 1].min()
    max_y = grid_np[:, 1].max()
    
    # Create dense interpolation grid
    interp_res = 512  # Resolution for interpolation grid
    x_interp = np.linspace(min_x, max_x, interp_res)
    y_interp = np.linspace(min_y, max_y, interp_res)
    X_interp, Y_interp = np.meshgrid(x_interp, y_interp, indexing='xy')
    points_interp = np.column_stack((X_interp.ravel(), Y_interp.ravel()))
    
    # Pre-compute interpolated values
    u_grid = griddata(grid_np, displacement_np[:, 0], points_interp, 
                        method='linear', fill_value=0.0).reshape(interp_res, interp_res)
    v_grid = griddata(grid_np, displacement_np[:, 1], points_interp, 
                        method='linear', fill_value=0.0).reshape(interp_res, interp_res)
    
    # analytic vortex field
    # def F(px, py):
    #     # map pixel → spatial coords
    #     x = (px/W) * (max_x - min_x) + min_x
    #     y = (py/H) * (max_y - min_y) + min_y
    #     return -y, x
    
    def F(px, py):
        """Fast lookup from pre-computed interpolation grid"""
        # map pixel → spatial coords
        x = (px/W) * (max_x - min_x) + min_x
        y = (py/H) * (max_y - min_y) + min_y
        
        # Convert to grid indices
        i = int((y - min_y) / (max_y - min_y) * (interp_res - 1))
        j = int((x - min_x) / (max_x - min_x) * (interp_res - 1))
        
        # Bounds check
        if 0 <= i < interp_res and 0 <= j < interp_res:
            return u_grid[i, j], v_grid[i, j]
        else:
            return 0.0, 0.0
        
    # ------------- integrate --------------
    img = Image.new("L", (W, H), bg_color)
    drw = ImageDraw.Draw(img, "L")
    
    rng = np.random.default_rng(0)
    starts_x = rng.uniform(0, W, particles)
    starts_y = rng.uniform(0, H, particles)
    
    for x0, y0 in zip(starts_x, starts_y):
        x, y = x0, y0
        pts   = []
        for _ in range(steps):
            u, v = F(x, y)
            norm = (u*u + v*v)**0.5 + 1e-6
            x   += (u / norm) * step_size * W
            y   += (v / norm) * step_size * W
            if not (0 <= x < W and 0 <= y < H):
                break
            pts.append((x, y))
        if len(pts) > 1:
            drw.line(pts, fill=255-1, width=1)   # low-alpha, additive

    return img
