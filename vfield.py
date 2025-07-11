import matplotlib.pyplot as plt
import numpy as np
import lic
from scipy.interpolate import griddata
from PIL import Image, ImageDraw
from scipy.interpolate import griddata


def plot_vector_field(grid, displacement):
    x = grid[:, 0].numpy()
    y = grid[:, 1].numpy()
    u = displacement[:, 0].numpy()
    v = displacement[:, 1].numpy()

    plt.figure(figsize=(5, 5))

    # color the vector field based on the magnitude of the displacement
    magnitude = np.sqrt(u**2 + v**2)
    plt.quiver(x, y, u, v, magnitude, cmap='viridis',
               angles='xy', scale_units='xy', scale=1, alpha=0.5)
    
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Vector Field')
    plt.grid()
    plt.show()
    

def render_lic(grid, displacement, resolution=1024, length=30, normalize=True):

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
    
    # display the LIC and vector field side by side

    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)

    plt.imshow(lic_img, cmap='gray', origin='lower')
    plt.title(f'LIC rendering (resolution={resolution}, length={length}, normalize={normalize})')
    plt.axis('off')
    
    plt.subplot(122)

    skip = max(1, resolution // 40)  # subsample for clarity
    x_sub = x_lic[::skip, ::skip]
    y_sub = y_lic[::skip, ::skip]
    u_sub = u[::skip, ::skip]
    v_sub = v[::skip, ::skip]
    
    plt.quiver(x_sub, y_sub, u_sub, v_sub, alpha=0.7, scale=30)
    plt.xlim(a, b)
    plt.ylim(c, d)
    plt.title(f'Vector Field (subsampled every {skip} points)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return lic_img


def render_flow_field(grid, displacement, W=4000, H=4000, N_PART=10_000, STEPS=400, 
                           STEP_SIZE=0.002, ALPHA=2, BG=255, extent=None):
    """
    Create flow field visualization using explicit particle integration
    
    Args:
        W, H: canvas size
        N_PART: number of particles
        STEPS: integration steps per particle  
        STEP_SIZE: step size as fraction of image width
        ALPHA: opacity per stroke (0-255)
        BG: background color (0-255)
        extent: spatial extent for loaded mode, tuple (min_x, max_x, min_y, max_y)
    """
    
    min_x, max_x, min_y, max_y = extent

    # Pre-compute interpolation on a dense grid for speed
    grid_np = grid.numpy()
    displacement_np = displacement.numpy()
    
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
    img = Image.new("L", (W, H), BG)
    drw = ImageDraw.Draw(img, "L")
    
    rng = np.random.default_rng(0)
    starts_x = rng.uniform(0, W, N_PART)
    starts_y = rng.uniform(0, H, N_PART)
    
    for x0, y0 in zip(starts_x, starts_y):
        x, y = x0, y0
        pts   = []
        for _ in range(STEPS):
            u, v = F(x, y)
            norm = (u*u + v*v)**0.5 + 1e-6
            x   += (u / norm) * STEP_SIZE * W
            y   += (v / norm) * STEP_SIZE * W
            if not (0 <= x < W and 0 <= y < H):
                break
            pts.append((x, y))
        if len(pts) > 1:
            drw.line(pts, fill=BG-ALPHA, width=1)   # low-alpha, additive

    return img
