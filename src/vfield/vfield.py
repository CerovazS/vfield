import numpy as np
from scipy.interpolate import griddata
from PIL import Image, ImageDraw


def render_flow_field(grid, displacement, W=1000, H=1000, particles=2_000, steps=100, step_size=0.002, bg_color=0, antialias=False, aa_factor=2) -> Image:
    """
    Create flow field visualization using explicit particle integration.
    
    Args:
        W, H: canvas size
        particles: number of particles
        steps: integration steps per particle
        step_size: step size as fraction of image width
        bg_color: background color for the image (default: 0)
        antialias: enable supersampling antialiasing (default: False)
        aa_factor: antialiasing factor for supersampling (default: 2)

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
    if antialias:
        return _render_supersampled(F, W, H, particles, steps, step_size, bg_color, aa_factor)
    else:
        # Original rendering method
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
                drw.line(pts, fill=255-bg_color, width=1)   # low-alpha, additive

        return img


def _render_supersampled(F, W, H, particles, steps, step_size, bg_color, aa_factor):
    """
    Render at higher resolution and downsample for antialiasing.
    """
    # Render at higher resolution
    high_W, high_H = W * aa_factor, H * aa_factor
    img = Image.new("L", (high_W, high_H), bg_color)
    drw = ImageDraw.Draw(img, "L")
    
    rng = np.random.default_rng(0)
    starts_x = rng.uniform(0, high_W, particles)
    starts_y = rng.uniform(0, high_H, particles)
    
    for x0, y0 in zip(starts_x, starts_y):
        x, y = x0, y0
        pts = []
        for _ in range(steps):
            # Scale coordinates for the flow field function
            u, v = F(x / aa_factor, y / aa_factor)
            norm = (u*u + v*v)**0.5 + 1e-6
            x += (u / norm) * step_size * high_W
            y += (v / norm) * step_size * high_H
            if not (0 <= x < high_W and 0 <= y < high_H):
                break
            pts.append((x, y))
        if len(pts) > 1:
            drw.line(pts, fill=255-bg_color, width=aa_factor)
    
    # Downsample using high-quality resampling
    return img.resize((W, H), Image.LANCZOS)
