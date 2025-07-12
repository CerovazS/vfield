import torch
import torch.nn.functional as torch_F
from PIL import Image, ImageDraw
from tqdm import tqdm


def render_flow_field(grid, displacement, W=1000, H=1000, particles=2_000, steps=100, step_size=0.002, bg_color=0, antialias=False, aa_factor=2) -> Image.Image:
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
    # Keep input tensors as tensors (no conversion to numpy)
    min_x = grid[:, 0].min().item()
    max_x = grid[:, 0].max().item()
    min_y = grid[:, 1].min().item()
    max_y = grid[:, 1].max().item()
    
    # Create dense interpolation grid using torch
    interp_res = 512  # Resolution for interpolation grid
    x_interp = torch.linspace(min_x, max_x, interp_res, device=grid.device)
    y_interp = torch.linspace(min_y, max_y, interp_res, device=grid.device)
    X_interp, Y_interp = torch.meshgrid(x_interp, y_interp, indexing='xy')
    points_interp = torch.stack([X_interp.flatten(), Y_interp.flatten()], dim=1)

    # Pre-compute interpolated values using torch interpolation
    # Create a regular grid interpolation function
    u_grid = _interpolate_grid(grid, displacement[:, 0], points_interp, interp_res)
    v_grid = _interpolate_grid(grid, displacement[:, 1], points_interp, interp_res)
    
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
            return u_grid[i, j].item(), v_grid[i, j].item()
        else:
            return 0.0, 0.0
        
    # ------------- integrate --------------
    if antialias:
        return _render_supersampled(F, W, H, particles, steps, step_size, bg_color, aa_factor, grid.device)
    else:
        # Original rendering method
        img = Image.new("L", (W, H), bg_color)
        drw = ImageDraw.Draw(img, "L")
        
        # Use torch random generation
        torch.manual_seed(0)  # For reproducibility
        starts_x = torch.rand(particles, device=grid.device) * W
        starts_y = torch.rand(particles, device=grid.device) * H
        
        for x0, y0 in zip(starts_x.cpu().numpy(), starts_y.cpu().numpy()):
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


def _render_supersampled(F, W, H, particles, steps, step_size, bg_color, aa_factor, device):
    """
    Render at higher resolution and downsample for antialiasing.
    """
    # Render at higher resolution
    high_W, high_H = W * aa_factor, H * aa_factor
    img = Image.new("L", (high_W, high_H), bg_color)
    drw = ImageDraw.Draw(img, "L")
    
    # Use torch random generation
    torch.manual_seed(0)  # For reproducibility
    starts_x = torch.rand(particles, device=device) * high_W
    starts_y = torch.rand(particles, device=device) * high_H

    for x0, y0 in tqdm(zip(starts_x, starts_y)):
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
    return img.resize((W, H), Image.Resampling.LANCZOS)


def _interpolate_grid(grid_points, values, query_points, interp_res):
    """
    Ultra-fast interpolation assuming grid_points form a regular grid.
    """
    device = grid_points.device
    
    # Determine original grid size (assuming it's roughly square)
    n_points = len(grid_points)
    orig_res = int(torch.sqrt(torch.tensor(n_points, dtype=torch.float)).item())
    
    # If the original data forms a regular grid, reshape it
    if orig_res * orig_res == n_points:
        # Perfect square - reshape directly
        values_2d = values.view(orig_res, orig_res)
    else:
        # Find closest square size
        orig_res = int(torch.sqrt(torch.tensor(n_points, dtype=torch.float)).ceil().item())
        # Pad if needed and reshape
        values_padded = torch.cat([values, torch.zeros(orig_res*orig_res - n_points, device=device)])
        values_2d = values_padded.view(orig_res, orig_res)
    
    # Use PyTorch's interpolate function for super fast resampling
    values_4d = values_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Resize to target resolution using bilinear interpolation
    interpolated = torch_F.interpolate(values_4d, size=(interp_res, interp_res), 
                                      mode='bilinear', align_corners=False)
    
    return interpolated.squeeze(0).squeeze(0)  # [interp_res, interp_res]


def render_flow_field_vectorized(grid, displacement, W=1000, H=1000, particles=2_000, steps=100, step_size=0.002, bg_color=0, antialias=False, aa_factor=2) -> Image.Image:
    """
    Vectorized flow field rendering - processes all particles simultaneously.
    Much faster than the loop-based version.
    """
    # Pre-compute interpolation on a dense grid for speed
    min_x = grid[:, 0].min().item()
    max_x = grid[:, 0].max().item()
    min_y = grid[:, 1].min().item()
    max_y = grid[:, 1].max().item()
    
    # Create dense interpolation grid using torch
    interp_res = 512
    x_interp = torch.linspace(min_x, max_x, interp_res, device=grid.device)
    y_interp = torch.linspace(min_y, max_y, interp_res, device=grid.device)
    X_interp, Y_interp = torch.meshgrid(x_interp, y_interp, indexing='xy')
    points_interp = torch.stack([X_interp.flatten(), Y_interp.flatten()], dim=1)
    
    # Pre-compute interpolated values using torch interpolation
    u_grid = _interpolate_grid(grid, displacement[:, 0], points_interp, interp_res)
    v_grid = _interpolate_grid(grid, displacement[:, 1], points_interp, interp_res)
    
    def F_vectorized(px, py):
        """Vectorized lookup from pre-computed interpolation grid"""
        # Convert to spatial coordinates
        x = (px/W) * (max_x - min_x) + min_x
        y = (py/H) * (max_y - min_y) + min_y
        
        # Convert to grid indices (vectorized)
        i = ((y - min_y) / (max_y - min_y) * (interp_res - 1)).long().clamp(0, interp_res-1)
        j = ((x - min_x) / (max_x - min_x) * (interp_res - 1)).long().clamp(0, interp_res-1)
        
        # Vectorized lookup
        u = u_grid[i, j]
        v = v_grid[i, j]
        
        return u, v
    
    # Vectorized particle integration
    if antialias:
        W_render, H_render = W * aa_factor, H * aa_factor
    else:
        W_render, H_render = W, H
    
    # Initialize all particles at once
    torch.manual_seed(0)
    pos_x = torch.rand(particles, device=grid.device) * W_render
    pos_y = torch.rand(particles, device=grid.device) * H_render
    
    # Create accumulator for drawing
    accumulator = torch.zeros(H_render, W_render, device=grid.device)
    
    # Vectorized integration loop
    for step in range(steps):
        # Get velocities for all particles at once
        u, v = F_vectorized(pos_x, pos_y)
        
        # Normalize velocities
        norm = torch.sqrt(u*u + v*v) + 1e-6
        u_norm = u / norm
        v_norm = v / norm
        
        # Update positions for all particles
        new_pos_x = pos_x + u_norm * step_size * W_render
        new_pos_y = pos_y + v_norm * step_size * H_render
        
        # Check bounds and keep only valid particles
        valid = ((new_pos_x >= 0) & (new_pos_x < W_render) & 
                (new_pos_y >= 0) & (new_pos_y < H_render))
        
        if valid.sum() == 0:
            break
            
        # Draw lines from old to new positions (vectorized)
        if step > 0:  # Skip first step
            _draw_lines_vectorized(accumulator, pos_x[valid], pos_y[valid], 
                                 new_pos_x[valid], new_pos_y[valid], 
                                 255 - bg_color)
        
        # Update positions (only for valid particles)
        pos_x = new_pos_x[valid]
        pos_y = new_pos_y[valid]
    
    # Convert accumulator to image
    img_array = accumulator.clamp(0, 255).cpu().numpy().astype('uint8')
    img = Image.fromarray(img_array, mode='L')
    
    # Downsample if antialiasing
    if antialias:
        img = img.resize((W, H), Image.Resampling.LANCZOS)
    
    return img


def _draw_lines_vectorized(accumulator, x1, y1, x2, y2, intensity):
    """
    Vectorized line drawing using simple pixel placement.
    Much faster than individual line drawing.
    """
    H, W = accumulator.shape
    
    # Simple approach: just place pixels at the endpoints
    # For more sophisticated line drawing, we'd need more complex vectorization
    
    # Place pixels at start points
    x1_int = x1.long().clamp(0, W-1)
    y1_int = y1.long().clamp(0, H-1)
    
    # Place pixels at end points  
    x2_int = x2.long().clamp(0, W-1)
    y2_int = y2.long().clamp(0, H-1)
    
    # Add intensity to accumulator (vectorized)
    # This is a simplified approach - for better quality we'd do proper line rasterization
    accumulator[y1_int, x1_int] += intensity * 0.5
    accumulator[y2_int, x2_int] += intensity * 0.5
