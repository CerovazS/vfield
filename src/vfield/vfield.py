import torch
import torch.nn.functional as torch_F
from PIL import Image
from tqdm import tqdm


def render_flow_field(grid, displacement, W=1000, H=1000, particles=2_000, steps=100, step_size=0.002, bg_color=0, antialias=False, aa_factor=2) -> Image.Image:
    """
    Vectorized flow field rendering - processes all particles simultaneously.
    Much faster than the loop-based version.
    """
    if antialias:
        return _render_supersampled(grid, displacement, W, H, particles, steps, step_size, bg_color, aa_factor)
    else:
        return _render_base(grid, displacement, W, H, particles, steps, step_size, bg_color)


def _interpolate_grid(grid_points, values, interp_res):
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


def _render_base(grid, displacement, W, H, particles, steps, step_size, bg_color):
    """
    Base vectorized rendering without antialiasing.
    """
    # Pre-compute interpolation on a dense grid for speed
    min_x = grid[:, 0].min().item()
    max_x = grid[:, 0].max().item()
    min_y = grid[:, 1].min().item()
    max_y = grid[:, 1].max().item()
    
    # Create dense interpolation grid using torch
    interp_res = 512
    
    # Pre-compute interpolated values using torch interpolation
    u_grid = _interpolate_grid(grid, displacement[:, 0], interp_res)
    v_grid = _interpolate_grid(grid, displacement[:, 1], interp_res)
    
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
    
    # Initialize all particles at once
    torch.manual_seed(0)
    pos_x = torch.rand(particles, device=grid.device) * W
    pos_y = torch.rand(particles, device=grid.device) * H
    
    # Create accumulator for drawing - start with background color
    accumulator = torch.full((H, W), float(bg_color), dtype=torch.float32, device=grid.device)
    
    # Vectorized integration loop
    for step in tqdm(range(steps)):
        # Get velocities for all particles at once
        u, v = F_vectorized(pos_x, pos_y)
        
        # Normalize velocities
        norm = torch.sqrt(u*u + v*v) + 1e-6
        u_norm = u / norm
        v_norm = v / norm
        
        # Update positions for all particles
        new_pos_x = pos_x + u_norm * step_size * W
        new_pos_y = pos_y + v_norm * step_size * H
        
        # Check bounds and keep only valid particles
        valid = ((new_pos_x >= 0) & (new_pos_x < W) & 
                (new_pos_y >= 0) & (new_pos_y < H))
        
        if valid.sum() == 0:
            break
            
        # Draw lines from old to new positions (vectorized)
        if step > 0:  # Skip first step
            _draw_lines(accumulator, pos_x[valid], pos_y[valid], 
                         new_pos_x[valid], new_pos_y[valid], 
                         255 - bg_color)

        # Update positions (only for valid particles)
        pos_x = new_pos_x[valid]
        pos_y = new_pos_y[valid]
    
    # Convert accumulator to image
    img_array = accumulator.clamp(0, 255).cpu().numpy().astype('uint8')
    img = Image.fromarray(img_array, mode='L')
    
    return img


def _render_supersampled(grid, displacement, W, H, particles, steps, step_size, bg_color, aa_factor):
    """
    Vectorized rendering with supersampling antialiasing.
    Renders at higher resolution and downsamples for smooth results.
    """
    # Render at higher resolution
    high_W, high_H = W * aa_factor, H * aa_factor
    
    # Pre-compute interpolation on a dense grid for speed
    min_x = grid[:, 0].min().item()
    max_x = grid[:, 0].max().item()
    min_y = grid[:, 1].min().item()
    max_y = grid[:, 1].max().item()
    
    # Create dense interpolation grid using torch
    interp_res = 512
    
    # Pre-compute interpolated values using torch interpolation
    u_grid = _interpolate_grid(grid, displacement[:, 0], interp_res)
    v_grid = _interpolate_grid(grid, displacement[:, 1], interp_res)
    
    def F_vectorized(px, py):
        """Vectorized lookup from pre-computed interpolation grid"""
        # Convert to spatial coordinates (scale back for the original coordinate system)
        x = (px/high_W) * (max_x - min_x) + min_x
        y = (py/high_H) * (max_y - min_y) + min_y
        
        # Convert to grid indices (vectorized)
        i = ((y - min_y) / (max_y - min_y) * (interp_res - 1)).long().clamp(0, interp_res-1)
        j = ((x - min_x) / (max_x - min_x) * (interp_res - 1)).long().clamp(0, interp_res-1)
        
        # Vectorized lookup
        u = u_grid[i, j]
        v = v_grid[i, j]
        
        return u, v
    
    # Initialize all particles at once at high resolution
    torch.manual_seed(0)
    pos_x = torch.rand(particles, device=grid.device) * high_W
    pos_y = torch.rand(particles, device=grid.device) * high_H
    
    # Create accumulator for drawing at high resolution - start with background color
    accumulator = torch.full((high_H, high_W), float(bg_color), dtype=torch.float32, device=grid.device)
    
    # Vectorized integration loop
    for step in tqdm(range(steps)):
        # Get velocities for all particles at once
        u, v = F_vectorized(pos_x, pos_y)
        
        # Normalize velocities
        norm = torch.sqrt(u*u + v*v) + 1e-6
        u_norm = u / norm
        v_norm = v / norm
        
        # Update positions for all particles (scale step size for high resolution)
        new_pos_x = pos_x + u_norm * step_size * high_W
        new_pos_y = pos_y + v_norm * step_size * high_H
        
        # Check bounds and keep only valid particles
        valid = ((new_pos_x >= 0) & (new_pos_x < high_W) & 
                (new_pos_y >= 0) & (new_pos_y < high_H))
        
        if valid.sum() == 0:
            break
            
        # Draw lines from old to new positions (vectorized)
        if step > 0:  # Skip first step
            _draw_lines(accumulator, pos_x[valid], pos_y[valid], 
                                 new_pos_x[valid], new_pos_y[valid], 
                                 255 - bg_color)
        
        # Update positions (only for valid particles)
        pos_x = new_pos_x[valid]
        pos_y = new_pos_y[valid]
    
    # Convert accumulator to image
    img_array = accumulator.clamp(0, 255).cpu().numpy().astype('uint8')
    img = Image.fromarray(img_array, mode='L')
    
    # Downsample using high-quality resampling (same as non-vectorized version)
    return img.resize((W, H), Image.Resampling.LANCZOS)


def _draw_lines(accumulator, x1, y1, x2, y2, intensity):
    """
    Fast vectorized line drawing - just draw endpoints and midpoints.
    Much faster than full line rasterization.
    """
    H, W = accumulator.shape
    
    # Convert to integer coordinates
    x1_int = x1.long().clamp(0, W-1)
    y1_int = y1.long().clamp(0, H-1)
    x2_int = x2.long().clamp(0, W-1)  
    y2_int = y2.long().clamp(0, H-1)
    
    # Draw start points
    accumulator[y1_int, x1_int] = intensity  # Set to line color directly
    
    # Draw end points  
    accumulator[y2_int, x2_int] = intensity  # Set to line color directly
    
    # Draw midpoints for better line appearance
    x_mid = ((x1_int + x2_int) // 2).clamp(0, W-1)
    y_mid = ((y1_int + y2_int) // 2).clamp(0, H-1)
    accumulator[y_mid, x_mid] = intensity  # Set to line color directly
