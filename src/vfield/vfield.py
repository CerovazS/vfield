import torch
import torch.nn.functional as torch_F
from PIL import Image
from tqdm import tqdm


def render_flow_field(grid, displacement, W=1000, H=1000, particles=2_000, steps=100, step_size=0.002, 
                     bg_color=(0, 0, 0), trace_color=(255, 255, 255), random_colors=False, 
                     antialias=True, aa_factor=2) -> Image.Image:
    """
    Vectorized flow field rendering with proper RGB color support.
    
    Args:
        bg_color: Background color as RGB tuple (r, g, b) where each value is 0-255
        trace_color: Default trace color as RGB tuple (r, g, b) where each value is 0-255
        random_colors: If True, each particle trail will be drawn with a random color variation
        antialias: Enable antialiasing for smoother output
        aa_factor: Antialiasing factor (higher = better quality but slower)
    """
    # Determine rendering dimensions
    if antialias:
        render_W, render_H = W * aa_factor, H * aa_factor
    else:
        render_W, render_H = W, H
    
    # Get grid bounds
    min_x = grid[:, 0].min().item()
    max_x = grid[:, 0].max().item()
    min_y = grid[:, 1].min().item()
    max_y = grid[:, 1].max().item()
    
    # Create interpolation function
    interp_res = 512
    F_vectorized = _create_interpolation_function(grid, displacement, min_x, max_x, min_y, max_y, interp_res, render_W, render_H)
    
    # Generate particle colors
    particle_colors = _generate_particle_colors(particles, trace_color, random_colors, grid.device)
    
    # Create RGB accumulator
    accumulator = _create_rgb_accumulator(render_H, render_W, bg_color, grid.device)
    
    # Run particle simulation
    _simulate_particles(F_vectorized, particles, steps, step_size, render_W, render_H, particle_colors, accumulator, grid.device)
    
    print("Converting to rgb image...")
    img = _fast_tensor_to_pil(accumulator)
    
    # Downsample if antialiasing was used
    if antialias:
        img = img.resize((W, H), Image.Resampling.LANCZOS)
    
    return img


def _interpolate_grid(grid_points, values, interp_res):
    """
    Ultra-fast interpolation assuming grid_points form a regular grid.
    """
    device = grid_points.device
    
    # Determine original grid size (assuming it's roughly square)
    n_points = len(grid_points)
    orig_res = int(torch.sqrt(torch.tensor(n_points, dtype=torch.float, device=device)).item())
    
    # If the original data forms a regular grid, reshape it
    if orig_res * orig_res == n_points:
        # Perfect square - reshape directly
        values_2d = values.view(orig_res, orig_res)
    else:
        # Find closest square size
        orig_res = int(torch.sqrt(torch.tensor(n_points, dtype=torch.float, device=device)).ceil().item())
        # Pad if needed and reshape
        values_padded = torch.cat([values, torch.zeros(orig_res*orig_res - n_points, device=device)])
        values_2d = values_padded.view(orig_res, orig_res)
    
    # Use PyTorch's interpolate function for super fast resampling
    values_4d = values_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Resize to target resolution using bilinear interpolation
    interpolated = torch_F.interpolate(values_4d, size=(interp_res, interp_res), 
                                      mode='bilinear', align_corners=False)
    
    return interpolated.squeeze(0).squeeze(0)  # [interp_res, interp_res]


def _create_interpolation_function(grid, displacement, min_x, max_x, min_y, max_y, interp_res, W, H):
    """
    Create a vectorized interpolation function for the given grid and displacement.
    """
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
    
    return F_vectorized


def _generate_particle_colors(particles, trace_color, random_colors, device):
    """
    Generate colors for all particles based on trace color and random_colors setting.
    """
    if random_colors:
        particle_colors = torch.zeros((particles, 3), device=device)
        
        # Check if trace color is black (0, 0, 0) for grayscale variations
        if trace_color == (0, 0, 0):
            # For black trace color, generate grayscale variations (same value for all channels)
            grayscale_values = torch.rand(particles, device=device) * 255  # 0 to 255
            particle_colors[:, 0] = grayscale_values  # R
            particle_colors[:, 1] = grayscale_values  # G  
            particle_colors[:, 2] = grayscale_values  # B
        else:
            # For non-black colors, generate random RGB variations around the base trace color
            for i in range(3):  # R, G, B channels
                # Add random variation (±50) to each channel
                variation = (torch.rand(particles, device=device) - 0.5) * 100
                particle_colors[:, i] = torch.clamp(trace_color[i] + variation, 0, 255)
    else:
        # All particles use the same trace color
        particle_colors = torch.tensor(trace_color, device=device).float().unsqueeze(0).expand(particles, 3)
    
    return particle_colors


def _create_rgb_accumulator(H, W, bg_color, device):
    """
    Create an RGB accumulator initialized with the background color.
    """
    accumulator = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    accumulator[:, :, 0] = bg_color[0]  # R
    accumulator[:, :, 1] = bg_color[1]  # G
    accumulator[:, :, 2] = bg_color[2]  # B
    return accumulator


def _simulate_particles(F_vectorized, particles, steps, step_size, W, H, particle_colors, accumulator, device):
    """
    Run the particle simulation and draw the flow lines.
    """
    # Initialize all particles at once
    torch.manual_seed(0)
    pos_x = torch.rand(particles, device=device) * W
    pos_y = torch.rand(particles, device=device) * H
    
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
            _draw_lines_rgb(accumulator, pos_x[valid], pos_y[valid], 
                         new_pos_x[valid], new_pos_y[valid], 
                         particle_colors[valid])

        # Update positions and colors (only for valid particles)
        pos_x = new_pos_x[valid]
        pos_y = new_pos_y[valid]
        particle_colors = particle_colors[valid]


def _draw_lines_rgb(accumulator, x1, y1, x2, y2, colors):
    """
    Fast vectorized RGB line drawing - draws endpoints and midpoints.
    
    Args:
        accumulator: RGB tensor of shape (H, W, 3)
        colors: RGB colors tensor of shape (N, 3) where N is number of lines
    """
    H, W, _ = accumulator.shape
    
    # Convert to integer coordinates
    x1_int = x1.long().clamp(0, W-1)
    y1_int = y1.long().clamp(0, H-1)
    x2_int = x2.long().clamp(0, W-1)  
    y2_int = y2.long().clamp(0, H-1)
    
    # Ensure colors have the correct dtype
    colors = colors.float()
    
    # Draw start points (all RGB channels)
    accumulator[y1_int, x1_int, :] = colors
    
    # Draw end points  
    accumulator[y2_int, x2_int, :] = colors
    
    # Draw midpoints for better line appearance
    x_mid = ((x1_int + x2_int) // 2).clamp(0, W-1)
    y_mid = ((y1_int + y2_int) // 2).clamp(0, H-1)
    accumulator[y_mid, x_mid, :] = colors


def _fast_tensor_to_pil(tensor, mode='RGB'):
    """
    Ultra-fast tensor to PIL conversion with memory optimization.
    For very large tensors, this can be significantly faster.
    """
    with torch.no_grad():
        # Ensure tensor is on CPU and in the right format
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Convert to uint8 efficiently
        if tensor.dtype != torch.uint8:
            tensor = tensor.clamp(0, 255).byte()
        
        # Convert to numpy and create PIL image
        return Image.fromarray(tensor.numpy(), mode=mode)
