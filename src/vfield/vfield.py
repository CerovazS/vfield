import torch
import torch.nn.functional as torch_F
from PIL import Image
from tqdm import tqdm


def render_flow_field(grid, displacement, W=1000, H=1000, particles=2_000, steps=100, step_size=0.002, 
                     bg_color=(255, 255, 255), trace_color=(0, 0, 0), random_colors=False, 
                     antialias=True, aa_factor=2) -> Image.Image:
    """
    Particle-based flow field rendering.
    Runs on the device of the grid tensor (CPU or GPU).
    
    Args:
        grid: Tensor of shape (N, 2) representing the grid points (x, y)
        displacement: Tensor of shape (N, 2) representing the flow vectors at each grid point
        W: Width of the output image
        H: Height of the output image
        particles: Number of particles to simulate; each particle will leave a trace
        steps: Number of simulation steps for each particle
        step_size: Size of each step in the simulation (relative to image dimensions)
        bg_color: Background color as RGB tuple (r, g, b) where each value is 0-255
        trace_color: Trace color as RGB tuple (r, g, b) where each value is 0-255
        random_colors: If True, each particle trace will be drawn with a random color variation wrt trace_color.
                       Useful to generate different gray levels when the trace_color is black.
        antialias: Enable antialiasing for smoother output
        aa_factor: Antialiasing factor (higher = better quality but slower)

    Returns:
        Image: Rendered flow field as a PIL Image.
    """
    if antialias:
        render_W, render_H = W * aa_factor, H * aa_factor
    else:
        render_W, render_H = W, H
    
    min_x = grid[:, 0].min().item()
    max_x = grid[:, 0].max().item()
    min_y = grid[:, 1].min().item()
    max_y = grid[:, 1].max().item()
    
    interp_res = 512
    F_vectorized = _create_interpolation_function(grid, displacement, (min_x, max_x, min_y, max_y), interp_res, render_W, render_H)

    particle_colors = _generate_particle_colors(particles, trace_color, random_colors, grid.device)
    
    accumulator = _create_rgb_accumulator(render_H, render_W, bg_color, grid.device)
    
    _simulate_particles(F_vectorized, particles, steps, step_size, render_W, render_H, particle_colors, accumulator, grid.device)
    
    print("Converting to rgb image...")
    img = _tensor_to_pil(accumulator)

    # downsample if antialiasing was used
    if antialias:
        print("Resampling...")
        img = img.resize((W, H), Image.Resampling.LANCZOS)
    
    return img


def _interpolate_grid(grid_points, values, interp_res):
    """
    Fast interpolation assuming grid_points form a regular grid.
    """
    device = grid_points.device
    
    # determine original grid size (assuming it's roughly square)
    n_points = len(grid_points)
    orig_res = int(torch.sqrt(torch.tensor(n_points, dtype=torch.float, device=device)).item())
    
    if orig_res * orig_res == n_points:  # perfect square: reshape directly
        values_2d = values.view(orig_res, orig_res)
    else:                                # find closest square size, pad and reshape
        orig_res = int(torch.sqrt(torch.tensor(n_points, dtype=torch.float, device=device)).ceil().item())
        values_padded = torch.cat([values, torch.zeros(orig_res*orig_res - n_points, device=device)])
        values_2d = values_padded.view(orig_res, orig_res)
    
    # resize to target resolution using bilinear interpolation
    values_4d = values_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    interpolated = torch_F.interpolate(values_4d, size=(interp_res, interp_res), 
                                      mode='bilinear', align_corners=False)
    
    return interpolated.squeeze(0).squeeze(0)  # [interp_res, interp_res]


def _create_interpolation_function(grid, displacement, bounds, interp_res, W, H):
    """
    Create a vectorized interpolation function for the given grid and displacement.
    """
    min_x, max_x, min_y, max_y = bounds

    # pre-compute interpolated values
    u_grid = _interpolate_grid(grid, displacement[:, 0], interp_res)
    v_grid = _interpolate_grid(grid, displacement[:, 1], interp_res)
    
    def F_vectorized(px, py):
        """Vectorized lookup from pre-computed interpolation grid"""

        # convert to spatial coordinates
        x = (px/W) * (max_x - min_x) + min_x
        y = (py/H) * (max_y - min_y) + min_y
        
        # convert to grid indices
        i = ((y - min_y) / (max_y - min_y) * (interp_res - 1)).long().clamp(0, interp_res-1)
        j = ((x - min_x) / (max_x - min_x) * (interp_res - 1)).long().clamp(0, interp_res-1)
        
        # lookup
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
        
        if trace_color == (0, 0, 0):  # then grayscale variations
            grayscale_values = torch.rand(particles, device=device) * 255
            particle_colors[:, 0] = grayscale_values  # R
            particle_colors[:, 1] = grayscale_values  # G
            particle_colors[:, 2] = grayscale_values  # B
        else:                         # random RGB variations around the base trace color
            for i in range(3):
                variation = (torch.rand(particles, device=device) - 0.5) * 100  # plus/minus 50 variation
                particle_colors[:, i] = torch.clamp(trace_color[i] + variation, 0, 255)
    else:
        # all particles use the same trace color
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
    torch.manual_seed(0)
    pos_x = torch.rand(particles, device=device) * W
    pos_y = torch.rand(particles, device=device) * H
    
    # vectorized integration loop
    for step in tqdm(range(steps)):

        # velocities
        u, v = F_vectorized(pos_x, pos_y)
        
        # normalize velocities
        norm = torch.sqrt(u*u + v*v) + 1e-6
        u_norm = u / norm
        v_norm = v / norm
        
        # update positions
        new_pos_x = pos_x + u_norm * step_size * W
        new_pos_y = pos_y + v_norm * step_size * H
        
        # check bounds and keep only valid particles
        valid = ((new_pos_x >= 0) & (new_pos_x < W) & 
                (new_pos_y >= 0) & (new_pos_y < H))
        
        if valid.sum() == 0:
            break
            
        # draw lines from old to new positions
        if step > 0:
            _draw_lines_rgb(accumulator, pos_x[valid], pos_y[valid], 
                         new_pos_x[valid], new_pos_y[valid], 
                         particle_colors[valid])

        # update positions and colors
        pos_x = new_pos_x[valid]
        pos_y = new_pos_y[valid]
        particle_colors = particle_colors[valid]


def _draw_lines_rgb(accumulator, x1, y1, x2, y2, colors):
    """
    Fast vectorized RGB line drawing, draws endpoints and midpoints.
    """
    H, W, _ = accumulator.shape
    colors = colors.float()
    
    # convert to integer coordinates
    x1_int = x1.long().clamp(0, W-1)
    y1_int = y1.long().clamp(0, H-1)
    x2_int = x2.long().clamp(0, W-1)  
    y2_int = y2.long().clamp(0, H-1)
    
    # draw start points (all RGB channels)
    accumulator[y1_int, x1_int, :] = colors
    
    # draw end points  
    accumulator[y2_int, x2_int, :] = colors

    # draw midpoints for better line appearance
    x_mid = ((x1_int + x2_int) // 2).clamp(0, W-1)
    y_mid = ((y1_int + y2_int) // 2).clamp(0, H-1)
    accumulator[y_mid, x_mid, :] = colors


def _tensor_to_pil(tensor, mode='RGB'):
    """
    Tensor to PIL conversion with memory optimization.
    """
    with torch.no_grad():
        
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        if tensor.dtype != torch.uint8:
            tensor = tensor.clamp(0, 255).byte()
        
        return Image.fromarray(tensor.numpy(), mode=mode)
