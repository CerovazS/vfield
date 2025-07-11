import torch
import matplotlib.pyplot as plt
import numpy as np
import lic
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
    

def render_lic(grid, displacement, R=1.6, RES=1024, length=30, normalize=True):

    # create high-resolution grid
    x_lic, y_lic = np.mgrid[-R:R:RES*1j, -R:R:RES*1j]
    
    # analytic test field: a clockwise vortex centred at (0,0)
    # u = -y_lic
    # v = x_lic
        
    grid_np = grid.numpy()
    displacement_np = displacement.numpy()

    # create points for interpolation (flatten the high-res grid)
    points_lic = np.column_stack((x_lic.ravel(), y_lic.ravel()))
    
    # interpolate displacement components
    u_interp = griddata(grid_np, displacement_np[:, 0], points_lic, 
                        method='linear', fill_value=0.0)
    v_interp = griddata(grid_np, displacement_np[:, 1], points_lic, 
                        method='linear', fill_value=0.0)
    
    # reshape back to grid
    u = u_interp.reshape(RES, RES)
    v = v_interp.reshape(RES, RES)

    if normalize:
        mag = np.hypot(u, v)
        u = u / (mag + 1e-9)
        v = v / (mag + 1e-9)

    # render the LIC
    seed = lic.gen_seed((RES, RES), noise="white")
    lic_img = lic.lic(u, v, seed, length=length)

    lic_img = np.rot90(lic_img, k=3)  # rotate 90 degrees ccw
    lic_img = np.fliplr(lic_img)      # flip left-right
    
    # display the LIC and vector field side by side

    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)

    plt.imshow(lic_img, cmap='gray', origin='lower', extent=[-R, R, -R, R])
    plt.title(f'LIC rendering (R={R}, RES={RES})')
    plt.axis('off')
    
    plt.subplot(122)

    skip = max(1, RES // 40)  # subsample for clarity
    x_sub = x_lic[::skip, ::skip]
    y_sub = y_lic[::skip, ::skip]
    u_sub = u[::skip, ::skip]
    v_sub = v[::skip, ::skip]
    
    plt.quiver(x_sub, y_sub, u_sub, v_sub, alpha=0.7, scale=30)
    plt.xlim(-R, R)
    plt.ylim(-R, R)
    plt.title(f'Vector Field (subsampled every {skip} points)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return lic_img