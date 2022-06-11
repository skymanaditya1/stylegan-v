import numpy as np
import torch

num_interpolate_points = 10

# spherical linear interpolation (slerp)
def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        # L'Hopital's rule/LERP
        return (1.0-val) * low + val * high
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high
    
# uniform interpolation between two points in latent space
def interpolate_points(p1, p2):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=num_interpolate_points)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = slerp(ratio, p1, p2)
        vectors.append(v)
    return vectors


num_videos = 5
z_dim = 128
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# sample random noise using torch
all_z = torch.randn(num_videos, z_dim, device=device) # [curr_batch_size, z_dim]
print(f'Z shape : {all_z.shape}')

# interpolating 
p1 = torch.randn(z_dim, device=device)
p2 = torch.randn(z_dim, device=device)

returned_vectors = interpolate_points(p1, p2)
print(returned_vectors[0].shape) # generates num_interpolate_points vectors


torch_tensors = torch.vstack(returned_vectors)
print(torch_tensors.shape)