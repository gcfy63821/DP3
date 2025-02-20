import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import time

def point_cloud_sampling(point_cloud: np.ndarray, num_points: int, method: str = 'fps'):
    """
    Support different point cloud sampling methods
    point_cloud: (N, 6), xyz+rgb or (N, 3), xyz
    """
    
    # Define workspace (bounding box) for cropping
    WORK_SPACE = [
        [0.08, 1,0],  # x-axis range
        [-1.2, 0.2],  # y-axis range
        [0.6, 1.6]    # z-axis range
    ]

    if num_points == 'all':  # Use all points
        return point_cloud
    
    if point_cloud.shape[0] <= num_points:
        # If fewer points than requested, pad with zeros
        point_cloud_dim = point_cloud.shape[-1]
        point_cloud = np.concatenate([point_cloud, np.zeros((num_points - point_cloud.shape[0], point_cloud_dim))], axis=0)
        return point_cloud
    
    # points = point_cloud

    if method == 'uniform':
        # Uniform sampling
        if points.shape[0] <= num_points:
            return points  # If cropped points are fewer than num_points, return the cropped set
        sampled_indices = np.random.choice(points.shape[0], num_points, replace=False)
        points = points[sampled_indices]
    
    elif method == 'fps':
        # Fast point cloud sampling using torch3d
        if point_cloud.shape[0] <= num_points:
            return point_cloud  # If cropped points are fewer than num_points, return the cropped set
        
        point_cloud = torch.from_numpy(point_cloud).cuda()

        # point_cloud = point_cloud[...,:3] * 
        # Crop points based on workspace range using np.where
        mask = (
            (point_cloud[:, 0] > WORK_SPACE[0][0]) & (point_cloud[:, 0] < WORK_SPACE[0][1]) &
            (point_cloud[:, 1] > WORK_SPACE[1][0]) & (point_cloud[:, 1] < WORK_SPACE[1][1]) &
            (point_cloud[:, 2] > WORK_SPACE[2][0]) & (point_cloud[:, 2] < WORK_SPACE[2][1])
        )
        point_cloud = point_cloud[mask].unsqueeze(0)

        num_points_tensor = torch.tensor([num_points]).cuda()
        # Remember to only use the coordinates for FPS sampling
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=point_cloud[...,:3], K=num_points_tensor)
        points = point_cloud.squeeze(0).cpu().numpy()
        points = points[sampled_indices.squeeze(0).cpu().numpy()]
    
    else:
        raise NotImplementedError(f"Point cloud sampling method {method} not implemented")

    return points
