import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import numpy as np
import visualizer
import pytorch3d.ops as torch3d_ops
import torch


def farthest_point_sampling(points, K, use_cuda=False):
    points = torch.tensor(points, dtype=torch.float32)
    if use_cuda and torch.cuda.is_available():
        points = points.cuda()
    else:
        points = points.cpu()
    
    sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
    sampled_points = sampled_points.squeeze(0)
    sampled_points = sampled_points.cpu().numpy()
    return sampled_points, indices.cpu().numpy()

def preprocess_point_cloud(points, num_points=1024, use_cuda=False):

    # qw = 0.722037047140051
    # qx = -0.6677685888235922
    # qy = -0.12216735823105529
    # qz = 0.13350187609419234
    # x = 0.6649676791489088
    # y = -1.0718525225295776
    # z = 0.20009733257876774

    # # 计算旋转矩阵
    # R = np.array([
    #     [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
    #     [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
    #     [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    # ])

    # # 组合齐次变换矩阵
    # T = np.array([
    #     [R[0, 0], R[0, 1], R[0, 2], x],
    #     [R[1, 0], R[1, 1], R[1, 2], y],
    #     [R[2, 0], R[2, 1], R[2, 2], z],
    #     [0, 0, 0, 1]
    # ])

    # extrinsics_matrix = T

    WORK_SPACE = [
        [0.02, 0.25],
        [-0.3, 0.05],
        [0.15, 0.4] 
    ]


    # scale
    point_xyz = points[..., :3] * 0.2500000118743628
    point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
    # point_homogeneous = np.dot(point_homogeneous, extrinsics_matrix)
    point_xyz = point_homogeneous[..., :-1]
    points[..., :3] = point_xyz

    print(f"Point cloud range before crop: x[{points[..., 0].min()}, {points[..., 0].max()}], "
          f"y[{points[..., 1].min()}, {points[..., 1].max()}], z[{points[..., 2].min()}, {points[..., 2].max()}]")

    
    # crop
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                             (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                             (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]

    points_xyz = points[..., :3]
    points_xyz, sample_indices = farthest_point_sampling(points_xyz, num_points, use_cuda)
    # sample_indices = sample_indices.cpu()
    sample_indices = torch.tensor(sample_indices, dtype=torch.long).cpu()  # 确保 sample_indices 是一个 torch.Tensor
    points_rgb = points[sample_indices, 3:][0]
    points = np.hstack((points_xyz, points_rgb))
    
    return points_xyz



class PointCloudSubscriber:
    def __init__(self):
        rospy.init_node('pointcloud_subscriber', anonymous=True)
        self.pointcloud_data = None
        self.received_data = False
        rospy.Subscriber("/k4a/points2", PointCloud2, self.callback)
        rospy.loginfo("Waiting for point cloud data...")
        self.rate = rospy.Rate(10)

    def callback(self, msg):
        print('Received point cloud data.')
        # 将 PointCloud2 消息转换为点云数据
        cloud_points = list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")))
        
        # 转换为 numpy 数组
        self.pointcloud_data = np.array(cloud_points)
        self.pointcloud_data = preprocess_point_cloud(self.pointcloud_data)
        print('pointcloud shape:', self.pointcloud_data.shape)
        
        # 存储为 .npy 文件
        np.save('pointcloud.npy', self.pointcloud_data)
        print('Saved point cloud data to pointcloud.npy')
        
        # 标记已接收到数据
        self.received_data = True
        # 只获取一帧点云数据后停止订阅
        # rospy.signal_shutdown("Got one frame of point cloud data")

if __name__ == "__main__":
    try:
        subscriber = PointCloudSubscriber()
        while not subscriber.received_data:
            subscriber.rate.sleep()
        print("Visualizing point cloud data...")
        visualizer.visualize_pointcloud(subscriber.pointcloud_data)
        
        
    except rospy.ROSInterruptException:
        pass