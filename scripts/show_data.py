import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from termcolor import cprint
import rospy, tf
import tf2_ros

from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped, WrenchStamped

tf_listener = tf.TransformListener()
curr_force = None

def ft_comp_callback(msg):
        ft_compensated = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])
        curr_force = ft_compensated

def get_panda_EE_transform():
        try:
            tf_listener.waitForTransform("/panda_link0", "/panda_EE", rospy.Time(0), rospy.Duration(4.0))
            (trans, rot) = tf_listener.lookupTransform("/panda_link0", "/panda_EE", rospy.Time(0))
            return np.array(trans), np.array(rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("TF Exception")
            return None, None

def extract_timestamp(file_path):
    # 提取文件名中的时间戳部分
    file_name = os.path.basename(file_path)
    time_str = file_name.split('_')[1]
    return int(time_str)

def display_data_3d(positions, forces, window_name="3D Position Display"):
    """
    在三维立体窗口中显示三维位置和力向量长度
    :param positions: 三维位置列表，形状为 (N, 3)
    :param forces: 力向量长度列表，形状为 (N,)
    :param window_name: 窗口名称
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(window_name)

    # 提取坐标
    xs = [pos[0] for pos in positions]
    ys = [pos[1] for pos in positions]
    zs = [pos[2] for pos in positions]

    # 绘制三维散点图
    scatter = ax.scatter(xs, ys, zs, c=forces, cmap='viridis', marker='o')

    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Force Length')

    # 设置轴标签
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')

    plt.show()

def main():
    rospy.init_node('ultrasound_policy_runner', anonymous=True)
    rospy.Subscriber("/ft_sensor/ft_compensated", WrenchStamped,ft_comp_callback)
        
    expert_data_path = '/media/robotics/ST_16T/crq/data/record_data/neck23'
    subfolders = [os.path.join(expert_data_path, f) for f in os.listdir(expert_data_path) if os.path.isdir(os.path.join(expert_data_path, f))]
    subfolders = sorted(subfolders)

    positions = []
    forces = []

    for subfolder in subfolders:
        print(subfolder)
        # 遍历子文件夹里的 .npy 文件
        npy_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.npy')]
        print(len(npy_files))
        files = sorted(npy_files, key=extract_timestamp)  # 确保文件按时间顺序排列
        if not files:
            continue  # 如果文件夹中没有 .npy 文件，则跳过

        first_file = files[0]
        data_dict = np.load(first_file, allow_pickle=True).item()
        position = data_dict['position']  # 假设三维位置存储在 'position' 键中
        force = data_dict['ft_compensated']  # 假设力向量存储在 'ft_compensated' 键中
        force_length = np.linalg.norm(force)  # 计算力向量的长度

        cprint(f'Processing {first_file}', 'green')
        positions.append(position)
        forces.append(force_length)


    position, orientation = get_panda_EE_transform()
    positions.append(position)
    forces.append()

    display_data_3d(positions, forces)

if __name__ == "__main__":
    main()