**0120**

changed hugging face version to 0.25.0

next step is dataset

    conda activate dp3

    bash scripts/train_policy.sh ultrasound_dp ultrasound_scan 0206 0 0  

**0122**

can train


    bash scripts/eval_policy.sh ultrasound_dp ultrasound_scan 0122 0 0


policy输出的格式是:array[14]前7维是关节角，之后三维是xyz，之后四维是orientation

参考scripts/convert_ultrasound_data.py

input参考ultrasound_scan.cfg中的obs, state 格式 = action

模型代码是3D-Diffusion-Policy/diffusion_policy_3d/policy/ultrasound_policy.py

输出的时候会调用predict_action(line 190)

checkpoint:
3D-Diffusion-Policy/data/outputs/ultrasound_scan-ultrasound_dp-0120_seed0/checkpoints/latest.ckpt
eval.py中写了如何调用训练好的模型(train.py line 346)，虽然现在输入数据还没有对齐。

**0123**
* 完成hybrid force-impedance controller：
    - Usage: `roslaunch franka_us_controllers my_ipd_force_controller.launch`
    - 接收期望位置的topic:`desired_pose`，接收期望力的topic:`desired_wrench`
    - 写的时候做了一个近似：假设在`panda_ft_frame`处的受力等于在`panda_link8`处了，两者的位置有一点不同，我使用的雅可比矩阵是法兰处的`getZeroJacobian(franka::Frame::kFlange)`。
        ![alt text](<Screenshot from 2025-01-24 11-49-35.png>)
    在urdf中区别如下：
    ```    
    <joint name="${arm_id}_ft_frame_joint" type="fixed">
      <parent link="${connected_to}" />
      <child link="${arm_id}_ft_frame" />
      <origin xyz="${x} ${y} ${z+0.02}" rpy="${r} ${p} ${y+0.7854}" />
    </joint>
    ```

**0124**

roslaunch franka_visualization franka_visualization.launch

roslaunch netft_rdt_driver ft_sensor.launch
roslaunch force_torque_sensor_calib pub_imu.launch
roslaunch gravity_compensation gravity_compensation.launch


source ~/.bashrc

conda activate dp3

bash scripts/run.sh ultrasound_dp ultrasound_scan 0124 0 0


倒是跑起来了

**0125**
normalizer里面action的normalize
obs的维数

**0206**

把数据迁移到了16T的盘里，只有处理好的zarr会放在3D文件夹

/media/robotics/ST_16T/crq/data/record_data/20250206
/home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/ultrasound_data0206.zarr'


完成了action和obs的修改，现在模型输入为相对位置以及总相对变换的12维（方位用rpy的delta角来表示）
之后需要确定一下这个采数据用的是panda_link8 还是别的，目前用的panda_link8
还有一个待改动的地方时num obs steps 现在只能是1，然后action为了方便我也改成1了，或许需要再修改一下。
然后采数据的频率，数据处理采样的N都是待确定的参数。

**0207**
之后要改一下这个输入只要compensated就行
然后控制器跟随的是panda_EE

看起来不需要这俩但是似乎跑policy的时候收不到原本的topic，明天检查一下名字
roslaunch force_torque_sensor_calib pub_imu.launch
roslaunch gravity_compensation gravity_compensation.launch

roslaunch franka_us_controllers my_ipd_force_controller_dp.launch

调试的时候会注释launch文件里marker部分。


### DP部分用法：

1. 采数据：

    python scripts/record_diffusion_data.py

    开始后单击q或点鼠标可以结束

2. 处理数据：先改下面这个代码里面的输入输出路径，输入为硬盘里采集好的数据（会自动创建日期文件夹）

    python scripts/convert_ultrasound_data.py 

    让 3D-Diffusion-Policy/diffusion_policy_3d/config/task/ultrasound_scan.yaml
    第32行的压缩包名字与输出匹配即可

3. train policy：

    bash scripts/train_policy.sh ultrasound_dp ultrasound_scan 0206-5 0 0 
    
    （好像那个日期是实验编号需要每次改一改）
    可以在wandb网页查看训练情况，本地每100epoch存一个policy

4. 运行policy：

    bash scripts/run.sh ultrasound_dp ultrasound_scan 0206-5 0 0



现在runner里面改成EE了。机械臂可以跟随policy的指令运动，目前的效果是会竖直向下动

**0211**

搞一个纯位置然后用相同输入看看

bash scripts/run_orbbec.sh ultrasound_dp ultrasound_force 0211-6 0 0 
用pretrained的，输入相对初始位置的位移、速度表征和位置。可以一定程度上跟随轨迹。

明天试着修改代码可以输出多个观测序列、加入点云。

roslaunch azure_kinect_ros_driver driver_orbbec.launch body_tracking_enabled:=false overwrite_robot_description:=false

**0212**

visuallize_pc.py可以查看点云，点云用env_prepare.launch得到的来当做输入，可能需要先calibrate再进行crop

今天先试着：
1.把n_obs_steps和n_action_steps搞出来 已完成
2.采集点云数据并且跑通 
    - 先写一个encoder，dataset包含图像，点云，深度和其他 已完成
    - 写采数据 已完成
    - 写runner 完成
3.在速度表征里面把时间带上 完成

目前已debug的代码版本在run_orbbec, ultrasound_force

bash scripts/train_policy.sh us_pc_dp ultrasound_pc 0212-1 0 0

**0213**
现在进行数据处理使用的是robodiff这个环境，之后估计想用点云输入也得使用这个环境。
把关于imagined robot的部分注释掉了

**0214**
想办法赶紧把加速搞定
先写runner看一眼eval里面的情况
runner写好了，但是注意：里面的很多参数除了用L2的都没有设置，因为没有设定什么算完成了轨迹。
然后train里面修改了一处，把304行注释了，认为step_log['test_mean_score'] = - train_loss
说明这个inference的速度问题不是来自模型本身，应该是数据存储的问题。
接下来根据inference代码来修改我们的inference代码

原本的环境改好了，现在所有的都用dp3就可以

**0215**

关于针对加点云的一些尝试：
在3D-Diffusion-Policy/diffusion_policy_3d/policy/ultrasound_policy.py 加入USPCDP，为使用混合encoder的policy，参考simple dp3进行修改得到
3D-Diffusion-Policy/diffusion_policy_3d/model/vision/my_encoder.py中写了混合encoder，与无点云的版本只加入了pointcloud
3D-Diffusion-Policy/diffusion_policy_3d/env_runner/ultrasound_runner.py中为eval时用到的runner


今天把只有力和位置的代码做个测试吧

bash scripts/train_policy.sh us_force_position_dp ultrasound_force_position 0215-2 0 0

bash scripts/eval_policy.sh us_force_position_dp ultrasound_force_position 0215-6 0 0

**0216**

今天
    1.训练一个走delta的model
    2. 严格inference和采数据时的时间控制。可以把周期定为0.2秒
    这个inference好像不太好办呀，需要跟控制器配合，但是控制器又不是跟速度相关的

训了两个，一个是5， 10， 一个是1，2，都是delta的，看看效果吧
但是这个2的曲线有一点过于陡峭了，不知道为什么

跑真机实验：打开控制器后，
bash scripts/run_force.sh us_force_position_dp ultrasound_force_position 0216-3 0 0

好像这个position还可以哈

不知道这两种是不是可以作为一个对比实验呢？

现在训了一个图像的和一个力的，明天来看看效果
图像的忘了改名字了,N=15,T=3
ultrasound_dp ultrasound_scan 0213-5 0 0
力的是
us_force_position_dp ultrasound_force_position 0216-4 0 0

**0217**
可以预测，等会儿打开效果吧
图像的改成N=15,T=5看看。 0217-2
bash scripts/train_policy.sh ultrasound_dp ultrasound_scan 0217-2 0 0

整理一下扫脖子的数据然后训一个吧
训练挂上了，0217-3
15 4

2.22
文章除了实验部分都ready
有force，image的实验

ablation
safety, comfortable,人填写问卷
和一般bc方法

和hardcode比

现在policy ros runner里面进行的是arm的测试

今天挂了一个20 3的，0218-1明天开一下服务器！之后万一训很多

**0218**
4Ss4JOM3JR3m

服务器

ssh -p 14822 crq@166.111.72.148 

**0220**
先用force来进行确认，把输入输出的频率固定到10hz

inference_force 使用了一些trik，让输出的desired叠加

接下来要做的事情：
    1. 测试一下扫脖子和扫手臂的能否work
    2. 重新处理数据 现在数据以0.1秒为间隔
    3. 配置一下服务器 已完成


scp -r -P 14822 /home/robotics/crq/3D-Diffusion-Policy/3D-Diffusion-Policy/data/ultrasound_data_neck.zarr crq@166.111.72.148:/home/crq/crq/DP3/3D-Diffusion-Policy/data

bash scripts/run_video.sh ultrasound_dp ultrasound_scan - 0 0

服务器上面跑了0220-3，4，5，5是cross attention film
那个里面的dp改成了复杂的，之后需要改回来。
明天搞相机

采数据的launch：
roslaunch franka_us_controllers record_data_env.launch


运用视频进行run的脚本：

bash scripts/run_video.sh ultrasound_dp ultrasound_scan - 0 0 

**0221**
搞了一个裁切的，应该可以帮助保存更多的信息。
今天把两个相机的encoder搞定

我希望把超声图像用resnet18，一个通道，然后腕部相机用resnet34，4个通道

已安装realsense sdk， 打开相机:
realsense-viewer

open_realsense.py 可以打开realsense相机


 bash scripts/run_2cam.sh ultrasound_dp_2cam ultrasound_2cam_scan - 0 0

 bash scripts/eval_policy.sh  ultrasound_dp_2cam ultrasound_2cam_scan 0221-1 0 0 

 bash scripts/train_policy.sh ultrasound_dp_2cam ultrasound_2cam_scan 0221-1 0 0 



 用欧拉角表示不够稳定，目前都换成了四元数