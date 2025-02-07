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