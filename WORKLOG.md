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
记录一下之后需要做的事情:
