**0120**

changed hugging face version to 0.25.0

next step is dataset

    conda activate dp3

    bash scripts/train_policy.sh ultrasound_dp ultrasound_scan 0120 0 0  

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