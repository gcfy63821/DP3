import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

from torchvision.models import resnet34, resnet18
from diffusion_policy_3d.model.vision.pointnet_extractor import PointNetEncoderXYZ, PointNetEncoderXYZRGB

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

    
class ForceSensorEncoder(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=12,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[ForceSensorEncoder] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[ForceSensorEncoder] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 12, cprint(f"ForceSensorEncoder only supports 12 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0] # ?
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

class UltrasoundDPEncoder(nn.Module):
    '''
    resnet-34 for image
    mlp for force and 
    '''
    def __init__(
        self,
        observation_space: Dict,
        out_channel = 256,
        ultrasound_encoder_cfg = None,
        debug: bool = False,
    ):
        super(UltrasoundDPEncoder, self).__init__()
        
        self.n_output_channels = out_channel
        self.resnet_output_dim = ultrasound_encoder_cfg.resnet_output_dim
        self.force_input_dim = observation_space['force']
        self.ee_input_dim = observation_space['state']

        block_channel = [64, 128, 256]
        self.debug = debug
        pretrained = ultrasound_encoder_cfg.pretrained
        use_layernorm = ultrasound_encoder_cfg.use_layernorm
       
        
        # ResNet-34 for Image Encoding
        resnet = resnet34(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Remove the fully connected layer
        self.cnn_fc = nn.Linear(resnet.fc.in_features, self.resnet_output_dim)
        
        
        # MLP for Force Sensor Encoding
        # self.force_mlp = nn.Sequential(
        #     create_mlp(self.force_input_dim, out_channel, net_arch, state_mlp_activation_fn),
        #     nn.Linear(net_arch[-1], self.resnet_output_dim),
        # )
        self.force_mlp = nn.Sequential(
            nn.Linear(self.force_input_dim[0], block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[-1], self.resnet_output_dim),
        )
        
        
        # MLP for Joint State Encoding
        # self.joint_mlp = nn.Sequential(
        #     create_mlp(self.ee_input_dim, out_channel, net_arch, state_mlp_activation_fn),
        #     nn.Linear(self.ee_input_dim[-1], self.resnet_output_dim),
        # )
        self.joint_mlp = nn.Sequential(
            nn.Linear(self.ee_input_dim[0], block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[-1], self.resnet_output_dim),
        )
        
        # Final Fully Connected Layer
        self.final_fc = nn.Linear(3 * self.resnet_output_dim, out_channel)
    
    # @staticmethod
    # def create_mlp(input_dim: int, net_arch: List[int], activation_fn: Type[nn.Module]) -> List[nn.Module]:
    #     layers = []
    #     for hidden_dim in net_arch:
    #         layers.append(nn.Linear(input_dim, hidden_dim))
    #         layers.append(activation_fn())
    #         input_dim = hidden_dim
    #     return layers

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:

        img = observations['img']  # Shape: [B, C, H, W]   
        # print('imgshape:',img.shape)     
        img_features = self.cnn(img)  # Shape: [B, 512, 1, 1]
        img_features = img_features.view(img_features.size(0), -1)  # Shape: [B, 512]
        img_features = self.cnn_fc(img_features)  # Shape: [B, resnet_output_dim]
        
        force = observations['force']  # Shape: [B, 6]
        force_features = self.force_mlp(force)  # Shape: [B, resnet_output_dim]

        joint = observations['state']  # Shape: [B, N]
        joint_features = self.joint_mlp(joint)  # Shape: [B, resnet_output_dim]

        # Concatenate and Process
        combined_features = torch.cat([img_features, force_features, joint_features], dim=-1)  # Shape: [B, 3 * resnet_output_dim]
            
        
        output = self.final_fc(combined_features)  # Shape: [B, output_dim]


        if self.debug:
            print(f"[DEBUG] Input image shape: {img.shape}")
            print(f"[DEBUG] Image features after CNN shape: {img_features.shape}")
            print(f"[DEBUG] Image features after FC layer: {img_features.shape}")
            print(f"[DEBUG] Force input shape: {force.shape}")
            print(f"[DEBUG] Force features after MLP: {force_features.shape}")
            print(f"[DEBUG] Joint input shape: {joint.shape}")
            print(f"[DEBUG] Joint features after MLP: {joint_features.shape}")
            print(f"[DEBUG] Combined features shape: {combined_features.shape}")
            print(f"[DEBUG] Final output shape: {output.shape}")

        return output
    
    def output_shape(self):
        return self.n_output_channels
    
class PositionEncoder(nn.Module):
    '''
    mlp for joint
    '''
    def __init__(
        self,
        observation_space: Dict,
        out_channel = 256,
        ultrasound_encoder_cfg = None,
        pretrained: bool = True,  # Whether to use pretrained weights for ResNet
        use_layernorm: bool=False,
        debug: bool = False,
    ):
        super(PositionEncoder, self).__init__()
        
        self.n_output_channels = out_channel
        self.ee_input_dim = observation_space['state']

        block_channel = [64, 128, 256]
        self.debug = debug
       
        
        
        self.joint_mlp = nn.Sequential(
            nn.Linear(self.ee_input_dim[0], block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[-1], out_channel),
        )
        
        # Final Fully Connected Layer
        self.final_fc = nn.Linear(out_channel, out_channel)
    
    # @staticmethod
    # def create_mlp(input_dim: int, net_arch: List[int], activation_fn: Type[nn.Module]) -> List[nn.Module]:
    #     layers = []
    #     for hidden_dim in net_arch:
    #         layers.append(nn.Linear(input_dim, hidden_dim))
    #         layers.append(activation_fn())
    #         input_dim = hidden_dim
    #     return layers

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:



        joint = observations['state']  # Shape: [B, N]
        joint_features = self.joint_mlp(joint)  # Shape: [B, resnet_output_dim]

 
        
        output = self.final_fc(joint_features)  # Shape: [B, output_dim]


        if self.debug:
            print(f"[DEBUG] Joint input shape: {joint.shape}")
            print(f"[DEBUG] Joint features after MLP: {joint_features.shape}")
            print(f"[DEBUG] Final output shape: {output.shape}")

        return output
    
    def output_shape(self):
        return self.n_output_channels


class USPCEncoder(nn.Module):
    '''
    resnet-34 for image
    mlp for force and joint
    mlp for pointcloud
    '''
    def __init__(
        self,
        observation_space: Dict,
        out_channel = 256,
        ultrasound_encoder_cfg = None,
        pointcloud_encoder_cfg=None,
        img_crop_shape = None,
        use_pc_color=False,
        pointnet_type='pointnet',
        debug: bool = False,
    ):
        super(USPCEncoder, self).__init__()

        self.point_cloud_key = 'point_cloud'
        self.imagination_shape = None
        
        self.n_output_channels = out_channel
        self.resnet_output_dim = ultrasound_encoder_cfg.resnet_output_dim
        self.state_output_dim = ultrasound_encoder_cfg.state_output_dim
        self.force_output_dim = ultrasound_encoder_cfg.force_output_dim
        self.force_input_dim = observation_space['force']
        self.state_shape = observation_space['state']
        self.point_cloud_shape = observation_space[self.point_cloud_key]

        block_channel = [64, 64]
        self.debug = debug

        pretrained = ultrasound_encoder_cfg.pretrained
        use_layernorm = ultrasound_encoder_cfg.use_layernorm
       
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        
        # ResNet-34 for Image Encoding
        resnet = resnet34(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Remove the fully connected layer
        self.cnn_fc = nn.Linear(resnet.fc.in_features, self.resnet_output_dim)
        
        
        # MLP for Force Sensor Encoding
        # self.force_mlp = nn.Sequential(
        #     create_mlp(self.force_input_dim, out_channel, net_arch, state_mlp_activation_fn),
        #     nn.Linear(net_arch[-1], self.resnet_output_dim),
        # )
        self.force_mlp = nn.Sequential(
            nn.Linear(self.force_input_dim[0], block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[-1], self.force_output_dim),
        )
        
        
        # MLP for Joint State Encoding
        # self.joint_mlp = nn.Sequential(
        #     create_mlp(self.ee_input_dim, out_channel, net_arch, state_mlp_activation_fn),
        #     nn.Linear(self.ee_input_dim[-1], self.resnet_output_dim),
        # )
        self.joint_mlp = nn.Sequential(
            nn.Linear(self.state_shape[0], block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[-1], self.state_output_dim),
        )
        
        # Final Fully Connected Layer
        # self.final_fc = nn.Linear(3 * self.resnet_output_dim, out_channel)
        self.n_output_channels += self.resnet_output_dim + self.state_output_dim + self.force_output_dim
    
    # @staticmethod
    # def create_mlp(input_dim: int, net_arch: List[int], activation_fn: Type[nn.Module]) -> List[nn.Module]:
    #     layers = []
    #     for hidden_dim in net_arch:
    #         layers.append(nn.Linear(input_dim, hidden_dim))
    #         layers.append(activation_fn())
    #         input_dim = hidden_dim
    #     return layers

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:

        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        # if self.use_imagined_robot:
        #     img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
        #     points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)    # B * out_channel

        img = observations['img']  # Shape: [B, C, H, W]   
        # print('imgshape:',img.shape)     
        img_features = self.cnn(img)  # Shape: [B, 512, 1, 1]
        img_features = img_features.view(img_features.size(0), -1)  # Shape: [B, 512]
        img_features = self.cnn_fc(img_features)  # Shape: [B, resnet_output_dim]
        
        force = observations['force']  # Shape: [B, 6]
        force_features = self.force_mlp(force)  # Shape: [B, resnet_output_dim]

        joint = observations['state']  # Shape: [B, N]
        joint_features = self.joint_mlp(joint)  # Shape: [B, resnet_output_dim]

        # Concatenate and Process
        combined_features = torch.cat([pn_feat, img_features, force_features, joint_features], dim=-1)  # Shape: [B, out_channels + resnet_output_dim + force_output_dim + state_output_dim]
            
        output = combined_features
        # output = self.final_fc(combined_features)  # Shape: [B, output_dim]


        if self.debug:
            print(f"[DEBUG] Input image shape: {img.shape}")
            print(f"[DEBUG] Image features after CNN shape: {img_features.shape}")
            print(f"[DEBUG] Image features after FC layer: {img_features.shape}")
            print(f"[DEBUG] Force input shape: {force.shape}")
            print(f"[DEBUG] Force features after MLP: {force_features.shape}")
            print(f"[DEBUG] Joint input shape: {joint.shape}")
            print(f"[DEBUG] Joint features after MLP: {joint_features.shape}")
            print(f"[DEBUG] Combined features shape: {combined_features.shape}")
            print(f"[DEBUG] Final output shape: {output.shape}")

        return output
    
    def output_shape(self):
        return self.n_output_channels
    


class ForcePositionEncoder(nn.Module):
    '''
    mlp for force and joint
    '''
    def __init__(
        self,
        observation_space: Dict,
        out_channel = 256,
        ultrasound_encoder_cfg = None,
        debug: bool = False,
    ):
        super(ForcePositionEncoder, self).__init__()

        self.point_cloud_key = 'point_cloud'
        self.imagination_shape = None
        
        self.n_output_channels = out_channel
        self.state_output_dim = ultrasound_encoder_cfg.state_output_dim
        self.force_output_dim = ultrasound_encoder_cfg.force_output_dim
        self.force_input_dim = observation_space['force']
        self.state_shape = observation_space['state']

        block_channel = [64, 128]
        self.debug = debug

        pretrained = ultrasound_encoder_cfg.pretrained
        use_layernorm = ultrasound_encoder_cfg.use_layernorm
       

        
        # MLP for Force Sensor Encoding
        # self.force_mlp = nn.Sequential(
        #     create_mlp(self.force_input_dim, out_channel, net_arch, state_mlp_activation_fn),
        #     nn.Linear(net_arch[-1], self.resnet_output_dim),
        # )
        self.force_mlp = nn.Sequential(
            nn.Linear(self.force_input_dim[0], block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[-1], self.force_output_dim),
        )
        
        
        # MLP for Joint State Encoding
        # self.joint_mlp = nn.Sequential(
        #     create_mlp(self.ee_input_dim, out_channel, net_arch, state_mlp_activation_fn),
        #     nn.Linear(self.ee_input_dim[-1], self.resnet_output_dim),
        # )
        self.joint_mlp = nn.Sequential(
            nn.Linear(self.state_shape[0], block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[-1], self.state_output_dim),
        )
        
        # Final Fully Connected Layer
        self.final_fc = nn.Linear(self.state_output_dim + self.force_output_dim, out_channel)
        # self.n_output_channels +=  self.state_output_dim + self.force_output_dim
    
    # @staticmethod
    # def create_mlp(input_dim: int, net_arch: List[int], activation_fn: Type[nn.Module]) -> List[nn.Module]:
    #     layers = []
    #     for hidden_dim in net_arch:
    #         layers.append(nn.Linear(input_dim, hidden_dim))
    #         layers.append(activation_fn())
    #         input_dim = hidden_dim
    #     return layers

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:

        # points = observations[self.point_cloud_key]
        # assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        # if self.use_imagined_robot:
        #     img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
        #     points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        # pn_feat = self.extractor(points)    # B * out_channel

        # img = observations['img']  # Shape: [B, C, H, W]   
        # print('imgshape:',img.shape)     
        # img_features = self.cnn(img)  # Shape: [B, 512, 1, 1]
        # img_features = img_features.view(img_features.size(0), -1)  # Shape: [B, 512]
        # img_features = self.cnn_fc(img_features)  # Shape: [B, resnet_output_dim]
        
        force = observations['force']  # Shape: [B, 6]
        force_features = self.force_mlp(force)  # Shape: [B, resnet_output_dim]

        joint = observations['state']  # Shape: [B, N]
        joint_features = self.joint_mlp(joint)  # Shape: [B, resnet_output_dim]

        # Concatenate and Process
        combined_features = torch.cat([ force_features, joint_features], dim=-1)  # Shape: [B, out_channels + resnet_output_dim + force_output_dim + state_output_dim]
            
        # output = combined_features
        output = self.final_fc(combined_features)  # Shape: [B, output_dim]


        if self.debug:
            print(f"[DEBUG] Force input shape: {force.shape}")
            print(f"[DEBUG] Force features after MLP: {force_features.shape}")
            print(f"[DEBUG] Joint input shape: {joint.shape}")
            print(f"[DEBUG] Joint features after MLP: {joint_features.shape}")
            print(f"[DEBUG] Combined features shape: {combined_features.shape}")
            print(f"[DEBUG] Final output shape: {output.shape}")

        return output
    
    def output_shape(self):
        return self.n_output_channels
    


class Ultrasound2CamEncoder(nn.Module):
    '''
    resnet-34 for image2, 4channel
    resnet-18 for image1, 1channel
    mlp for force and 
    '''
    def __init__(
        self,
        observation_space: Dict,
        out_channel = 256,
        ultrasound_encoder_cfg = None,
        debug: bool = False,
    ):
        super(Ultrasound2CamEncoder, self).__init__()
        
        self.n_output_channels = out_channel
        self.resnet_output_dim = ultrasound_encoder_cfg.resnet_output_dim
        self.force_input_dim = observation_space['force']
        self.state_shape = observation_space['state']

        self.state_output_dim = ultrasound_encoder_cfg.state_output_dim
        self.force_output_dim = ultrasound_encoder_cfg.force_output_dim

        

        block_channel = [64, 128]
        self.debug = debug
        pretrained = ultrasound_encoder_cfg.pretrained
        use_layernorm = ultrasound_encoder_cfg.use_layernorm
       
        
        # ResNet-18 for Image Encoding
        resnet1 = resnet18(pretrained=False)
        resnet1.conv1 = nn.Conv2d(1, resnet1.conv1.out_channels, kernel_size=resnet1.conv1.kernel_size, 
                                  stride=resnet1.conv1.stride, padding=resnet1.conv1.padding, bias=resnet1.conv1.bias)
        resnet1_fc_in_features = resnet1.fc.in_features  # 获取 fc 层的输入特征数
        resnet1.fc = nn.Identity()
        state_dict1 = torch.load('data/resnet_model.pth')
        state_dict1 = {k.replace('resnet.', ''): v for k, v in state_dict1.items() if not k.startswith('fc.')}  # 移除全连接层的键
        state_dict1 = {k: v for k, v in state_dict1.items() if not k.startswith('fc.')}  # 移除全连接层的键
        resnet1.load_state_dict(state_dict1)
        self.cnn1 = nn.Sequential(*list(resnet1.children())[:-1])  # Remove the fully connected layer
        self.cnn_fc1 = nn.Linear(resnet1_fc_in_features, self.resnet_output_dim)

        # Resnet-34 for wrist camera
        resnet2 = resnet34(pretrained=False)
        resnet2.conv1 = nn.Conv2d(4, resnet2.conv1.out_channels, kernel_size=resnet2.conv1.kernel_size, 
                                  stride=resnet2.conv1.stride, padding=resnet2.conv1.padding, bias=resnet2.conv1.bias)
        resnet2_fc_in_features = resnet2.fc.in_features  # 获取 fc 层的输入特征数
        resnet2.fc = nn.Identity()
        state_dict2 = torch.load('data/resnet2_model.pth')
        state_dict2 = {k.replace('resnet.', ''): v for k, v in state_dict2.items() if not k.startswith('fc.')}  # 移除全连接层的键
        state_dict2 = {k: v for k, v in state_dict2.items() if not k.startswith('fc.')}  # 移除全连接层的键
        resnet2.load_state_dict(state_dict2)
        self.cnn2 = nn.Sequential(*list(resnet2.children())[:-1])  # Remove the fully connected layer
        self.cnn_fc2 = nn.Linear(resnet2_fc_in_features, self.resnet_output_dim)


        
        
        # MLP for Force Sensor Encoding
        # self.force_mlp = nn.Sequential(
        #     create_mlp(self.force_input_dim, out_channel, net_arch, state_mlp_activation_fn),
        #     nn.Linear(net_arch[-1], self.resnet_output_dim),
        # )
        self.force_mlp = nn.Sequential(
            nn.Linear(self.force_input_dim[0], block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[-1], self.force_output_dim),
        )
        
        
        # MLP for Joint State Encoding
        # self.joint_mlp = nn.Sequential(
        #     create_mlp(self.ee_input_dim, out_channel, net_arch, state_mlp_activation_fn),
        #     nn.Linear(self.ee_input_dim[-1], self.resnet_output_dim),
        # )
        self.joint_mlp = nn.Sequential(
            nn.Linear(self.state_shape[0], block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[-1], self.state_output_dim),
        )
        
        # Final Fully Connected Layer
        # self.final_fc = nn.Linear(3 * self.resnet_output_dim, out_channel)

        # self.final_fc = nn.Linear(self.state_output_dim + self.force_output_dim + 2*self.resnet_output_dim, out_channel)
    
    # @staticmethod
    # def create_mlp(input_dim: int, net_arch: List[int], activation_fn: Type[nn.Module]) -> List[nn.Module]:
    #     layers = []
    #     for hidden_dim in net_arch:
    #         layers.append(nn.Linear(input_dim, hidden_dim))
    #         layers.append(activation_fn())
    #         input_dim = hidden_dim
    #     return layers

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:

        img1 = observations['img']  # Shape: [B, C, H, W]   
        # print('imgshape:',img.shape)     
        img_features1 = self.cnn1(img1)  # Shape: [B, 512, 1, 1]
        img_features1 = img_features1.view(img_features1.size(0), -1)  # Shape: [B, 512]
        img_features1 = self.cnn_fc1(img_features1)  # Shape: [B, resnet_output_dim]

        img2 = observations['img2']  # Shape: [B, C, H, W]   
        # print('imgshape:',img.shape)     
        img_features2 = self.cnn2(img2)  # Shape: [B, 512, 1, 1]
        img_features2 = img_features2.view(img_features2.size(0), -1)  # Shape: [B, 512]
        img_features2 = self.cnn_fc2(img_features2)  # Shape: [B, resnet_output_dim]
        
        force = observations['force']  # Shape: [B, 6]
        force_features = self.force_mlp(force)  # Shape: [B, resnet_output_dim]

        joint = observations['state']  # Shape: [B, N]
        joint_features = self.joint_mlp(joint)  # Shape: [B, resnet_output_dim]

        # Concatenate and Process
        combined_features = torch.cat([img_features1, img_features2, force_features, joint_features], dim=-1)  # Shape: [B, 3 * resnet_output_dim]
            
        output = combined_features
        # output = self.final_fc(combined_features)  # Shape: [B, output_dim]


        if self.debug:
            print(f"[DEBUG] Input image shape: {img1.shape}")
            print(f"[DEBUG] Image features after CNN shape: {img_features1.shape}")
            print(f"[DEBUG] Image features after FC layer: {img_features1.shape}")
            print(f"[DEBUG] Force input shape: {force.shape}")
            print(f"[DEBUG] Force features after MLP: {force_features.shape}")
            print(f"[DEBUG] Joint input shape: {joint.shape}")
            print(f"[DEBUG] Joint features after MLP: {joint_features.shape}")
            print(f"[DEBUG] Combined features shape: {combined_features.shape}")
            print(f"[DEBUG] Final output shape: {output.shape}")

        return output
    
    def output_shape(self):
        return self.state_output_dim + self.force_output_dim + 2*self.resnet_output_dim
    