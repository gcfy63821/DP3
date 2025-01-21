import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

from torchvision.models import resnet34

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
        pretrained: bool = True,  # Whether to use pretrained weights for ResNet
        use_layernorm: bool=False,
        debug: bool = False,
    ):
        super(UltrasoundDPEncoder, self).__init__()
        
        self.n_output_channels = out_channel
        self.resnet_output_dim = ultrasound_encoder_cfg.resnet_output_dim
        self.force_input_dim = observation_space['force']
        self.ee_input_dim = observation_space['ee_pos']
        print(self.force_input_dim, self.ee_input_dim)

        block_channel = [64, 128, 256]

       
        
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

        img = observations['image']  # Shape: [B, C, H, W]        
        img_features = self.cnn(img)  # Shape: [B, 512, 1, 1]
        img_features = img_features.view(img_features.size(0), -1)  # Shape: [B, 512]
        img_features = self.cnn_fc(img_features)  # Shape: [B, resnet_output_dim]
        
        force = observations['force']  # Shape: [B, 6]
        force_features = self.force_mlp(force)  # Shape: [B, resnet_output_dim]

        joint = observations['ee_pos']  # Shape: [B, N]
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
    