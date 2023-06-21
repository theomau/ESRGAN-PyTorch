# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor


from escnn import gspaces  
import escnn.nn as enn  

__all__ = [
    "Discriminator", "RRDBNet", "ContentLoss",
    "discriminator", "g_p4_discriminator", "g_p4m_discriminator",
    "rrdbnet_x1", "rrdbnet_x2", "g_p8_rrdbnet_x2","g_p4m_rrdbnet_x2", "rrdbnet_x4", "rrdbnet_x8", 
    "content_loss",
]
class _G_P8_ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_G_P8_ResidualDenseBlock, self).__init__()
        # the model is equivariant under rotations by 90 degrees, modelled by C4
        self.r2_act = gspaces.rot2dOnR2(N=8)
        
        # the input image is a 3 or 1 scalar field, corresponding to the regular representation
        in_type = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        out_type = enn.FieldType(self.r2_act, growth_channels//8*[self.r2_act.regular_repr])
           
        # convolution 1
        self.block1 = enn.SequentialModule(
            # input size. (3) x 128 x 128
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=True),
            enn.ELU(out_type, inplace = True))    
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = enn.FieldType(self.r2_act, (channels + growth_channels * 1)//8*[self.r2_act.regular_repr])
        # the output type of the second convolution layer are 32 (growth_channel) regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, growth_channels//8*[self.r2_act.regular_repr])
        self.block2 = enn.SequentialModule(
            # input size. (3) x 128 x 128
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=True),
            enn.ELU(out_type, inplace = True))
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = enn.FieldType(self.r2_act, (channels + growth_channels * 2)//8*[self.r2_act.regular_repr])
        # the output type of the second convolution layer are 32 (growth_channel) regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, growth_channels//8*[self.r2_act.regular_repr])
        self.block3 = enn.SequentialModule(
            # input size. (3) x 128 x 128
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=True),
            enn.ELU(out_type, inplace = True))
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = enn.FieldType(self.r2_act, (channels + growth_channels * 3)//8*[self.r2_act.regular_repr])
        # the output type of the second convolution layer are 32 (growth_channel) regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, growth_channels//8*[self.r2_act.regular_repr])
        self.block4 = enn.SequentialModule(
            # input size. (3) x 128 x 128
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=True),
            enn.ELU(out_type, inplace = True))
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = enn.FieldType(self.r2_act, (channels + growth_channels * 4)//8*[self.r2_act.regular_repr])
        # the output type of the second convolution layer are 32 (growth_channel) regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        self.block5 = enn.SequentialModule(
            # input size. (3) x 128 x 128
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=True),
            enn.ELU(out_type, inplace = True))
        
        
        # the old output type is the input type to the next layer
        in_type = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        
        self.identity = enn.IdentityModule(in_type)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = enn.GeometricTensor(input, self.input_type)
        
        identity = x

        out1 = self.block1(x)
        
        cat = enn.tensor_directsum([x, out1])
        out2 = self.block2(cat)
        
        cat = enn.tensor_directsum([x, out1, out2])
        out3 = self.block3(cat)
        
        cat = enn.tensor_directsum([x, out1, out2, out3])
        out4 = self.block4(cat)
        
        cat = enn.tensor_directsum([x, out1, out2, out3, out4])
        out5 = self.block5(cat)
        
        out = out5 * 0.2
        
        out = out + identity
        
        return out


class _G_P8_ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_G_P8_ResidualResidualDenseBlock, self).__init__()
        
        # the model is equivariant under rotations by 90 degrees, modelled by C8
        self.r2_act = gspaces.rot2dOnR2(N=8)
        
        # the input image is a 3 or 1 scalar field, corresponding to the regular representation
        in_type = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        self.rdb1 = _G_P8_ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = _G_P8_ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = _G_P8_ResidualDenseBlock(channels, growth_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # # wrap the input tensor in a GeometricTensor
        # # (associate it with the input type)
        x = enn.GeometricTensor(input, self.input_type)
        
        identity = x
        
        x = x.tensor
        
        out = self.rdb1(x)
        out = out.tensor
    
        out = self.rdb2(out)
        out = out.tensor
        
        out = self.rdb3(out)
        
        
        out = out * 0.2
        out = out + identity
        out = out.tensor
        return out


class G_P8_RRDBNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
            growth_channels: int = 32,
            num_blocks: int = 23,
            upscale_factor: int = 4,
    ) -> None:
        
        super(G_P8_RRDBNet, self).__init__()
        self.upscale_factor = upscale_factor
        
        # the model is equivariant under rotations by 90 degrees, modelled by C8
        self.r2_act = gspaces.rot2dOnR2(N=8)
        
        # the input image is a 3 or 1 scalar field, corresponding to the trivial representation
        in_type = enn.FieldType(self.r2_act, in_channels*[self.r2_act.trivial_repr])
        in_type1 = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        self.input_type1 = in_type1
        
        # The first layer of convolutional layer.
        # first specify the output type of the convolutional layer
        # we choose 64 feature fields, each transforming under the regular representation of C8
        out_type = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        
        self.g_conv1 = enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1)
                       
        # Feature extraction backbone network.
        # the old output type is the input type to the next layer
        in_type = self.g_conv1.out_type
        
        trunk = []
        for _ in range(num_blocks):
            trunk.append(_G_P8_ResidualResidualDenseBlock(channels, growth_channels))
        
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.     
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        # the output type of the second convolution layer are 64 regular feature fields of C8
        out_type = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        
        self.g_conv2 = enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1)
        
        # Upsampling convolutional layer.
        if upscale_factor == 2:
            
            # the old output type is the input type to the next layer
            in_type = self.g_conv2.out_type
            
            
            self.upsampling0 = enn.R2Upsampling(in_type, scale_factor = 2, mode = "bilinear")
            
            
            # the output type of the Upsampling convolutional layer are 64 regular feature fields of C4
            out_type = enn.FieldType(self.r2_act, channels*[self.r2_act.regular_repr])
            
            #in_type = self.enn.R2Upsampling(in_type, scale_factor = 2, mode = "bilinear")
            self.upsampling1 = enn.SequentialModule(
                enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1),
                enn.ELU(out_type, inplace=True)
            )
                
        if upscale_factor !=2:
            
            print("pas fais encore")
        
        
        # the old output type is the input type to the next layer
        in_type = self.upsampling1.out_type
        # the output type of the Upsampling convolutional layer are 64 regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, channels*[self.r2_act.regular_repr])
        
        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1),
            enn.ELU(out_type, inplace=True)
        )
        
        # the old output type is the input type to the next layer
        in_type = self.conv3.out_type
        # the output type of the Upsampling convolutional layer are 64 regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, out_channels*[self.r2_act.trivial_repr])
        
        # Output layer.
        self.conv4 = enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1)

        # Initialize all layer
        self._initialize_weights()

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, input: torch.Tensor) -> torch.Tensor:
    
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = enn.GeometricTensor(input, self.input_type)
    
        # apply each equivariant block
    
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs is a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        out0 = self.g_conv1(x)
    
        out1 = out0.tensor
    
        out = self.trunk(out1)

        out = enn.GeometricTensor(out, self.input_type1)
        out2 = self.g_conv2(out)

        out = out0 + out2
    
    
        if self.upscale_factor == 2:
            out = self.upsampling0(out)
            out = self.upsampling1(out)
    
        
        
        out = self.conv3(out)

        out = self.conv4(out)
        
        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        out = out.tensor
        out = torch.clamp_(out, 0.0, 1.0)
    
        if out.shape[1] == 1:
            out = torch.cat((out, out, out), 1)
    
    
    
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    



class _G_P4M_ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_G_P4M_ResidualDenseBlock, self).__init__()
        # the model is equivariant under rotations by 90 degrees, modelled by C4
        self.r2_act = gspaces.flipRot2dOnR2(N=4)
        
        # the input image is a 3 or 1 scalar field, corresponding to the regular representation
        in_type = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        out_type = enn.FieldType(self.r2_act, growth_channels//8*[self.r2_act.regular_repr])
           
        # convolution 1
        self.block1 = enn.SequentialModule(
            # input size. (3) x 128 x 128
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=True),
            enn.ELU(out_type, inplace = True))    
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = enn.FieldType(self.r2_act, (channels + growth_channels * 1)//8*[self.r2_act.regular_repr])
        # the output type of the second convolution layer are 32 (growth_channel) regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, growth_channels//8*[self.r2_act.regular_repr])
        self.block2 = enn.SequentialModule(
            # input size. (3) x 128 x 128
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=True),
            enn.ELU(out_type, inplace = True))
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = enn.FieldType(self.r2_act, (channels + growth_channels * 2)//8*[self.r2_act.regular_repr])
        # the output type of the second convolution layer are 32 (growth_channel) regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, growth_channels//8*[self.r2_act.regular_repr])
        self.block3 = enn.SequentialModule(
            # input size. (3) x 128 x 128
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=True),
            enn.ELU(out_type, inplace = True))
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = enn.FieldType(self.r2_act, (channels + growth_channels * 3)//8*[self.r2_act.regular_repr])
        # the output type of the second convolution layer are 32 (growth_channel) regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, growth_channels//8*[self.r2_act.regular_repr])
        self.block4 = enn.SequentialModule(
            # input size. (3) x 128 x 128
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=True),
            enn.ELU(out_type, inplace = True))
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = enn.FieldType(self.r2_act, (channels + growth_channels * 4)//8*[self.r2_act.regular_repr])
        # the output type of the second convolution layer are 32 (growth_channel) regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        self.block5 = enn.SequentialModule(
            # input size. (3) x 128 x 128
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=True),
            enn.ELU(out_type, inplace = True))
        
        
        # the old output type is the input type to the next layer
        in_type = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        
        self.identity = enn.IdentityModule(in_type)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = enn.GeometricTensor(input, self.input_type)
        
        identity = x

        out1 = self.block1(x)
        
        cat = enn.tensor_directsum([x, out1])
        out2 = self.block2(cat)
        
        cat = enn.tensor_directsum([x, out1, out2])
        out3 = self.block3(cat)
        
        cat = enn.tensor_directsum([x, out1, out2, out3])
        out4 = self.block4(cat)
        
        cat = enn.tensor_directsum([x, out1, out2, out3, out4])
        out5 = self.block5(cat)
        
        out = out5 * 0.2
        
        out = out + identity
        
        return out


class _G_P4M_ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_G_P4M_ResidualResidualDenseBlock, self).__init__()
        
        # the model is equivariant under rotations by 90 degrees, modelled by C4
        self.r2_act = gspaces.flipRot2dOnR2(N=4)
        
        # the input image is a 3 or 1 scalar field, corresponding to the regular representation
        in_type = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        self.rdb1 = _G_P4M_ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = _G_P4M_ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = _G_P4M_ResidualDenseBlock(channels, growth_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # # wrap the input tensor in a GeometricTensor
        # # (associate it with the input type)
        x = enn.GeometricTensor(input, self.input_type)
        
        identity = x
        
        x = x.tensor
        
        out = self.rdb1(x)
        out = out.tensor
    
        out = self.rdb2(out)
        out = out.tensor
        
        out = self.rdb3(out)
        
        
        out = out * 0.2
        out = out + identity
        out = out.tensor
        return out


class G_P4M_RRDBNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
            growth_channels: int = 32,
            num_blocks: int = 23,
            upscale_factor: int = 4,
    ) -> None:
        
        super(G_P4M_RRDBNet, self).__init__()
        self.upscale_factor = upscale_factor
        
        # the model is equivariant under rotations by 90 degrees, modelled by C4
        self.r2_act = gspaces.flipRot2dOnR2(N=4)
        
        # the input image is a 3 or 1 scalar field, corresponding to the trivial representation
        in_type = enn.FieldType(self.r2_act, in_channels*[self.r2_act.trivial_repr])
        in_type1 = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        self.input_type1 = in_type1
        
        # The first layer of convolutional layer.
        # first specify the output type of the convolutional layer
        # we choose 64 feature fields, each transforming under the regular representation of C4
        out_type = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        
        self.g_conv1 = enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1)
                       
        # Feature extraction backbone network.
        # the old output type is the input type to the next layer
        in_type = self.g_conv1.out_type
        
        trunk = []
        for _ in range(num_blocks):
            trunk.append(_G_P4M_ResidualResidualDenseBlock(channels, growth_channels))
        
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.     
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        # the output type of the second convolution layer are 64 regular feature fields of C8
        out_type = enn.FieldType(self.r2_act, channels//8*[self.r2_act.regular_repr])
        
        self.g_conv2 = enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1)
        
        # Upsampling convolutional layer.
        if upscale_factor == 2:
            
            # the old output type is the input type to the next layer
            in_type = self.g_conv2.out_type
            
            
            self.upsampling0 = enn.R2Upsampling(in_type, scale_factor = 2, mode = "bilinear")
            
            
            # the output type of the Upsampling convolutional layer are 64 regular feature fields of C4
            out_type = enn.FieldType(self.r2_act, channels*[self.r2_act.trivial_repr])
            
            #in_type = self.enn.R2Upsampling(in_type, scale_factor = 2, mode = "bilinear")
            self.upsampling1 = enn.SequentialModule(
                enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1),
                enn.ELU(out_type, inplace=True)
            )
                
        if upscale_factor !=2:
            
            print("pas fais encore")
            

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.ELU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        # Initialize all layer
        self._initialize_weights()

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, input: torch.Tensor) -> torch.Tensor:
        
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = enn.GeometricTensor(input, self.input_type)
        
        # apply each equivariant block
        
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs is a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        out0 = self.g_conv1(x)
        
        out1 = out0.tensor
        
        out = self.trunk(out1)

        out = enn.GeometricTensor(out, self.input_type1)
        out2 = self.g_conv2(out)

        out = out0 + out2
        
        
        if self.upscale_factor == 2:
            out = self.upsampling0(out)
            out = self.upsampling1(out)
        
        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        
        out = out.tensor   
        out = self.conv3(out)

        out = self.conv4(out)
        
        out = torch.clamp_(out, 0.0, 1.0)
        
        if out.shape[1] == 1:
            out = torch.cat((out, out, out), 1)
        
        
        
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

 
class RRDBNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,  # tensor --> canneaux, longeur, largeur
            out_channels: int = 1,
            channels: int = 64,
            growth_channels: int = 32,
            num_blocks: int = 23,
            upscale_factor: int = 4,
    ) -> None:
        super(RRDBNet, self).__init__()
        self.upscale_factor = upscale_factor

        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network.
        trunk = []
        for _ in range(num_blocks):
            trunk.append(_ResidualResidualDenseBlock(channels, growth_channels))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer.
        if upscale_factor == 2:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.ELU(0.2, True)
            )
        if upscale_factor == 4:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.ELU(0.2, True)
            )
            self.upsampling2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.ELU(0.2, True)
            )
        if upscale_factor == 8:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.ELU(0.2, True)
            )
            self.upsampling2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.ELU(0.2, True)
            )
            self.upsampling3 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.ELU(0.2, True)
            )

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.ELU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        # Initialize all layer
        self._initialize_weights()

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
    
        out = self.trunk(out1)

        out2 = self.conv2(out)

        out = torch.add(out1, out2)

        if self.upscale_factor == 2:
            out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
        
        if self.upscale_factor == 4:
            out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
          
            out = self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))
          
        if self.upscale_factor == 8:
            out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
            out = self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))
            out = self.upsampling3(F.interpolate(out, scale_factor=2, mode="nearest"))


        out = self.conv3(out)

        out = self.conv4(out)

        out = torch.clamp_(out, 0.0, 1.0)
        
        if out.shape[1] == 1:
            out = torch.cat((out, out, out), 1)
     
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class _ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1))
        
        self.leaky_relu = nn.ELU(0.2)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        out1 = self.leaky_relu(self.conv1(x))

        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
 
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
    
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
       
        out = torch.mul(out5, 0.2)
        
        out = torch.add(out, identity)
        
        return out



class _ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = _ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)
        
        return out






    
    
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ELU(0.2, True),
            # state size. (64) x 64 x 64
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(0.2, True),
            # state size. (128) x 32 x 32
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(0.2, True),
            # state size. (256) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ELU(0.2, True),
            # state size. (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ELU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ELU(0.2, True),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ELU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), 
            nn.ELU(0.2, True),
            nn.Linear(100, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

class G_P4_Discriminator(nn.Module):
    def __init__(self) -> None:
        super(G_P4_Discriminator, self).__init__()
        
        # the model is equivariant under rotations by 90 degrees, modelled by C4
        self.r2_act = gspaces.rot2dOnR2(N=4)
        
        # the input image is a 3 scalar field, corresponding to the trivial representation
        in_type = enn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 64 feature fields, each transforming under the regular representation of C4
        out_type = enn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        
        self.block1 = enn.SequentialModule(
            # input size. (3) x 128 x 128
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=True),
            enn.ELU(out_type, inplace = True))
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 64 regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        
        self.block2 = enn.SequentialModule(        
            # state size. (64) x 64 x 64
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the second convolution layer are 128 regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        
        self.block3 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the second convolution layer are 128 regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        
        self.block4 = enn.SequentialModule(
            # state size. (128) x 32 x 32
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the second convolution layer are 256 regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, 256*[self.r2_act.regular_repr])
        
        self.block5 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
            
        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the second convolution layer are 256 regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, 256*[self.r2_act.regular_repr])
        
        self.block6 = enn.SequentialModule(
            # state size. (256) x 16 x 16
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 7
        # the old output type is the input type to the next layer
        in_type = self.block6.out_type
        # the output type of the second convolution layer are 512 regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, 512*[self.r2_act.regular_repr])

        self.block7 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 8
        # the old output type is the input type to the next layer
        in_type = self.block7.out_type
        # the output type of the second convolution layer are 512 regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, 512*[self.r2_act.regular_repr])
        
        self.block8 = enn.SequentialModule(
            # state size. (512) x 8 x 8
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 9
        # the old output type is the input type to the next layer
        in_type = self.block8.out_type
        # the output type of the second convolution layer are 512 regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, 512*[self.r2_act.regular_repr])
        self.block9 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 10
        # the old output type is the input type to the next layer
        in_type = self.block9.out_type
        # the output type of the second convolution layer are 512 regular feature fields of C4
        out_type = enn.FieldType(self.r2_act, 512*[self.r2_act.regular_repr])
        self.block10 = enn.SequentialModule(
            # state size. (512) x 4 x 4
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
                
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4*4, 100), 
            nn.ELU(0.2, True),
            nn.Linear(100, 1)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = enn.GeometricTensor(input, self.input_type)

        # apply each equivariant block
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)     
        x = self.block6(x)     
        x = self.block7(x)
        x = self.block8(x)      
        x = self.block9(x)
        x = self.block10(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        out = x.tensor
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out


   
class G_P4M_Discriminator(nn.Module):
    def __init__(self) -> None:
        super(G_P4M_Discriminator, self).__init__()
        
        # the model is equivariant under rotations  by 90 degrees and reflexion
        self.r2_act = gspaces.flipRot2dOnR2(N=4)
        
        # the input image is a 3 scalar field, corresponding to the trivial representation
        in_type = enn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 64 feature fields
        out_type = enn.FieldType(self.r2_act, 64//8*[self.r2_act.regular_repr])
        
        self.block1 = enn.SequentialModule(
            # input size. (3) x 128 x 128
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=True),
            enn.ELU(out_type, inplace = True))
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 64 regular feature fields
        out_type = enn.FieldType(self.r2_act, 64//8*[self.r2_act.regular_repr])
        
        self.block2 = enn.SequentialModule(        
            # state size. (64) x 64 x 64
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the second convolution layer are 128 regular feature fields
        out_type = enn.FieldType(self.r2_act, 128//8*[self.r2_act.regular_repr])
        
        self.block3 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the second convolution layer are 128 regular feature fields
        out_type = enn.FieldType(self.r2_act, 128//8*[self.r2_act.regular_repr])
        
        self.block4 = enn.SequentialModule(
            # state size. (128) x 32 x 32
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the second convolution layer are 256 regular feature fields
        out_type = enn.FieldType(self.r2_act, 256//8*[self.r2_act.regular_repr])
        
        self.block5 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
            
        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the second convolution layer are 256 regular feature fields
        out_type = enn.FieldType(self.r2_act, 256//8*[self.r2_act.regular_repr])
        
        self.block6 = enn.SequentialModule(
            # state size. (256) x 16 x 16
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 7
        # the old output type is the input type to the next layer
        in_type = self.block6.out_type
        # the output type of the second convolution layer are 512 regular feature fields
        out_type = enn.FieldType(self.r2_act, 512//8*[self.r2_act.regular_repr])

        self.block7 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 8
        # the old output type is the input type to the next layer
        in_type = self.block7.out_type
        # the output type of the second convolution layer are 512 regular feature fields
        out_type = enn.FieldType(self.r2_act, 512//8*[self.r2_act.regular_repr])
        
        self.block8 = enn.SequentialModule(
            # state size. (512) x 8 x 8
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 9
        # the old output type is the input type to the next layer
        in_type = self.block8.out_type
        # the output type of the second convolution layer are 512 regular feature fields
        out_type = enn.FieldType(self.r2_act, 512//8*[self.r2_act.regular_repr])
        self.block9 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 10
        # the old output type is the input type to the next layer
        in_type = self.block9.out_type
        # the output type of the second convolution layer are 512 regular feature fields
        out_type = enn.FieldType(self.r2_act, 512//8*[self.r2_act.regular_repr])
        self.block10 = enn.SequentialModule(
            # state size. (512) x 4 x 4
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
                
        self.classifier = nn.Sequential(
            nn.Linear(512//8*4*4*4*2, 100),  # *4 for rotation *2 for reflexion
            nn.ELU(0.2, True),
            nn.Linear(100, 1)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = enn.GeometricTensor(input, self.input_type)

        # apply each equivariant block
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)     
        x = self.block6(x)     
        x = self.block7(x)
        x = self.block8(x)      
        x = self.block9(x)
        x = self.block10(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        out = x.tensor
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out
  
class G_P8_Discriminator(nn.Module):
    def __init__(self) -> None:
        super(G_P8_Discriminator, self).__init__()
        
        # the model is equivariant under rotations  by 90 degrees and reflexion
        self.r2_act = gspaces.Rot2dOnR2(N=8)
        
        # the input image is a 3 scalar field, corresponding to the trivial representation
        in_type = enn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 64 feature fields
        out_type = enn.FieldType(self.r2_act, 64//8*[self.r2_act.regular_repr])
        
        self.block1 = enn.SequentialModule(
            # input size. (3) x 128 x 128
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=True),
            enn.ELU(out_type, inplace = True))
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 64 regular feature fields
        out_type = enn.FieldType(self.r2_act, 64//8*[self.r2_act.regular_repr])
        
        self.block2 = enn.SequentialModule(        
            # state size. (64) x 64 x 64
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the second convolution layer are 128 regular feature fields
        out_type = enn.FieldType(self.r2_act, 128//8*[self.r2_act.regular_repr])
        
        self.block3 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the second convolution layer are 128 regular feature fields
        out_type = enn.FieldType(self.r2_act, 128//8*[self.r2_act.regular_repr])
        
        self.block4 = enn.SequentialModule(
            # state size. (128) x 32 x 32
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the second convolution layer are 256 regular feature fields
        out_type = enn.FieldType(self.r2_act, 256//8*[self.r2_act.regular_repr])
        
        self.block5 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
            
        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the second convolution layer are 256 regular feature fields
        out_type = enn.FieldType(self.r2_act, 256//8*[self.r2_act.regular_repr])
        
        self.block6 = enn.SequentialModule(
            # state size. (256) x 16 x 16
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 7
        # the old output type is the input type to the next layer
        in_type = self.block6.out_type
        # the output type of the second convolution layer are 512 regular feature fields
        out_type = enn.FieldType(self.r2_act, 512//8*[self.r2_act.regular_repr])

        self.block7 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 8
        # the old output type is the input type to the next layer
        in_type = self.block7.out_type
        # the output type of the second convolution layer are 512 regular feature fields
        out_type = enn.FieldType(self.r2_act, 512//8*[self.r2_act.regular_repr])
        
        self.block8 = enn.SequentialModule(
            # state size. (512) x 8 x 8
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 9
        # the old output type is the input type to the next layer
        in_type = self.block8.out_type
        # the output type of the second convolution layer are 512 regular feature fields
        out_type = enn.FieldType(self.r2_act, 512//8*[self.r2_act.regular_repr])
        self.block9 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
        
        # convolution 10
        # the old output type is the input type to the next layer
        in_type = self.block9.out_type
        # the output type of the second convolution layer are 512 regular feature fields
        out_type = enn.FieldType(self.r2_act, 512//8*[self.r2_act.regular_repr])
        self.block10 = enn.SequentialModule(
            # state size. (512) x 4 x 4
            enn.R2Conv(in_type, out_type, kernel_size=4, stride=2, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace = True))
                
        self.classifier = nn.Sequential(
            nn.Linear(512//8*4*4*8, 100),  # *8 for rotation
            nn.ELU(0.2, True),
            nn.Linear(100, 1)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = enn.GeometricTensor(input, self.input_type)

        # apply each equivariant block
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)     
        x = self.block6(x)     
        x = self.block7(x)
        x = self.block8(x)      
        x = self.block9(x)
        x = self.block10(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        out = x.tensor
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out


class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(
            self,
            feature_model_extractor_node: str,
            feature_model_normalize_mean: list,
            feature_model_normalize_std: list
    ) -> None:
        super(ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_node = feature_model_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Extract the thirty-fifth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data.
        # This is the VGG model preprocessing method of the ImageNet dataset
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: torch.Tensor, gt_tensor: torch.Tensor) -> torch.Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
        gt_feature = self.feature_extractor(gt_tensor)[self.feature_model_extractor_node]

        # Find the feature map difference between the two images
        loss = F.l1_loss(sr_feature, gt_feature)

        return loss


def discriminator() -> Discriminator:
    model = Discriminator()

    return model

def g_p4_discriminator() -> Discriminator:
    model = G_P4_Discriminator()

    return model

def g_p4m_discriminator() -> Discriminator:
    model = G_P4M_Discriminator()

    return model

def g_p8_discriminator() -> Discriminator:
    model = G_P4M_Discriminator()

    return model

def rrdbnet_x1(**kwargs: Any) -> RRDBNet:
    model = RRDBNet(upscale_factor=1, **kwargs)

    return model


def rrdbnet_x2(**kwargs: Any) -> RRDBNet:
    model = RRDBNet(upscale_factor=2, **kwargs)

    return model

def g_p8_rrdbnet_x2(**kwargs: Any) -> RRDBNet:
    model = G_P8_RRDBNet(upscale_factor=2, **kwargs)

    return model

def g_p4m_rrdbnet_x2(**kwargs: Any) -> RRDBNet:
    model = G_P4M_RRDBNet(upscale_factor=2, **kwargs)

    return model

def rrdbnet_x4(**kwargs: Any) -> RRDBNet:
    model = RRDBNet(upscale_factor=4, **kwargs)

    return model


def rrdbnet_x8(**kwargs: Any) -> RRDBNet:
    model = RRDBNet(upscale_factor=8, **kwargs)

    return model


def content_loss(feature_model_extractor_node,
                 feature_model_normalize_mean,
                 feature_model_normalize_std) -> ContentLoss:
    content_loss = ContentLoss(feature_model_extractor_node,
                               feature_model_normalize_mean,
                               feature_model_normalize_std)

    return content_loss
