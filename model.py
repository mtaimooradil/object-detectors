# -----------------------------------------------------
# Written by Muhammad Taimoor Adil on 13/07/2023
# 
# This file contains the basic YOLOv3 model
# -----------------------------------------------------


import config
import torch
import torch.nn as nn


class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bn_act = True, **kwargs): # bn_act -> batch normalization and activation     
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = not bn_act, **kwargs) # you only need bias when batch norm and activation are not applied
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(negative_slope = 0.1) 
        self.bn_act = bn_act
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):  
        x = self.leaky(self.batch_norm(self.conv(x))) if self.bn_act else self.conv(x)
        return x # expression is not returned in single statement for debugging purposes


class ResidualBlock(nn.Module):

    def __init__(self, channels, use_residual = True, num_repeats = 1):     
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()  
        
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size = 1),
                    CNNBlock(channels // 2, channels, kernel_size = 3, padding = 1)
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + self.use_residual * x 
        return x
        

class ScalePrediction(nn.Module):

    def __init__(self, channels, num_classes, anchors_per_scale = 3):
        super(ScalePrediction, self).__init__()
        self.pred = nn.Sequential(
            CNNBlock(channels, channels * 2, kernel_size = 3, padding = 1),
            # 5 for [p,x,y,w,h] -> object score and bounding box parameters
            CNNBlock(channels * 2, anchors_per_scale * (num_classes + 5), bn_act = False, kernel_size = 1) 
        )
        self.anchors_per_scale = anchors_per_scale
        self.num_classes = num_classes

    def forward(self, x):
        return ( 
            self.pred(x)
            .reshape(x.shape[0], self.anchors_per_scale, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        ) # shape (batch_size, anchors_per_scale, grid_cell_size(x,y), num_classes + 5)
    

class YOLOv3(nn.Module):

    def __init__(self, in_channels = 3, num_classes = 20):
        super(YOLOv3, self).__init__()    
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_layers()

    def forward(self, x):
        outputs = [] # outputs at three different scales
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                # This is done because we want to compute loss for each of the predictions seperately.
                outputs.append(layer(x))
                continue
        
            x = layer(x)
            
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim = 1)
                route_connections.pop()  
    
        return outputs

    def _create_layers(self):
        architecture = config.YOLO_ARCHITECTURE
        layers = nn.ModuleList()
        in_channels = self.in_channels
        
        for module in architecture:    
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = 1 if kernel_size == 3 else 0
                    )
                )
                in_channels = out_channels # in channels for next layer are out channels of previous layer

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(
                        in_channels,
                        num_repeats = num_repeats 
                    )
                )
                in_channels = in_channels # channels remain same after residual block

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual = False, num_repeats = 1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size = 1),
                        ScalePrediction(in_channels // 2, num_classes = self.num_classes)
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(
                        nn.Upsample(scale_factor = 2)
                    )

                    # Input channels are tripled after upsampling layer due to the route that will be concatenated
                    # in the forward propagation that has twice as many channels as the output from the upsampling layer.
                    # There is a reason for this, the concatentation works out in a way that it concatenates along the 
                    # dimension of channels so after the concatenaion the out channels become sum of out channels of 
                    # corresponding residual block and out channels of upsampling layer. The way channel dimensions are
                    # designed it natuurally comes as triple the output channels after upsampling layer. And we need to
                    # set this as in channels for the next conv layer
                    
                    in_channels = in_channels * 3

        return layers




   
    

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")
            
