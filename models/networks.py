from torch import nn
import torch
import torch.nn.functional as F

class Conv_3d_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
            padding=3, norm='none', activation="relu", pad_type="reflect") -> None:
        super().__init__()
        self.pad_type = pad_type
        self.padding = padding
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU()
        else:
            self.activation = None
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride,
                bias=True)

    def forward(self, x):
        x = F.pad(x, [self.padding]*6, self.pad_type)
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, norm="none", activation="relu",
            pad_type="reflect") -> None:
        super().__init__()
        layers = []
        layers += [Conv_3d_Block(channels, channels, 3, 1, 1, norm, activation,
            pad_type)]
        layers += [Conv_3d_Block(channels, channels, 3, 1, 1, norm,
            activation="none", pad_type=pad_type)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.model(x)

class ResidualBlocks(nn.Module):
    def __init__(self, n_blocks, channels, norm="none", activation="relu",
            pad_type="reflect") -> None:
        super().__init__()
        layers = []
        for _ in range(n_blocks):
            layers += [ResidualBlock(channels, norm, activation, pad_type)]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class StyleEncoder(nn.Module):
    def __init__(self, in_channels=14, grid_size=24, n_downsample=4,
            style_dim=8, bottom_dim=32) -> None:
        super().__init__()
        self.model = []
        self.model += [Conv_3d_Block(in_channels, bottom_dim, 7, 1)]
        for _ in range(2):
            self.model += [Conv_3d_Block(bottom_dim, 2*bottom_dim, 4, 2, 1)]
            bottom_dim *= 2
        for _ in range(n_downsample - 2):
            self.model += [Conv_3d_Block(bottom_dim, bottom_dim, 4, 2, 1)]
        self.model += [nn.AdaptiveAvgPool3d(1)]
        self.model += [nn.Conv3d(bottom_dim, style_dim, 1, 1, 0)]
        self.final_model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.final_model(x)

class ContentEncoder(nn.Module):
    def __init__(self, in_channels=14, n_downsample=2, bottom_dim=32, n_res=4) -> None:
        super().__init__()
        layers = []
        layers += [Conv_3d_Block(in_channels, bottom_dim, 7, 1)]
        for _ in range(n_downsample):
            layers += [Conv_3d_Block(bottom_dim, 2*bottom_dim, 4, 2)]
            bottom_dim *= 2
        layers += [ResidualBlocks(n_res, bottom_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

