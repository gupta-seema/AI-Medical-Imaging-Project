import torch
import torch.nn as nn
import torchvision
from chestray_labels import NUM_CLASSES

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.shape

        # Channel attention
        avg = self.avg_pool(x).view(b, c)
        mx  = self.max_pool(x).view(b, c)
        ch  = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x   = x * ch

        # Spatial attention
        avg_map = x.mean(1, keepdim=True)
        max_map, _ = x.max(1, keepdim=True)
        sp  = self.spatial(torch.cat([avg_map, max_map], dim=1))
        return x * sp


class ChestRayNet(nn.Module):
    def __init__(self, backbone='efficientnet_b0', pretrained=True, use_cbam=True, num_classes=NUM_CLASSES):
        super().__init__()
        if backbone == 'efficientnet_b0':
            eff = torchvision.models.efficientnet_b0(
                weights=(torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            )
        elif backbone == 'efficientnet_b2':
            eff = torchvision.models.efficientnet_b2(
                weights=(torchvision.models.EfficientNet_B2_Weights.DEFAULT if pretrained else None)
            )
        elif backbone == 'efficientnet_b4':
            eff = torchvision.models.efficientnet_b4(
                weights=(torchvision.models.EfficientNet_B4_Weights.DEFAULT if pretrained else None)
            )
        else:
            raise ValueError('Unsupported backbone')

        self.features = eff.features                      # [B, C, H, W]
        self.pool = eff.avgpool                           # global avg pool -> [B, C, 1, 1]
        in_ch = eff.classifier[1].in_features             # EfficientNet classifier in-features
        self.cbam = CBAM(in_ch) if use_cbam else nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_ch, num_classes))

    def forward(self, x):
        # Backbone features
        x = self.features(x)              # [B, C, H, W]

        # >>> Apply CBAM BEFORE pooling so spatial attention has effect
        x = self.cbam(x)                  # [B, C, H, W]

        # Global pooling and classification head
        x = self.pool(x)                  # [B, C, 1, 1]
        x = torch.flatten(x, 1)           # [B, C]
        return self.head(x)               # [B, num_classes]

    def gradcam_target_layer(self):
        # Last feature block is a good CAM target
        return self.features[-1]
