import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GAPExtractor(BaseFeaturesExtractor):
    """
    Gated Attention Pooling (GAP) 特征提取器
    专为极度干净的 BEV (Bird's Eye View) 图像设计
    """
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        # 因为在 train_bev.py 中堆叠了 4 帧，输入通道数为 4
        n_input_channels = observation_space.shape[0] 
        
        # 1. 极简 CNN (剥离环境噪音后，不再需要庞大的 ResNet)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # 2. 门控注意力机制 (Sigmoid 过滤无用特征)
        self.attention_gate = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.Sigmoid() 
        )
        
        self.flatten = nn.Flatten()
        # 64x64 经过两次 stride=2 池化变为 16x16
        self.linear = nn.Linear(32 * 16 * 16, features_dim)

    def forward(self, observations):
        features = self.cnn(observations)
        attention_weights = self.attention_gate(features)
        # 核心：将注意力权重作用于基础特征
        gated_features = features * attention_weights 
        return self.linear(self.flatten(gated_features))