import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GAPExtractor(BaseFeaturesExtractor):
    """
    Gated Attention Pooling (GAP) 提取器。
    它将对输入的 BEV 图像进行特征提取，并利用 Sigmoid 门控自动聚焦障碍物所在的白色栅格。
    """

    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        # SB3 的 FrameStack 会把通道数变成 4 (因为我们叠了4张图)
        n_input_channels = observation_space.shape[0]

        # 1. 基础 CNN 特征提取 (极其轻量)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # 2. 👑 门控注意力机制 (Gated Attention)
        self.attention_gate = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.Sigmoid()  # 输出 0~1 的权重，过滤无用背景
        )

        self.flatten = nn.Flatten()
        # 经过两次 stride=2 的卷积，64x64 会降维成 16x16
        self.linear = nn.Linear(32 * 16 * 16, features_dim)

    def forward(self, observations):
        # 提取基础特征
        features = self.cnn(observations)
        # 计算注意力热力图
        attention_weights = self.attention_gate(features)
        # GAP 核心数学操作：Element-wise 相乘，抑制背景，凸显障碍物！
        gated_features = features * attention_weights

        return self.linear(self.flatten(gated_features))