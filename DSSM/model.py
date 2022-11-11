# -*- coding: utf-8 -*-

"""
@author: WUNNAN

@Created on: 2022/9/29 9:09
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSSM(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_classes=2, device='cpu'):
        super(DSSM, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size+1, embedding_size)

        self.fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(embedding_size, 300),
            nn.Tanh(),
            nn.Linear(300, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh()
        )
        self.classification = nn.Sequential(
            nn.BatchNorm1d(256*2),
            nn.Linear(256*2, 256),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, p1, p2):
        # 对于p1
        p1 = self.embedding(p1).sum(1)
        p1 = self.fc(p1)

        # 对于p2
        p2 = self.embedding(p2).sum(1)
        p2 = self.fc(p2)

        # 拼接, 按行并排
        merged_vec = torch.cat([p1, p2], dim=1)
        # 分类
        logits = self.classification(merged_vec)
        pro = F.softmax(logits, dim=-1)
        return logits, pro

    # 初始化权重参数 uniform distribution
    # torch.nn.init.xavier_uniform_满足条件
    def _init_weights(self):
        for layer in self.modules():
            # 为线性层时，权重初始化
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
