#coding=utf-8
import torch
from head import AdaFace

# typical inputs with 512 dimension
B = 5
embbedings = torch.randn((B, 512)).float()  # latent code
norms = torch.norm(embbedings, 2, -1, keepdim=True)  # 二范数
normalized_embedding  = embbedings / norms # 输入x的归一化之后的结果
labels =  torch.randint(70722, (B,)) # 随机选取索引为目标类别 shape [5,]

# instantiate AdaFace
adaface = AdaFace(embedding_size=512, # 输入图片的特征维度
                  classnum=70722, # 类别数量
                  m=0.4, # 图片质量指标的系数
                  h=0.333, # 用于取到 68%
                  s=64., # 缩放系数
                  t_alpha=0.01,)  # ema 指数滑动平均

# calculate loss
cosine_with_margin = adaface(normalized_embedding, norms, labels) # 归一化之后的输入 图片的质量norm 标签
loss = torch.nn.CrossEntropyLoss()(cosine_with_margin, labels)
print(loss)
