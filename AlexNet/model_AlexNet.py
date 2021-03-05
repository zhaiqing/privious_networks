import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 数据集比较小，为了加快训练速度，卷积核的个数只取一半
        self.features = nn.Sequential(
            # 卷积过程中N不是整数时，会舍弃小数部分，下面padding=2一样的效果
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            # inplace=True 对从上层网络Conv2d中传递下来的tensor直接进行修改，
            # 这样能够节省运算内存，不用多存储其他变量
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        # 初始化权重函数
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #从第一维开始，batch不要动[batch,channel,height,width]
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 返回一个迭代器，迭代遍历每一个神经网络
        for m in self.modules():
            # 是否属于Conv2d类别
            if isinstance(m, nn.Conv2d):
                # 初始化变量方法kaiming_normal_
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            #如果传进来的是全连接层，使用正态分布进行赋值，bias=0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
