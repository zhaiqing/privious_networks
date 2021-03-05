import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

transform = transforms.Compose(
    #图像缩放到32×32形状
    [transforms.Resize((32,32)),
     #转化为Tensor
     transforms.ToTensor(),
     #标准化处理
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net=LeNet()
#载入权重文件
net.load_state_dict(torch.load('Lenet.pth'))

#载入图像
im=Image.open('1.jpg')
#图像预处理
im=transform(im)  #[C,H,W]
#在最前面增加一个新的维度batch
im=torch.unsqueeze(im,dim=0)  #[N,C,H,W]

with torch.no_grad():
    outputs=net(im)
    predict=torch.max(outputs,dim=1)[1].data.numpy()
    # predict = torch.softmax(outputs,dim=1)
print(classes[int(predict)])
# print(predict)