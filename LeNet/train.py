import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#50000张训练图片
train_set=torchvision.datasets.CIFAR10(root='./data',train=True,
                                      download=False,transform=transform)

train_loader=torch.utils.data.DataLoader(train_set,batch_size=36,
                                         shuffle=False,num_workers=0)

#10000张验证图片
val_set=torchvision.datasets.CIFAR10(root='./data',train=False,
                                     download=False,transform=transform)

val_loader=torch.utils.data.DataLoader(val_set,batch_size=10000,
                                       shuffle=False,num_workers=0)

#转化成为迭代器
val_data_iter=iter(val_loader)
#获取数据测试图像以及标签
val_image,val_label=val_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net=LeNet()
loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=0.001)

for epoch in range(5):    # 多次遍历数据集
    running_loss=0.0    #累加损失
    for step,data in enumerate(train_loader,start=0):
        #get the inputs:data is a list of [inputs,labels]
        inputs,labels=data

        #参数梯度清零
        optimizer.zero_grad()

        #forward + backward + optimize
        outputs=net(inputs)
        loss=loss_function(outputs,labels)
        loss.backward()
        #update
        optimizer.step()

        running_loss+=loss.item()
        if step % 500 ==499:
            #不要计算误差损失梯度
            with torch.no_grad():
                outputs=net(val_image)   #[batch,10]
                predict_y=torch.max(outputs,dim=1)[1]
                accuracy=(predict_y == val_label).sum().item() / val_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')

#生成模型权重文件
save_path='./Lenet.pth'
torch.save(net.state_dict(),save_path)





