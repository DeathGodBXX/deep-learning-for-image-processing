import torch
import torchvision
import torchvision.transforms as transforms
from model import LeNet
from torch import nn
from torch import optim


def main():
    # 下载图片同时进行的预处理
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # 下载数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)

    # 加载数据集,设置一次处理36个数据集
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=36,
                                              shuffle=True, num_workers=0)

    # 下载测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)

    # 加载数据集，批次处理5000个图片数据
    testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                             shuffle=True, num_workers=0)

    # 使用gpu训练,将测试集移动到gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    testiter = iter(testloader)
    test_image, test_label = testiter.next()
    test_image = test_image.to(device)
    test_label = test_label.to(device)

    # 类别
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')

    # 训练模型移动到gpu上
    net = LeNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):
        """每次epoch，50000/36约等于1000，每次训练结果就是500+500，500打印一次"""

        running_loss = 0.0
        for step, data in enumerate(trainloader, start=0):
            """一次处理36个图片，枚举遍历所有下载的trainset"""

            # data is a list of [inputs,labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the optimizer grad
            optimizer.zero_grad()

            # forward + loss + backward + optimizer
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 500 == 499:
                with torch.no_grad():
                    outputs = net(test_image)
                    # test_result = torch.max(outputs, dim=1)[1]
                    test_result = torch.max(torch.softmax(outputs, dim=1), dim=1)[1]
                    accuracy = torch.eq(test_result, test_label).sum().item() / test_label.size(0)
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print("Finished Training!!!")

    torch.save(net.state_dict(), './Lenet.pth')


if __name__ == '__main__':
    main()
