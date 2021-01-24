import torch
from torchvision import transforms
from PIL import Image


from model import LeNet

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

net = LeNet()
net.load_state_dict(torch.load('./Lenet.pth'))


def main():
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    img = Image.open('3.jpg')
    img = transform(img)  # 3*32*32
    img = torch.unsqueeze(img, dim=0)  # 增加batch，原图权值是[16, 3, 5, 5]是4维的

    with torch.no_grad():
        output = net(img)
        predict = torch.max(output, dim=1)[1].data.numpy()
        print("Predicted: ", classes[int(predict)])


if __name__ == '__main__':
    main()

















