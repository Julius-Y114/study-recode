import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081))])
train_dataset = datasets.MNIST(root='./dataset/mnist/',
                               train=True,
                               download=False,
                               transform=transform)
train_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)

test_dataset = datasets.MNIST(root='./dataset/mnist/',
                               train=False,
                               download=False,
                               transform=transform)
test_loader = DataLoader(dataset=test_dataset,shuffle=True,batch_size=batch_size)

class ResidualBlock(torch.nn.Module):
    def __init__(self,channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        y =F.relu(self.conv1(x))
        y =self.conv2(y)
        return F.relu(y + x)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 5)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)

        self.rblock1 = ResidualBlock(channels=16)
        self.rblock2 = ResidualBlock(channels=32)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)

        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader,0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print(epoch,batch_idx,running_loss/300)
            running_loss = 0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))

if '__main__' == __name__:
    for epoch in range(10):
        train(epoch)
        test()




