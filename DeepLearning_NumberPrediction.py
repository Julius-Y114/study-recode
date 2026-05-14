import torch
import torch.nn as nn
import pandas as pd

url = ".\\Samples\\train.csv"
raw_df = pd.read_csv(url)

# 标签：
label = raw_df['label'].values
# 特征：
raw_df = raw_df.drop(['label'],axis=1)
feature = raw_df.values

# 整个数据划分成2个数据集：训练集 测试集
train_feature = feature[:int(len(feature)*0.8)]
train_label = label[:int(len(label)*0.8)]
test_feature = feature[int(len(feature)*0.8):]
test_label = label[int(len(label)*0.8):]

train_feature = torch.tensor(train_feature).to(torch.float)
train_label = torch.tensor(train_label)
test_feature = torch.tensor(test_feature).to(torch.float)
test_label = torch.tensor(test_label)

# 神经网络：黑盒子
# 784个像素点构成的灰度图-->函数-->10个概率(0,1,2,3,4,5,6,7,8,9)
# 定义网络结构:[1*784]*[784*444]-->[1*444]*[444*512]-->[1*512]*[512*512]-->[1*512]*[512*10]-->[1*10]
#     输入层： in_channel 784 out_channel 444
#     隐藏层1：in_channel 444 out_channel 512
#     隐藏层2：in_channel 512 out_channel 512
#     输出层： in_channel 512 out_channel 10
model = nn.Sequential(
    nn.Linear(784,444),
    nn.ReLU(),
    nn.Linear(444, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
    nn.ReLU(),
    nn.Softmax()
)

# 损失函数
lossfunction = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.0001)

# 训练轮数：200次
for i in range(200):
    # 清空优化器梯度(偏导)
    optimizer.zero_grad()
    # 训练模型
    predict = model(train_feature)
    result = torch.argmax(predict,axis=1)
    train_acc = torch.mean((result==train_label).to(torch.float))
    # 计算损失值
    loss = lossfunction(predict,train_label)
    loss.backward()
    optimizer.step()
    print('train loss is {}, train acc is {}'.format(loss.item(),train_acc.item()))

    # 用测试集检查模型
    optimizer.zero_grad()
    predict = model(test_feature)
    result = torch.argmax(predict,axis=1)
    test_acc = torch.mean((result==test_label).to(torch.float))
    loss = lossfunction(predict,test_label)
    print('test loss is {}, test acc is {}'.format(loss.item(), test_acc.item()))

#存储训练好的模型
# torch.save(model.state_dict(),'./mymodel.pt')

#加载模型
# params = torch.load('./mymodel.pt')
# model.load_state_dict(params)
#
# new_test_data = test_feature[100:120]
# new_test_label = test_label[100:120]
# predict = model(new_test_data)
# result = torch.argmax(predict,axis=1)
# print(new_test_label)
# print(result)



