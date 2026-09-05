import torch
import torch.nn as nn
import torch.optim as optim
class AuditoryAttentionCNN3(nn.Module):
    def __init__(self, samples=256,n_chans=64):
        super(AuditoryAttentionCNN3, self).__init__()

        # Define the parameters
        # samples, n_chans = inp_size  # Extract sample size and channels from inp_size
        filter_width = round(0.130 * 64)  # fs = 128 (hardcoded as in the original model)
        n_filters = 2

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=n_chans, out_channels=n_filters, kernel_size=(filter_width, 1), stride=(1, 1), padding=(filter_width-1, 0))
        self.relu1 = nn.ReLU()

        # Pooling layer
        self.pool = nn.AvgPool2d(kernel_size=(samples, 1), stride=(samples, 1))

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=5, kernel_size=(1, 1), stride=(1, 1))

        # Sigmoid layer
        self.sigmoid = nn.Sigmoid()

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=2, kernel_size=(1, 1), stride=(1, 1))

        # Loss layer is implicitly handled by the loss function in PyTorch
        
    def forward(self, x):
        # First convolution + ReLU
        x = x.unsqueeze(3)  # 扩展一个维度，变为 (batch_size, 64, T, 1)

        x = self.conv1(x)
        x = self.relu1(x)
        
        # Pooling layer
        x = self.pool(x)
        
        # Second convolution + Sigmoid
        x = self.conv2(x)
        x = self.sigmoid(x)
        
        # Third convolution
        x = self.conv3(x)
        
        # Flatten the output to apply loss
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        return x
    
class AuditoryAttentionCNN2(nn.Module):
    def __init__(self, samples=256,n_chans=64):
        super(AuditoryAttentionCNN2, self).__init__()

        # Define the parameters
        # samples, n_chans = inp_size  # Extract sample size and channels from inp_size
        filter_width = round(0.130 * 128)  # fs = 128 (hardcoded as in the original model)
        n_filters = 5

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=n_chans, out_channels=n_filters, kernel_size=(filter_width, 1), stride=(1, 1), padding=(filter_width-1, 0))
        self.relu1 = nn.ReLU()

        # Pooling layer
        self.pool = nn.AvgPool2d(kernel_size=(samples, 1), stride=(samples, 1))

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=5, kernel_size=(1, 1), stride=(1, 1))

        # Sigmoid layer
        self.sigmoid = nn.Sigmoid()

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=2, kernel_size=(1, 1), stride=(1, 1))

        # Loss layer is implicitly handled by the loss function in PyTorch
        
    def forward(self, x):
        # First convolution + ReLU
        x = x.unsqueeze(3)  # 扩展一个维度，变为 (batch_size, 64, T, 1)

        x = self.conv1(x)
        x = self.relu1(x)
        
        # Pooling layer
        x = self.pool(x)
        
        # Second convolution + Sigmoid
        x = self.conv2(x)
        x = self.sigmoid(x)
        
        # Third convolution
        x = self.conv3(x)
        
        # Flatten the output to apply loss
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        return x


class AuditoryAttentionCNN(nn.Module):
    def __init__(self, input_channels=64, time_steps=128, num_classes=2):
        super(AuditoryAttentionCNN, self).__init__()
        
        # 卷积层: 5个独立的64x17滤波器
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=5, kernel_size=(17, 1), stride=(1, 1))
        
        # 平均池化层: 对时间维度进行池化
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出 1x1 的特征图
        
        # 全连接层: 5个神经元
        self.fc1 = nn.Linear(5, 5)
        
        # 第二个全连接层: 2个输出神经元 (二分类)
        self.fc2 = nn.Linear(5, num_classes)
        
        # Sigmoid 激活函数用于输出层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, 64, T)，我们需要处理成 (batch_size, 64, T, 1) 输入卷积层
        x = x.unsqueeze(3)  # 扩展一个维度，变为 (batch_size, 64, T, 1)

        # 卷积层
        x = self.conv1(x)  # 输出大小 (batch_size, 5, T-16, 1) 因为卷积核大小为17
        
        # ReLU 激活
        x = torch.relu(x)
        
        # 平均池化层
        x = self.pool(x)  # 输出大小 (batch_size, 5, 1, 1)，即每个滤波器的输出在时间维度上平均

        # 拉平特征图
        x = x.view(x.size(0), -1)  # 展平为 (batch_size, 5)

        # 第一个全连接层
        x = torch.relu(self.fc1(x))  # 输出大小 (batch_size, 5)

        # 第二个全连接层
        x = self.fc2(x)  # 输出大小 (batch_size, num_classes)

        # Sigmoid 输出层
        x = self.sigmoid(x)  # 输出形状 (batch_size, num_classes)

        return x

if __name__ == "__main__":
    model = AuditoryAttentionCNN(input_channels=64, time_steps=128, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(model)

