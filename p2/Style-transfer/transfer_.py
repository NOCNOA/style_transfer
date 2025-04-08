# -*- coding:utf-8 -*-

import torch #pytorch       
from PIL import Image #PIL库
from torch import nn #torch.nn
from torchvision import transforms, models #torchvision
import os  # 添加在文件开头的import部分
from torch import optim #torch.optim
import torch.nn.functional as F

# 定义图像大小
img_size = 512

# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#标准化
transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# 加载图片
def load_img(img_path):
    """加载并预处理图像"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))  # 现在 img_size 已定义
    img = transforms.ToTensor()(img)
    img = transform(img).unsqueeze(0)
    return img


#显示图片
def show_img(tensor):
    image = tensor.cpu().clone()#克隆
    image = image.squeeze(0)#降维
    return image


# 构建vgg特征提取网络
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()#初始化

        self.select = ['0', '5', '10', '19', '28']#选择卷积层
        self.vgg = models.vgg19(pretrained=True).features  # .features用于提取卷积层

    def forward(self, x):#前向传播
        features = []#特征
        for name, layer in self.vgg._modules.items():
            x = layer(x)  # name为第几层的序列号，layer就是卷积层,,x为输入的图片。x = layer(x)的意思是，x经过layer层卷积后再赋值给x
            if name in self.select:
                features.append(x)
        return features


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc = nn.Linear(512 * 4 * 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = x.view(-1, 512 * 4 * 4)
        x = self.sigmoid(self.fc(x))
        return x

def gan_loss(discriminator, real_images, generated_images):
    # 计算GAN损失
    real_labels = torch.ones(real_images.size(0), 1).to(real_images.device)
    fake_labels = torch.zeros(generated_images.size(0), 1).to(generated_images.device)
    
    # 判别器对真实图像的判断
    real_output = discriminator(real_images)
    # 判别器对生成图像的判断
    fake_output = discriminator(generated_images)
    
    # 计算生成器的损失
    generator_loss = F.binary_cross_entropy(fake_output, real_labels)
    
    return generator_loss

def style_transfer(content_path, style_path, total_step=2000, callback=None, 
                  progress_callback=None, stop_flag=None):#风格迁移函数
    """
    执行风格迁移
    
    参数:
        content_path: 内容图像路径
        style_path: 风格图像路径
        total_step: 迭代次数
        callback: 回调函数，用于更新GUI界面
        progress_callback: 进度更新回调函数
        stop_flag: 停止检查函数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载图像
    content_img = load_img(content_path).to(device)
    style_img = load_img(style_path).to(device)
    target = content_img.clone().requires_grad_(True)
    
    # 初始化模型
    vgg = VGGNet().to(device)
    for param in vgg.parameters():
        param.requires_grad = False
    
    # 设置优化器
    optimizer = optim.Adam([target], lr=0.003)
    
    # 设置权重
    content_weight = 1
    style_weight = 100
    
    # 训练循环
    for step in range(total_step):
        # 检查是否需要停止
        if stop_flag and stop_flag():
            print("处理已停止")
            break
            
        target_features = vgg(target)
        content_features = vgg(content_img)
        style_features = vgg(style_img)
        
        content_loss = torch.mean((target_features[3] - content_features[3]) ** 2)
        
        style_loss = 0
        for f1, f2 in zip(target_features, style_features):
            _, c, h, w = f1.size()
            f1 = f1.view(c, h * w)
            f2 = f2.view(c, h * w)
            f1 = torch.mm(f1, f1.t())
            f2 = torch.mm(f2, f2.t())
            style_loss += torch.mean((f1 - f2) ** 2) / (c * h * w)
        
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if progress_callback:
            progress_callback(step + 1)
            
        if (step + 1) % 100 == 0:
            print(f'Step [{step+1}/{total_step}]')
            
        if callback and (step + 1) % 500 == 0:
            callback(target)
    
    return target

class StyleTransfer:
    def __init__(self):
        self.discriminator = Discriminator()
        self.discriminator.to(device)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
    
    def train(self, content_image, style_image, num_steps):
        # ... existing code ...
        # 在计算总损失时添加GAN损失
        content_loss = content_weight * mse_loss(content_features[0], target_features[0])
        style_loss = style_weight * style_loss(style_features, target_features)
        gan_loss_value = gan_loss(self.discriminator, content_image, target)
        
        total_loss = content_loss + style_loss + 0.1 * gan_loss_value  # 0.1是GAN损失的权重
        # ... existing code ...

