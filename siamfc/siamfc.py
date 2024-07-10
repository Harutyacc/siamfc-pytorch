from __future__ import absolute_import, division, print_function  # 保证兼容性，浮点数除法，print为函数

# 外部库
import torch  # Pytorch主库，包含张量操作和自动微分
import torch.nn as nn  # 包含了用于构建神经网络的模块
import torch.nn.functional as F  # 包含了常用的函数，激活函数和损失函数等
import torch.optim as optim  # 包含了优化器，用于更新神经网络的权重
import numpy as np  # 用于科学计算的库，擅长处理多维数组
import time  
import cv2  # opencv的接口，用于图像处理
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR  # 包含学习率调度器，用于训练中调整学习率
from torch.utils.data import DataLoader  # 包含了一些工具，例如数据加载器
from got10k.trackers import Tracker  #got-10k数据集跟踪器接口

# 自定义模块
from . import ops  # 包含一些常用的操作函数
from .backbones import AlexNetV1  # 主干网络，用于特征提取
from .heads import SiamFC  # 头部网络，用于匹配模板和搜索图像
from .losses import BalancedLoss  # 损失函数，用来训练网络
from .datasets import Pair  # 数据集，加载图像对
from .transforms import SiamFCTransforms  # 数据转换类，用于预处理图像


__all__ = ['TrackerSiamFC']


class Net(nn.Module):
    
    # 初始化网络结构
    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    # 定义前向传播
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    # 初始化追踪器
    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)  # 继承父类Tracker,Ture表示确定性参数，方便复现
        self.cfg = self.parse_args(**kwargs)

        # 设置计算设备，GPU或CPU
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # 初始化模型
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)
        
        # 加载预训练模型权重，如果提供了的话
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        
        # 移动模型到计算设备上
        self.net = self.net.to(self.device)

        # 设置损失函数
        self.criterion = BalancedLoss()

        # 设置优化器，随机梯度下降
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # 设置学习率调度器，指数递减
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    # 解析配置参数
    def parse_args(self, **kwargs):
        # 默认参数字典
        cfg = {
            # basic parameters（基本参数）
            'out_scale': 0.001,  # 输出缩放比例
            'exemplar_sz': 127,  # 模板图像的大小，模板图像适用于跟踪的目标图像
            'instance_sz': 255,  # 搜索图像大小，搜索图像是包含目标的较大区域图像
            'context': 0.5,  # 上下文比例，控制模板图像周围上下文区域的大小，上下文有助于提供更多的背景信息，从而提高追踪的准确性
            # inference parameters（推理参数）
            'scale_num': 3,  # 尺度参数，跟踪过程中，算法会在不同尺度下搜索目标，以应对目标大小的变化
            'scale_step': 1.0375,  # 尺度步长，控制每个尺度之间的间隔比例，此处表示每个尺度之间的比例差为3.75% 
            'scale_lr': 0.59,  # 尺度学习率，控制跟踪过程中的尺度更新速度
            'scale_penalty': 0.9745,  # 尺度惩罚，惩罚尺度变化，放置尺度过大或过小地变化
            'window_influence': 0.176,  # 窗口影响，用于控制响应图中的平滑窗口对最终结果的影响
            'response_sz': 17,  # 响应图用来衡量模板图像与搜索图像之间的匹配程度，通过提取特征然后特征匹配生成，这是响应图大小
            'response_up': 16,  # 响应图上采样比例，用来提高响应图的分辨率，获得更精确的目标位置
            'total_stride': 8,  # 总步幅，控制网络层之间的步幅，影响特征图的分解
            # train parameters
            'epoch_num': 50,  # 训练总轮数
            'batch_size': 8,  # 批次大小，指每次训练所使用的样本数
            'num_workers': 32,  # 数据加载线程数，用于加速数据加载过程
            'initial_lr': 1e-2,  # 初始化学习率，用于控制训练初期的学习速度
            'ultimate_lr': 1e-5,  # 最终学习率，用于控制训练后期的学习速度
            'weight_decay': 5e-4,  # 权重衰减，用于正则化，放置过拟合
            'momentum': 0.9,  # 动量，用于加速收敛，稳定训练过程
            'r_pos': 16,   # 正样本半径，控制正样本区域大小，正样本区域内的目标位置被认为是正确的
            'r_neg': 0  # 负样本半径，控制负样本区域的大小，负样本区域内的目标位置被认为是不正确的
            } 
        
        # 查询有无需要更新的参数并更新
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
                
        # 返回一个namedtuple类型对象，便于访问配置参数
        return namedtuple('Config', cfg.keys())(**cfg)
    
    # 初始化跟踪器
    @torch.no_grad()  # 禁用梯度计算，减小内存消耗
    def init(self, img, box):
        
        # 设置网络为评估模式
        self.net.eval()

        # 将输入的边界框转换为以0为索引，并基于中心的格式即[y,x,h,w],中心点坐标，高和宽 
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # 计算Hanning窗口，用于加权响应图
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # 计算搜索尺度因子，用于多尺度搜索
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # 计算模板和搜索图像的尺寸
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # 提取模板图像特征，并将其存储在self.kernel中
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)
    
    # 更新过程
    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        
        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        return boxes, times
    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels
