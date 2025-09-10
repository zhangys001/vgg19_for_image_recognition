# coding=utf-8
import numpy as np
import struct
import os
import time

def show_matrix(mat, name):
    #print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass

def show_time(time, name):
    #print(name + str(time))
    pass

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride):
        # 卷积层的初始化
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))
    def init_param(self, std=0.01):  # 参数初始化
        if weight.shape == (self.channel_in, self.kernel_size, self.kernel_size, self.channel_out):
            print(f"Converting weight from {weight.shape} to [cout, cin, k, k]")
            weight = np.transpose(weight, (3, 0, 1, 2))  # → [cout, cin, k, k]
        else:
            raise ValueError(f"Weight shape mismatch: expected {(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out)}, got {weight.shape}")

        # 确保形状正确
        assert weight.shape == (self.channel_out, self.channel_in, self.kernel_size, self.kernel_size), \
            f"Got {weight.shape}, expected {(self.channel_out, self.channel_in, self.kernel_size, self.kernel_size)}"

        self.weight = weight
        self.bias = np.squeeze(self.bias)  # 去除冗余维度
    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input  # [N, C, H, W]
        N, C, H, W = self.input.shape
        kh, kw = self.kernel_size, self.kernel_size

        # 计算 padding 后的尺寸
        H_pad = H + 2 * self.padding
        W_pad = W + 2 * self.stride  # ❌ 你写的是 self.padding*2，但这里是 self.stride？不！是 padding！

        # 正确 padding
        self.input_pad = np.zeros((N, C, H_pad, W_pad))
        self.input_pad[:, :, self.padding:self.padding+H, self.padding:self.padding+W] = self.input

        # 输出尺寸：必须能整除
        H_out = (H_pad - kh) // self.stride + 1
        W_out = (W_pad - kw) // self.stride + 1
        self.output = np.zeros((N, self.channel_out, H_out, W_out))

        for n in range(N):
            for c_out in range(self.channel_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        # 计算感受野位置
                        h_start = h_out * self.stride
                        h_end = h_start + kh
                        w_start = w_out * self.stride
                        w_end = w_start + kw

                        # 提取输入区域 [cin, kh, kw]
                        input_region = self.input_pad[n, :, h_start:h_end, w_start:w_end]

                        # 卷积操作
                        # self.weight[c_out] shape: [cin, kh, kw]
                        self.output[n, c_out, h_out, w_out] = \
                            np.sum(input_region * self.weight[c_out, :, :, :]) + self.bias[c_out]

        return self.output
      

        return self.output
    
    def load_param(self, weight, bias):
        """
            weight: [kh, kw, cin, cout] -> 转为 [cout, cin, kh, kw]
        """
        if weight.shape == (self.kernel_size, self.kernel_size, self.channel_in, self.channel_out):
            print(f"Converting weight from {weight.shape} to [cout, cin, kh, kw]")
            weight = np.transpose(weight, (3, 2, 0, 1))  # [3,3,3,64] -> [64,3,3,3]
    
        assert weight.shape == (self.channel_out, self.channel_in, self.kernel_size, self.kernel_size), \
            f"Weight shape mismatch: got {weight.shape}, expected {(self.channel_out, self.channel_in, self.kernel_size, self.kernel_size)}"
    
        self.weight = weight
        self.bias = np.squeeze(bias)

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride):  # 最大池化层的初始化
        self.kernel_size = kernel_size
        self.stride = stride
        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))
    def forward_raw(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        
        # 确保输出维度计算正确
        height_out =int( (self.input.shape[2] - self.kernel_size) / self.stride + 1)
        width_out =int( (self.input.shape[3] - self.kernel_size) / self.stride + 1)
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # 计算最大池化层的前向传播，取池化窗口内的最大值
                        h_start = idxh * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = idxw * self.stride
                        w_end = w_start + self.kernel_size
                        self.output[idxn, idxc, idxh, idxw] = np.max(
                            self.input[idxn, idxc, h_start:h_end, w_start:w_end]
                        )
        return self.output
    
    def forward(self, input):
        return self.forward_raw(input)

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):  # 扁平化层的初始化
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))
    def forward(self, input):  # 前向传播的计算
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        show_matrix(self.output, 'flatten out ')
        return self.output
