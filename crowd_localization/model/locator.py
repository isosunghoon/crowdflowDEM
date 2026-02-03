from model.HR_Net.seg_hrnet import get_seg_model
from model.VGG.VGG16_FPN import VGG16_FPN
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from model.PBM import BinarizedModule
from torchvision import models

class Crowd_locator(nn.Module):
    def __init__(self, net_name, pretrained=True):
        super(Crowd_locator, self).__init__()

        if net_name == 'HR_Net':
            self.Extractor = get_seg_model()
            self.Binar = BinarizedModule(input_channels=720)
        if net_name == 'VGG16_FPN':
            self.Extractor = VGG16_FPN()
            self.Binar = BinarizedModule(input_channels=768)

        self.loss_BCE = nn.BCELoss()

    @property
    def loss(self):
        return self.head_map_loss, self.binar_map_loss

    def forward(self, img, mask_gt, mode='train'):
        feature, pre_map = self.Extractor(img)
        threshold_matrix, binar_map = self.Binar(feature, pre_map)

        if mode == 'train':
            self.binar_map_loss = (torch.abs(binar_map - mask_gt)).mean()
            self.head_map_loss = F.mse_loss(pre_map, mask_gt)

        return threshold_matrix, pre_map, binar_map

    def test_forward(self, img):
        feature, pre_map = self.Extractor(img)
        return feature, pre_map