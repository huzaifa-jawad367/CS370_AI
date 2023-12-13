import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms import functional as F
import torch
from vision.datasets.voc_dataset import VOCDataset

from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.config import mobilenetv1_ssd_config

from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.ssd import MatchPrior
from vision.utils.misc import Timer, freeze_net_layers, store_labels

import logging

create_net = create_mobilenetv1_ssd
config = mobilenetv1_ssd_config
config.set_image_size(300)

# create data transforms for train/test/val
train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
target_transform = MatchPrior(config.priors, config.center_variance,
                            config.size_variance, 0.5)


test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)



# Load data from data/landing_pad folder as VOCDataset
dataset = VOCDataset('data/landingpad_resize', transform=train_transform, target_transform=target_transform, is_test=True)
label_file = os.path.join("labels.txt")
store_labels(label_file, dataset.class_names)
num_classes = len(dataset.class_names)

train_dataset = ConcatDataset([dataset])
train_loader = DataLoader(train_dataset, 9,
                        num_workers=3,
                        shuffle=True)

for images, boxes, labels in train_loader:
    img = F.to_pil_image(images[0]) 
    # BGR to RGB
    # img = img.convert('RGB')
    img.save('test.png')

    break