import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import albumentations as A
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary as infosummary
import contextlib

from config import *
from models import SmarterPoolNet
from utils.utils import get_images_and_labels, load_state, set_freeze_root_children, change_learning_rate
from datasets import ClassificationDataset
from trainers import ClassifcationTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


augments = A.Compose([
    A.D4(),
    A.AutoContrast(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
])

_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_NORM_MEANS, std=IMAGE_NORM_STD)
])


def main2():
    train_path = f'{DB_PATH}/train/'
    _, _, val_images, val_labels, _, _ = get_images_and_labels(train_path, limit_per_class=IMAGE_LIMIT_PER_CLASS, val_split=0.1, shuffle_seed=123, print_info=True)

    train_db = ClassificationDataset(None, None, IMAGE_SIZE, db_path_root=train_path, augments=augments, transforms=_transforms)
    val_db = ClassificationDataset(val_images, val_labels, IMAGE_SIZE, augments=None, transforms=_transforms)



    model = SmarterPoolNet(feature_dim=EMBED_DIM, num_classes=NUM_CLASSES).to(device)
    # infosummary(model, (1, 3, IMAGE_SIZE, IMAGE_SIZE), col_names=["input_size", "output_size", "num_params", "params_percent", "trainable"])
    optim = torch.optim.AdamW(model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    # loss_fn = RegularizedCrossentropy(temperature=0.5)

    trainer = ClassifcationTrainer(model, optim, loss_fn, NUM_CLASSES)
    set_freeze_root_children(model, 1, freeze=True)
    infosummary(model, (1, 3, IMAGE_SIZE, IMAGE_SIZE), col_names=["input_size", "output_size", "num_params", "params_percent", "trainable"])
    trainer.fit(20, train_db, val_db, BATCH_SIZE)

    change_learning_rate(optim, INIT_LEARNING_RATE / 5)
    set_freeze_root_children(model, 1, freeze=False)
    trainer.fit(NUM_EPOCHS, train_db, val_db, BATCH_SIZE)


    print('*'*30)
    print('Testing')
    print('*'*30)
    test_path = f'{DB_PATH}/test/'
    test_images, test_labels, _, _, _, _ = get_images_and_labels(test_path, limit_per_class=1206, val_split=0.0, print_info=True)

    test_db = ClassificationDataset(test_images, test_labels, IMAGE_SIZE, augments=None, transforms=_transforms)
    test_loader = DataLoader(test_db, batch_size=32, shuffle=False, num_workers=2)
    model, optim = load_state(f'./checkpoints/{model.__class__.__name__}.pt', model, optim)
    trainer.evaluate(test_loader, conf=True)

    with open('out.txt', 'w') as f:
        with contextlib.redirect_stdout(f):
            return trainer.evaluate(test_loader)

    os.system("sudo shutdown now")



if __name__ == '__main__':
    main2()