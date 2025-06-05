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
from utils.utils import get_images_and_labels, load_state, set_freeze_root_children, change_learning_rate
from datasets import ClassificationDataset
from trainers import ClassifcationTrainer, Distiller
from models import ResNetModel, MobileNetModel
from utils.losses import DistillationLoss

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



def main():
    # 1. Train teacher model from scratch
    model = ResNetModel(num_classes=NUM_CLASSES).to(device)
    infosummary(model, (1, 3, IMAGE_SIZE, IMAGE_SIZE), col_names=["input_size", "output_size", "num_params", "params_percent", "trainable"])

    # Configs loaded from config.py
    train_model(model, checkpoint_path='checkpoints/teacher.pt')


    # 2. Train student model from scratch
    model = MobileNetModel(num_classes=NUM_CLASSES).to(device)
    infosummary(model, (1, 3, IMAGE_SIZE, IMAGE_SIZE), col_names=["input_size", "output_size", "num_params", "params_percent", "trainable"])

    # Configs loaded from config.py
    train_model(model, checkpoint_path='checkpoints/student.pt')

    # 3. Distill Teacher weights into student
    teacher_model = ResNetModel(num_classes=NUM_CLASSES).to(device)
    teacher_model, _ = load_state('checkpoints/teacher.pt', teacher_model, None)
    student_model = MobileNetModel(num_classes=NUM_CLASSES).to(device)
    infosummary(student_model, (1, 3, IMAGE_SIZE, IMAGE_SIZE), col_names=["input_size", "output_size", "num_params", "params_percent", "trainable"])
    distill(teacher_model, student_model)



def train_model(model, checkpoint_path):
    train_path = f'{DB_PATH}/train/'
    train_images, train_labels, val_images, val_labels = get_images_and_labels(train_path, limit_per_class=IMAGE_LIMIT_PER_CLASS, val_split=0.1, shuffle_seed=123, print_info=True)
    
    train_db = ClassificationDataset(train_images, train_labels, IMAGE_SIZE, db_path_root=train_path, augments=augments, transforms=_transforms)
    val_db = ClassificationDataset(val_images, val_labels, IMAGE_SIZE, augments=None, transforms=_transforms)


    
    optim = torch.optim.AdamW(model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainer = ClassifcationTrainer(model, optim, loss_fn, NUM_CLASSES, device=device)
    trainer.fit(NUM_EPOCHS, train_db, val_db, BATCH_SIZE, checkpoint_path=checkpoint_path)


    print('*'*30)
    print('Testing')
    print('*'*30)
    test_path = f'{DB_PATH}/test/'
    test_images, test_labels, _, _ = get_images_and_labels(test_path, limit_per_class=100000, val_split=0.0, print_info=True)

    test_db = ClassificationDataset(test_images, test_labels, IMAGE_SIZE, augments=None, transforms=_transforms)
    test_loader = DataLoader(test_db, batch_size=32, shuffle=False, num_workers=2)
    model, optim = load_state(f'./checkpoints/student.pt', model, optim)
    trainer.evaluate(test_loader, conf=False)


def distill(teacher_model, student_model):
    train_path = f'{DB_PATH}/train/'
    train_images, train_labels, val_images, val_labels = get_images_and_labels(train_path, limit_per_class=IMAGE_LIMIT_PER_CLASS, val_split=0.1, shuffle_seed=123, print_info=True)
    
    train_db = ClassificationDataset(train_images, train_labels, IMAGE_SIZE, db_path_root=train_path, augments=augments, transforms=_transforms)
    val_db = ClassificationDataset(val_images, val_labels, IMAGE_SIZE, augments=None, transforms=_transforms)

    
    optim = torch.optim.AdamW(student_model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=1e-2)
    loss_fn = DistillationLoss()

    distiller = Distiller(
        teacher=teacher_model,
        student=student_model, 
        optim=optim, 
        loss_fn=loss_fn,
        eval_loss_fn=nn.CrossEntropyLoss(),
        num_classes=NUM_CLASSES, 
        device=device
    )
    distiller.fit(NUM_EPOCHS, train_db, val_db, BATCH_SIZE, checkpoint_path='checkpoints/distilled.pt')

    print('*'*30)
    print('Testing')
    print('*'*30)
    test_path = f'{DB_PATH}/test/'
    test_images, test_labels, _, _ = get_images_and_labels(test_path, limit_per_class=100000, val_split=0.0, print_info=True)

    test_db = ClassificationDataset(test_images, test_labels, IMAGE_SIZE, augments=None, transforms=_transforms)
    test_loader = DataLoader(test_db, batch_size=32, shuffle=False, num_workers=2)
    student_model, optim = load_state(f'./checkpoints/distilled.pt', student_model, optim)
    distiller.evaluate(test_loader, conf=False)


if __name__ == '__main__':
    main()