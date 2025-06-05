from glob import glob
import cv2
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np
import os


def show_tensor_image(tensor, title=None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Displays a single PyTorch tensor image using matplotlib.

    Args:
        tensor (torch.Tensor): Image tensor of shape (C, H, W) or (1, C, H, W).
        title (str, optional): Title to display above the image.
        mean (tuple): Mean used in normalization (for unnormalization).
        std (tuple): Std used in normalization (for unnormalization).
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension

    # Move to CPU and convert to numpy
    image = tensor.detach().cpu().numpy()

    # Unnormalize
    for i in range(image.shape[0]):
        image[i] = image[i] * std[i] + mean[i]

    # Clamp and transpose to (H, W, C)
    image = np.clip(image, 0, 1)
    image = np.transpose(image, (1, 2, 0))

    # Show image
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()


def change_learning_rate(optimizer: torch.optim.Optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def set_freeze_root_children(model: torch.nn.Module, n: int, freeze: bool = True):
    """
    Freezes or unfreezes the first `n` root-level children modules of the model.

    Args:
        model (torch.nn.Module): The model to modify.
        n (int): Number of top-level modules to affect.
        freeze (bool): If True, freeze the modules; if False, unfreeze them.
    """
    children = list(model.children())
    
    for i, child in enumerate(children):
        if i < n:
            for param in child.parameters():
                param.requires_grad = not freeze



def read_image(image_path: str, size:int=256):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))
    return image


def save_state(save_path: str, model: nn.Module, optim: torch.optim.Optimizer, info: dict | None = None):
    state_dict = {
        'model_state': model.state_dict(),
        'optim_state': optim.state_dict(),
        'info': info
    }
    try:
        os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
        torch.save(state_dict, save_path)
        print(f'State saved.')
        # print('Info:', info)
    except Exception as e:
        print(f'ERROR: {e}')


def load_state(path: str, model: nn.Module=None, optim: torch.optim.Optimizer=None)-> tuple[nn.Module, torch.optim.Optimizer]:
    obj = torch.load(path)
    if model is not None:
        model.load_state_dict(obj['model_state'])
        print('Model State Loaded')
    if optim is not None:
        optim.load_state_dict(obj['optim_state'])
        print('Optimizer State Loaded')
    print(f'Loaded state.')
    return model, optim


def denormalize(image, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return image * std + mean


def get_images_and_labels(db_path, limit_per_class=20, val_split=0.2, shuffle_seed=224, random_state=224, print_info=False):
    classes = sorted(glob(f'{db_path}/*'))
    images = sorted(glob(f'{classes}/*'))

    train_images_list = []
    train_labels_list = []
    val_images_list = []
    val_labels_list = []

    for _class in classes:
        random.seed(random_state)
        images = sorted(glob(f'{_class}/*'))
        random.shuffle(images)
        labels = [int(image.split('/')[-2]) for image in images]
        val_count = int((val_split) * len(labels[:limit_per_class]))
        
        val_images_list += images[:val_count]
        val_labels_list += labels[:val_count]

        random.seed(shuffle_seed)
        images = images[val_count:]
        labels = labels[val_count:]
        random.shuffle(images)

        train_images_list += images[:limit_per_class]
        train_labels_list += labels[:limit_per_class]

    if print_info:
        print('Train Images:', len(train_images_list), 'Val Images:', len(val_images_list))
    return train_images_list, train_labels_list, val_images_list, val_labels_list