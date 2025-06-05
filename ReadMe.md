# Knowledge Distillation in PyTorch

This repository provides a PyTorch-based implementation of knowledge distillation, proposed by Hinton et al. (2015), facilitating the transfer of knowledge from a larger, pre-trained teacher model to a smaller student model. I have created this repository from the ground up for educational purposes.

## Features

* **Modular Architecture**: Organized into distinct modules for datasets, models, trainers, and utilities, promoting clarity and ease of maintenance.
* **Configurable Training**: Utilizes a `config.py` file to manage hyperparameters and training settings, allowing for flexible experimentation.
* **Comprehensive Training Pipeline**: Includes a `main.py` script that orchestrates the training process, integrating all components seamlessly.

## Project Structure

```

Knowledge-Distillation-Pytorch/
├── datasets/        # Data loading and preprocessing scripts
├── models/          # Definitions of teacher and student models
├── trainers/        # Training loops and distillation logic
├── utils/           # Utility functions (e.g., logging, metrics)
├── config.py        # Configuration settings for training
├── main.py          # Entry point for training the models
└── .gitignore       # Specifies files to ignore in version control
```

## Getting Started

### Prerequisites

* Python 3.6 or higher
* PyTorch 1.7 or higher

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ahanaf019/Knowledge-Distillation-Pytorch.git
   cd Knowledge-Distillation-Pytorch
   ```
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

*Note: If `requirements.txt` is not provided, manually install necessary packages such as `torch`, `torchvision`, etc.*

### Usage

1. **Configure Training Parameters**:

   Edit the `config.py` file to set your desired training parameters, including learning rates, batch sizes, and dataset paths.
2. **Start Training**:

   Run the training script:

   ```bash
   python main.py
   ```

This will initiate the training process using the configurations specified in `config.py`.

## Acknowledgments

This implementation is inspired by-

* Hinton et al. (2018), [&#34;Distilling the Knowledge in a Neural Network&#34;](https://arxiv.org/abs/1503.02531)
