import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.1, temperature=10):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

        self.student_loss_fn = nn.CrossEntropyLoss()
        self.div_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, teacher_preds, student_preds, labels):
        student_loss = self.student_loss_fn(student_preds, labels)
        distillation_loss = self.div_loss(
            F.log_softmax(student_preds / self.temperature, dim=1),
            F.softmax(teacher_preds / self.temperature, dim=1),
        ) * (self.temperature**2)
        return self.alpha * distillation_loss + (1 - self.alpha) * student_loss