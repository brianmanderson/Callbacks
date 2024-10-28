import torch.nn as nn
import torch.nn.functional as F


class MeanDSC(nn.Module):
    def __init__(self, num_classes, apply_softmax=False):
        super(MeanDSC, self).__init__()
        self.num_classes = num_classes
        self.eps = 1e-6
        self.apply_softmax = apply_softmax

    def forward(self, y_true, y_pred):
        if self.apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true.argmax(dim=1), num_classes=self.num_classes)
        y_pred = F.one_hot(y_pred.argmax(dim=1), num_classes=self.num_classes)

        y_true = y_true.reshape(-1, self.num_classes)
        y_pred = y_pred.reshape(-1, self.num_classes)

        intersection = (y_true * y_pred).sum(dim=0)
        union = y_true.sum(dim=0) + y_pred.sum(dim=0) - intersection

        iou = (intersection + self.eps) / (union + self.eps)
        mean_dsc = 2 * iou / (1 + iou)

        # Ignore background (assume background is class 0)
        mean_dsc = mean_dsc[1:].mean()

        return mean_dsc