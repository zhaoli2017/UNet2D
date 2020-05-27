import torch
import torch.nn as nn

def get_dice(pred, organ_target, num_class):
    # the organ_target should be one-hot code
    assert len(pred.shape) == len(organ_target.shape), 'the organ_target should be one-hot code'
    dice = 0
    for organ_index in range(1, num_class):
        P = pred[:, organ_index, :, :]
        _P = 1 - pred[:, organ_index, :, :]
        G = organ_target[:, organ_index, :, :]
        _G = 1 - organ_target[:, organ_index, :, :]
        mulPG = (P * G).sum(dim=1).sum(dim=1)
        mul_PG = (_P * G).sum(dim=1).sum(dim=1)
        mulP_G = (P * _G).sum(dim=1).sum(dim=1)

        dice += (mulPG + 1) / (mulPG + 0.8 * mul_PG + 0.2 * mulP_G + 1)
    return dice/(num_class-1)

class DiceLoss(nn.Module):
    def __init__(self, num_class=3):
        super().__init__()
        self.num_class = num_class
        self.num_organ = self.num_class - 1

    def forward(self, pred, target):
        shape = target.shape
        organ_target = torch.zeros((target.size(0), self.num_organ + 1, shape[-2], shape[-1]))

        for organ_index in range(self.num_class):

            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index, :, :] = temp_target

        organ_target = organ_target.cuda()

        return 1-get_dice(pred, organ_target, self.num_class).mean()



