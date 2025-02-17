import torch
import torch.nn.functional as F


class Inception(torch.nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels = in_channels, out_channels = 16, kernel_size = 1)
        self.branch5x5_1 = torch.nn.Conv2d(in_channels = in_channels, out_channels = 16, kernel_size = 1)
        self.branch5x5_2 = torch.nn.Conv2d(in_channels = 16, out_channels = 24, kernel_size = 5, padding = 2)
        self.branch3x3dbl_1 = torch.nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 1)
        self.branch3x3dbl_2 = torch.nn.Conv2d(in_channels = 16, out_channels = 24, kernel_size = 3, padding = 1)
        self.branch3x3dbl_3 = torch.nn.Conv2d(in_channels = 24, out_channels = 24, kernel_size = 3, padding = 1)
        self.branch_pool = torch.nn.Conv2d(in_channels = in_channels, out_channels = 24, kernel_size = 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size = 3, stride = 1, padding = 1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


in_channels = 64

model = Inception(in_channels = in_channels)


def DisplayLayer():
    print(model.branch1x1, '\n', model.branch5x5_1, '\n', model.branch5x5_2, '\n', model.branch3x3dbl_1, '\n',
          model.branch3x3dbl_2, '\n', model.branch3x3dbl_3, '\n', model.branch_pool)


if __name__ == '__main__':
    DisplayLayer()