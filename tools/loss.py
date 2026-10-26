import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

try:
    import lpips
except ImportError:
    lpips = None


class PerceptualLoss(torch.nn.modules.loss._Loss):

    def __init__(self, pixel_loss=1.0, l1_loss=False, style_loss=0.0, lambda_feat=1, include_vgg_layers=('1', '2', '3', '4', '5'), in_channels=3):
        super(PerceptualLoss, self).__init__(True, True)

        # download pretrained vgg19 if necessary and instantiate it
        self.channel_pre = torch.nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=(1, 1), stride=(1, 1))

        vgg19 = torchvision.models.vgg.vgg19(pretrained=True)
        self.vgg_layers = vgg19.features

        # the vgg feature layers we want to use for the perceptual loss
        self.layer_name_mapping = {
        }
        if '1' in include_vgg_layers:
            self.layer_name_mapping['3'] = "conv1_2"
        if '2' in include_vgg_layers:
            self.layer_name_mapping['8'] = "conv2_2"
        if '3' in include_vgg_layers:
            self.layer_name_mapping['13'] = "conv3_2"
        if '4' in include_vgg_layers:
            self.layer_name_mapping['22'] = "conv4_2"
        if '5' in include_vgg_layers:
            self.layer_name_mapping['31'] = "conv5_2"

        # weights for pixel loss and style loss (feature loss assumed 1.0)
        self.pixel_loss = pixel_loss
        self.l1_loss = l1_loss
        self.lambda_feat = lambda_feat
        self.style_loss = style_loss

    def forward(self, input, target):
        input = self.channel_pre(input)
        target = self.channel_pre(target)

        lossValue = torch.tensor(0.0).to(input.device)
        l2_loss_func = lambda ipt, tgt: torch.sum(torch.pow(ipt - tgt, 2))  # amplitude to intensity
        l1_loss_func = lambda ipt, tgt: torch.sum(torch.abs(ipt - tgt))  # amplitude to intensity

        # get size
        s = input.size()

        # number of tensors in this mini batch
        num_images = s[0]

        # L2 loss  (L1 originally)
        if self.l1_loss:
            scale = s[1] * s[2] * s[3]
            lossValue += l1_loss_func(input, target) * (2 * self.pixel_loss / scale)
            loss_func = l2_loss_func
        elif self.pixel_loss:
            scale = s[1] * s[2] * s[3]
            lossValue += l2_loss_func(input, target) * (2 * self.pixel_loss / scale)
            loss_func = l2_loss_func

        # stack input and output so we can feed-forward it through vgg19
        x = torch.cat((input, target), 0)

        for name, module in self.vgg_layers._modules.items():

            # run x through current module
            x = module(x)
            s = x.size()

            # scale factor
            scale = s[1] * s[2] * s[3]

            if name in self.layer_name_mapping:
                a, b = torch.split(x, num_images, 0)
                lossValue += self.lambda_feat * loss_func(a, b) / scale

                # Gram matrix for style loss
                if self.style_loss:
                    A = a.reshape(num_images, s[1], -1)
                    B = b.reshape(num_images, s[1], -1).detach()

                    G_A = A @ torch.transpose(A, 1, 2)
                    del A
                    G_B = B @ torch.transpose(B, 1, 2)
                    del B

                    lossValue += loss_func(G_A, G_B) * (self.style_loss / scale)

        return lossValue


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

        pad = self.window_size // 2
        mu_x = F.avg_pool2d(pred, self.window_size, stride=1, padding=pad)
        mu_y = F.avg_pool2d(target, self.window_size, stride=1, padding=pad)

        mu_x_sq = mu_x.square()
        mu_y_sq = mu_y.square()
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.avg_pool2d(pred * pred, self.window_size, stride=1, padding=pad) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(target * target, self.window_size, stride=1, padding=pad) - mu_y_sq
        sigma_xy = F.avg_pool2d(pred * target, self.window_size, stride=1, padding=pad) - mu_xy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-8
        )
        ssim_value = ssim_map.mean()
        return 1.0 - ssim_value


class LPIPSLoss(nn.Module):
    def __init__(self, net: str = "alex"):
        super().__init__()
        if lpips is None:
            raise ImportError("lpips is not installed but LPIPSLoss was requested")
        self.model = lpips.LPIPS(net=net)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred_3ch = pred.clamp(0.0, 1.0).repeat(1, 3, 1, 1) * 2.0 - 1.0
        target_3ch = target.clamp(0.0, 1.0).repeat(1, 3, 1, 1) * 2.0 - 1.0
        return self.model(pred_3ch, target_3ch).mean()
