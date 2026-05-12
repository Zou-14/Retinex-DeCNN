import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==================== Swin Transformer组件 ====================
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim=32, num_heads=2, window_size=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        input_H, input_W = H, W
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W
        x = x.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)
        x = self.norm1(x)
        x = self.attn(x)
        x = x.reshape(B, Hp, Wp, C).permute(0, 3, 1, 2)
        x = x[:, :, :input_H, :input_W]
        x = shortcut + x
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp(x_flat)
        x = x + x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


# ==================== 改进的分解网络 ====================
class ImprovedDecomNet(nn.Module):
    def __init__(self, channel=32, kernel_size=3, num_heads=2, window_size=4):
        super().__init__()
        self.conv1 = nn.Conv2d(4, channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, padding_mode='replicate')
        self.relu2 = nn.ReLU()
        self.swin_block = SwinTransformerBlock(dim=channel, num_heads=num_heads, window_size=window_size)
        self.recon = nn.Conv2d(channel, 4, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        _, _, H, W = input_img.shape
        x = self.relu1(self.conv1(input_img))
        x = self.relu2(self.conv2(x))
        x = self.swin_block(x)
        if x.shape[2] != H or x.shape[3] != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        outs = self.recon(x)
        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L


# ==================== DnCNN去噪网络（无BatchNorm）====================
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_layers=7):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(channels, 32, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, channels, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.dncnn(x)
        return x - residual


# ==================== 照度调整网络 ====================
class RelightNet(nn.Module):
    def __init__(self, channel=32, kernel_size=3):
        super(RelightNet, self).__init__()
        self.relu = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(4, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_deconv1_1 = nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_deconv1_2 = nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_deconv1_3 = nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_fusion = nn.Conv2d(channel*3, channel, kernel_size=1, padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)
        out0 = self.relu(self.net2_conv0_1(input_img))
        out1 = self.relu(self.net2_conv1_1(out0))
        out2 = self.relu(self.net2_conv1_2(out1))
        out3 = self.relu(self.net2_conv1_3(out2))
        out3_up = F.interpolate(out3, size=out2.shape[2:], mode='bilinear', align_corners=False)
        deconv1 = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up = F.interpolate(deconv1, size=out1.shape[2:], mode='bilinear', align_corners=False)
        deconv2 = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up = F.interpolate(deconv2, size=out0.shape[2:], mode='bilinear', align_corners=False)
        deconv3 = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))
        deconv1_rs = F.interpolate(deconv1, size=input_R.shape[2:], mode='bilinear', align_corners=False)
        deconv2_rs = F.interpolate(deconv2, size=input_R.shape[2:], mode='bilinear', align_corners=False)
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus = self.net2_fusion(feats_all)
        output = self.net2_output(feats_fus)
        return torch.sigmoid(output)


# ==================== 改进版完整模型 ====================
class ImprovedRetinexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.DecomNet = ImprovedDecomNet()
        self.RelightNet = RelightNet()
        self.DnCNN = DnCNN(channels=3, num_layers=7)

    def forward(self, input_low, input_high=None):
        if torch.is_tensor(input_low):
            device = input_low.device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_low = torch.from_numpy(input_low).float().to(device)
        if input_high is not None and not torch.is_tensor(input_high):
            input_high = torch.from_numpy(input_high).float().to(device)

        R_low, I_low = self.DecomNet(input_low)
        if input_high is not None:
            R_high, I_high = self.DecomNet(input_high)

        R_low_denoised = self.DnCNN(R_low)
        I_delta = self.RelightNet(I_low, R_low_denoised)
        I_delta = torch.clamp(I_delta, 0.5, 1.0)  # 强制照度图至少0.5

        if R_low_denoised.shape[2:] != I_delta.shape[2:]:
            I_delta = F.interpolate(I_delta, size=R_low_denoised.shape[2:], mode='bilinear', align_corners=False)

        I_delta_3 = torch.cat([I_delta] * 3, dim=1)
        enhanced_img = R_low_denoised * I_delta_3
        enhanced_img = torch.clamp(enhanced_img, 0.0, 1.0)

        self.output_R_low = R_low_denoised.detach().cpu()
        self.output_I_low = torch.cat([I_low] * 3, dim=1).detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_S = enhanced_img.detach().cpu()

        if input_high is not None and self.training:
            return self._compute_losses(R_low, I_low, I_delta, R_low_denoised)
        return enhanced_img

    def _compute_losses(self, R_low, I_low, I_delta, R_low_denoised):
        # 反射图去噪损失
        denoise_loss = F.l1_loss(R_low_denoised, R_low.detach())
        
        # 照度图平滑损失
        smooth_loss = self.smooth(I_delta, R_low_denoised)
        
        # ✅ 照度图亮度损失 - 目标0.9，不是1.0！
        target_I = torch.ones_like(I_delta) * 0.9
        brightness_loss = F.l1_loss(I_delta, target_I)
        
        # 分解损失（简化）
        loss_Decom = F.l1_loss(R_low * torch.cat([I_low] * 3, dim=1), 
                              torch.cat([I_low] * 3, dim=1)) * 0.1
        
        # 照度调整损失
        loss_Relight = brightness_loss + 0.1 * smooth_loss + 0.01 * denoise_loss
        
        return loss_Decom, loss_Relight

    def smooth(self, I, R):
        R_gray = 0.299 * R[:, 0, :, :] + 0.587 * R[:, 1, :, :] + 0.114 * R[:, 2, :, :]
        R_gray = torch.unsqueeze(R_gray, dim=1)
        grad_I_x = self.gradient(I, 'x')
        grad_I_y = self.gradient(I, 'y')
        grad_R_x = self.ave_gradient(R_gray, 'x')
        grad_R_y = self.ave_gradient(R_gray, 'y')
        return torch.mean(grad_I_x * torch.exp(-10 * grad_R_x) + grad_I_y * torch.exp(-10 * grad_R_y))

    def gradient(self, x, direction):
        if direction == 'x':
            kernel = torch.FloatTensor([[0, 0], [-1, 1]]).view(1, 1, 2, 2).to(x.device)
        else:
            kernel = torch.FloatTensor([[0, -1], [0, 1]]).view(1, 1, 2, 2).to(x.device)
        kernel = kernel.repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, kernel, stride=1, padding=1, groups=x.shape[1])

    def ave_gradient(self, x, direction):
        grad = self.gradient(x, direction)
        return F.avg_pool2d(grad, kernel_size=3, stride=1, padding=1)