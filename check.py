from PIL import Image
import numpy as np
import torch
from model_improved import ImprovedRetinexNet

# 检查输入图片
img = Image.open('./data/test/low/1.bmp').convert('RGB')
img_arr = np.array(img)
print(f'输入图片范围: [{img_arr.min()}, {img_arr.max()}]')
print(f'输入图片形状: {img_arr.shape}')

# 检查模型输出
model = ImprovedRetinexNet()
model.eval()
checkpoint = torch.load('./ckpts_improved/model_epoch3.pth', map_location='cpu')
model.load_state_dict(checkpoint)

img = img.resize((128, 120), Image.LANCZOS)
img = np.array(img, dtype=np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)
input_tensor = torch.from_numpy(img).float()

with torch.no_grad():
    enhanced = model(input_tensor)
    print(f'模型输出范围: min={enhanced.min():.4f}, max={enhanced.max():.4f}, mean={enhanced.mean():.4f}')