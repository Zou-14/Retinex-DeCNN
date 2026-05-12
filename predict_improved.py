import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
from glob import glob
import numpy as np
import torch
from PIL import Image
from model_improved import ImprovedRetinexNet
import gc

parser = argparse.ArgumentParser(description='Predict with Improved RetinexNet')
parser.add_argument('--data_dir', dest='data_dir', default='./data/test/low/')
parser.add_argument('--ckpt_path', dest='ckpt_path', default='./ckpts_improved/model_epoch3.pth')
parser.add_argument('--res_dir', dest='res_dir', default='./results_improved/')
args = parser.parse_args()

def predict():
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    model = ImprovedRetinexNet().to(device)
    model.eval()
    
    if os.path.exists(args.ckpt_path):
        print(f"加载模型: {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print('模型加载成功！')
    else:
        print(f'错误: 找不到模型文件 {args.ckpt_path}')
        return
    
    test_images = glob(args.data_dir + '/*.*')
    test_images.sort()
    test_images = test_images[:1]  # 只处理第一张
    
    print(f'找到 {len(test_images)} 张测试图像')
    
    if len(test_images) == 0:
        print(f"错误: {args.data_dir} 中没有图片文件")
        return
    
    with torch.no_grad():
        for i, img_path in enumerate(test_images):
            img_name = os.path.basename(img_path)
            print(f'处理 [{i+1}/{len(test_images)}]: {img_name}')
            
            try:
                # 读取原图
                original = Image.open(img_path).convert('RGB')
                original_np = np.array(original)
                h, w = original_np.shape[:2]
                
                # 创建缩略图用于模型输入 - 关键修复：从256改成128
                img_small = original.copy()
                max_size = 128  # 🚨 改成128
                if img_small.width > max_size or img_small.height > max_size:
                    img_small.thumbnail((max_size, max_size), Image.LANCZOS)
                    print(f'  ⚠️ 已缩放到 {img_small.width}x{img_small.height}')
                
                # 预处理
                img_array = np.array(img_small, dtype=np.float32) / 255.0
                img_array = np.transpose(img_array, (2, 0, 1))
                img_array = np.expand_dims(img_array, axis=0)
                input_tensor = torch.from_numpy(img_array).float().to(device)
                
                # 增强
                enhanced = model(input_tensor)
                
                # 转回 numpy
                enhanced = enhanced.squeeze().cpu().numpy()
                enhanced = np.transpose(enhanced, (1, 2, 0))
                
                print(f'  📊 模型输出范围: [{enhanced.min():.4f}, {enhanced.max():.4f}]')
                
                # 强制拉伸亮度到 0-1 范围
                enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-6)
                enhanced = np.clip(enhanced, 0, 1)
                
                # 转为 uint8
                enhanced = (enhanced * 255).astype(np.uint8)
                
                # 放大到原图尺寸
                enhanced_resized = np.array(Image.fromarray(enhanced).resize((w, h), Image.LANCZOS))
                
                # 保存增强图
                out_path = f'{args.res_dir}/{img_name}'
                Image.fromarray(enhanced_resized).save(out_path)
                print(f'  ✅ 已保存增强图: {out_path}')
                
                # 保存对比图
                compare = np.zeros((h, w*2, 3), dtype=np.uint8)
                compare[:, :w] = original_np
                compare[:, w:] = enhanced_resized
                compare_path = f'{args.res_dir}/compare_{img_name}'
                Image.fromarray(compare).save(compare_path)
                print(f'  ✅ 已保存对比图: {compare_path}')
                
                gc.collect()
                
            except Exception as e:
                print(f'  ❌ 处理失败: {e}')
                import traceback
                traceback.print_exc()
    
    print(f'\n结果已保存到: {args.res_dir}')

if __name__ == '__main__':
    predict()