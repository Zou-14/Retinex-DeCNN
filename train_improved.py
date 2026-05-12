import os
import argparse
import random
from glob import glob
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from model_improved import ImprovedRetinexNet

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.000003)
parser.add_argument('--ckpt_dir', default='./ckpts_improved/')
args = parser.parse_args()

def train():
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device('cpu')
    
    model = ImprovedRetinexNet().to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    train_low = glob(args.data_dir + '/data/our485/low/*.png') + glob(args.data_dir + '/data/syn/low/*.png')
    train_high = glob(args.data_dir + '/data/our485/high/*.png') + glob(args.data_dir + '/data/syn/high/*.png')
    train_low.sort()
    train_high.sort()
    
    print(f'训练数据数量: {len(train_low)}')
    num_batch = len(train_low) // args.batch_size
    
    for epoch in range(args.epochs):
        for batch_id in range(num_batch):
            batch_low = np.zeros((args.batch_size, 3, args.patch_size, args.patch_size), dtype=np.float32)
            batch_high = np.zeros((args.batch_size, 3, args.patch_size, args.patch_size), dtype=np.float32)
            
            for i in range(args.batch_size):
                idx = batch_id * args.batch_size + i
                low_img = Image.open(train_low[idx % len(train_low)]).convert('RGB')
                high_img = Image.open(train_high[idx % len(train_high)]).convert('RGB')
                
                low_img = np.array(low_img, dtype=np.float32) / 255.0
                high_img = np.array(high_img, dtype=np.float32) / 255.0
                
                h, w, _ = low_img.shape
                x = random.randint(0, h - args.patch_size)
                y = random.randint(0, w - args.patch_size)
                low_img = low_img[x:x+args.patch_size, y:y+args.patch_size, :]
                high_img = high_img[x:x+args.patch_size, y:y+args.patch_size, :]
                
                low_img = np.transpose(low_img, (2, 0, 1))
                high_img = np.transpose(high_img, (2, 0, 1))
                
                batch_low[i] = low_img
                batch_high[i] = high_img
            
            input_low = torch.from_numpy(batch_low).float().to(device)
            input_high = torch.from_numpy(batch_high).float().to(device)
            
            loss_decom, loss_relight = model(input_low, input_high)
            total_loss = loss_decom + loss_relight
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001)
            optimizer.step()
            
            print(f'Epoch [{epoch+1}/{args.epochs}], Batch [{batch_id+1}/{num_batch}], '
                  f'Decom Loss: {loss_decom.item():.4f}, Relight Loss: {loss_relight.item():.4f}')
        
        # 每个epoch都保存
        torch.save(model.state_dict(), f'{args.ckpt_dir}/model_epoch{epoch+1}.pth')
        print(f'✅ 已保存 epoch {epoch+1}')
    
    torch.save(model.state_dict(), f'{args.ckpt_dir}/model_final.pth')
    print('训练完成！')

if __name__ == '__main__':
    train()