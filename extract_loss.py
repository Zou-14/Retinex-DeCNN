import re
import csv

# 自动检测编码
import chardet

# 读取文件并检测编码
with open('train_log.txt', 'rb') as f:
    raw_data = f.read()
    encoding = chardet.detect(raw_data)['encoding']

print(f'检测到文件编码: {encoding}')

# 用检测到的编码读取文件
with open('train_log.txt', 'r', encoding=encoding) as f:
    lines = f.readlines()

# 准备存储数据
batch_data = []
decom_losses = []
relight_losses = []

# 正则表达式提取每行的batch编号和loss值
for line in lines:
    # 匹配 Batch [数字/数字]
    batch_match = re.search(r'Batch \[(\d+)/', line)
    # 匹配 Decom Loss: 数字
    decom_match = re.search(r'Decom Loss: ([-+]?\d*\.?\d+)', line)
    # 匹配 Relight Loss: 数字
    relight_match = re.search(r'Relight Loss: ([-+]?\d*\.?\d+)', line)
    
    if batch_match and decom_match and relight_match:
        batch = int(batch_match.group(1))
        decom = float(decom_match.group(1))
        relight = float(relight_match.group(1))
        
        batch_data.append(batch)
        decom_losses.append(decom)
        relight_losses.append(relight)

# 保存为CSV文件
with open('loss_curve.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['Batch', 'Decom Loss', 'Relight Loss'])
    for i in range(len(batch_data)):
        writer.writerow([batch_data[i], decom_losses[i], relight_losses[i]])

print(f'✅ 成功提取 {len(batch_data)} 个batch的数据')
print(f'✅ 已保存到 loss_curve.csv')