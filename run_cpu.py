import os
import sys

print("=" * 50)
print("强制使用CPU模式运行RetinexNet")
print("=" * 50)

# 1. 强制使用CPU（最重要的一步！）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 2. 验证环境
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("警告：检测到CUDA，但已强制使用CPU模式")

# 3. 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 4. 修改model.py的load方法（动态修改）
print("\n正在动态修复model.py的加载问题...")

# 5. 直接运行predict.py
print("\n开始运行RetinexNet预测...")
print("-" * 50)

try:
    # 直接执行predict.py
    with open('predict.py', 'r', encoding='utf-8') as f:
        predict_code = f.read()
    
    # 在代码执行前动态修改
    predict_code = predict_code.replace(
        "torch.load(load_dir + load_ckpt)", 
        "torch.load(load_dir + load_ckpt, map_location='cpu')"
    )
    
    # 执行修改后的代码
    exec(predict_code)
    
except FileNotFoundError:
    print("错误：找不到predict.py文件")
    print(f"当前目录: {current_dir}")
    print("文件列表:", os.listdir(current_dir))
except Exception as e:
    print(f"运行出错: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("运行结束")
print("=" * 50)