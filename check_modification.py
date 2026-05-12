with open('model.py', 'r', encoding='utf-8') as f:
    content = f.read()
    if "torch.load(load_dir + load_ckpt, map_location='cpu')" in content:
        print("✅ model.py修改正确！")
    elif "torch.load(load_dir + load_ckpt, map_location=" in content:
        print("✅ model.py已修改（可能有不同格式）")
    else:
        print("❌ model.py未修改或修改不正确")
        print("请找到 torch.load(load_dir + load_ckpt) 并添加 map_location='cpu'")