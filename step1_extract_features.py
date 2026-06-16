import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'正在使用的设备: {device}')

model = resnet50(weights='DEFAULT')
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features_for_folder(root_folder, save_name):
    """
    遍历 root_folder 及其子文件夹，提取所有图片的特征。
    返回的字典 key 是图片相对于 root_folder 的路径，
    比如 'BJTU/fhy-2i3j4k5l6m7n.jpeg'。
    """
    features = {}
    img_paths = []
    # 用 os.walk 递归遍历所有子目录
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                img_paths.append(os.path.join(dirpath, fname))
    
    print(f'在 {root_folder} 下共找到 {len(img_paths)} 张图片')
    for full_path in tqdm(img_paths, desc=f'提取 {root_folder}'):
        try:
            img = Image.open(full_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model(img_tensor)
            # 用相对于 base 或 query 文件夹的路径作为 key
            rel_path = os.path.relpath(full_path, root_folder)
            features[rel_path] = feature.cpu().numpy().flatten()
        except Exception as e:
            print(f'处理 {full_path} 时出错: {e}')
    
    with open(f'{save_name}.pkl', 'wb') as f:
        pickle.dump(features, f)
    print(f'已保存到 {save_name}.pkl，共 {len(features)} 张')
    return features

# ---------- 执行 ----------
BASE_PATH = './base'
QUERY_PATH = './query'

if not os.path.exists(BASE_PATH) or not os.path.exists(QUERY_PATH):
    print("错误：请确保 base 和 query 文件夹和本脚本在同一目录下！")
else:
    base_feats = extract_features_for_folder(BASE_PATH, 'base_features')
    query_feats = extract_features_for_folder(QUERY_PATH, 'query_features')
    print("\n搞定！提取完成！")