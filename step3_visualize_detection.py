import pickle
import numpy as np
import os
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import easyocr

random.seed(42)

def cosine_similarity_manual(query_vec, base_matrix):
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    base_norm = base_matrix / (np.linalg.norm(base_matrix, axis=1, keepdims=True) + 1e-8)
    return np.dot(query_norm, base_norm.T)

print("加载特征文件...")
with open('base_features.pkl', 'rb') as f:
    base_feats = pickle.load(f)
with open('query_features.pkl', 'rb') as f:
    query_feats = pickle.load(f)

base_names = list(base_feats.keys())
base_array = np.array([base_feats[name] for name in base_names])
query_names = list(query_feats.keys())

def search_top_k(query_feat, k=5):
    similarities = cosine_similarity_manual(query_feat, base_array)
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return [base_names[i] for i in top_k_indices]

def get_label(rel_path, is_query=False):
    filename = os.path.basename(rel_path)
    if is_query:
        if '-' in filename:
            return filename.split('-')[0]
        else:
            return None
    else:
        if 'BJTU' in rel_path and '-' in filename:
            return filename.split('-')[0]
        else:
            return None

print("加载 EasyOCR 文字检测模型...")
reader = easyocr.Reader(['ch_sim'], gpu=False)
print("模型加载完毕。")

def draw_ocr_boxes(image, boxes):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for box in boxes:
        try:
            pts = [(int(p[0]), int(p[1])) for p in box]
            if len(pts) >= 2:
                draw.polygon(pts, outline='red', width=3)
        except:
            pass
    return img

def detect_text(img_path):
    try:
        result = reader.detect(img_path)
        boxes = []
        if result is not None and len(result) > 0:
            raw_boxes = result[0]  # horizontal_list
            if raw_boxes:
                for box in raw_boxes:
                    if box is not None and len(box) >= 2:
                        boxes.append(box)
    except Exception as e:
        print(f"检测失败 {img_path}: {e}")
        boxes = []
    
    image = Image.open(img_path).convert('RGB')
    image_with_boxes = draw_ocr_boxes(image, boxes)
    return image_with_boxes, len(boxes)

os.makedirs('visualization_results', exist_ok=True)

from collections import defaultdict
class_query_map = defaultdict(list)
for q_path in query_names:
    label = get_label(q_path, is_query=True)
    if label is not None:
        class_query_map[label].append(q_path)

print(f"将处理 {len(class_query_map)} 个地标类别：{list(class_query_map.keys())}")

for landmark, q_list in class_query_map.items():
    selected_queries = random.sample(q_list, min(2, len(q_list)))
    
    for idx, q_path in enumerate(selected_queries):
        print(f"正在处理 {landmark} 的第 {idx+1} 组...")
        q_full_path = os.path.join('query', q_path)
        if not os.path.exists(q_full_path):
            q_full_path = q_path
        
        q_feat = query_feats[q_path]
        top_k_names = search_top_k(q_feat, k=4)
        
        q_img, q_box_count = detect_text(q_full_path)
        print(f"  Query 检测到 {q_box_count} 个文字区域")
        
        retrieved_imgs = []
        for b_path in top_k_names:
            b_full_path = os.path.join('base', b_path)
            if not os.path.exists(b_full_path):
                b_full_path = b_path
            b_img, b_box_count = detect_text(b_full_path)
            retrieved_imgs.append(b_img)
        
        thumb_width = 300
        q_thumb = q_img.resize((thumb_width, int(q_img.height * thumb_width / q_img.width)))
        r_thumbs = [img.resize((thumb_width, int(img.height * thumb_width / img.width))) for img in retrieved_imgs]
        
        padding = 10
        img_height = q_thumb.height + padding + max(t.height for t in r_thumbs)
        img_width = thumb_width * 4 + padding * 3
        canvas = Image.new('RGB', (img_width + padding*2, img_height + padding*2), (255, 255, 255))
        canvas.paste(q_thumb, (padding, padding))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("simhei.ttf", 16)
        except:
            font = ImageFont.load_default()
        draw.text((padding, padding + q_thumb.height + 5), f"Query ({q_box_count} text boxes)", fill='black', font=font)
        
        y_offset = padding + q_thumb.height + padding + 20
        for i, r_thumb in enumerate(r_thumbs):
            x_offset = padding + i * (thumb_width + padding)
            canvas.paste(r_thumb, (x_offset, y_offset))
            draw.text((x_offset, y_offset + r_thumb.height + 5), f"Top{i+1}", fill='black', font=font)
        
        save_path = f'visualization_results/{landmark}_group{idx+1}.png'
        canvas.save(save_path)
        print(f"  已保存 {save_path}")

print("\n全部完成！可视化结果保存在 visualization_results 文件夹。")