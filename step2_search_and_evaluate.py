import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# ---------- 手动实现余弦相似度（不依赖 sklearn） ----------
def cosine_similarity_manual(query_vec, base_matrix):
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    base_norm = base_matrix / (np.linalg.norm(base_matrix, axis=1, keepdims=True) + 1e-8)
    return np.dot(query_norm, base_norm.T)

# ---------- 加载特征 ----------
print("正在加载指纹...")
with open('base_features.pkl', 'rb') as f:
    base_feats = pickle.load(f)
with open('query_features.pkl', 'rb') as f:
    query_feats = pickle.load(f)

base_names = list(base_feats.keys())    # 例如 ['BJTU/fhy-xxx.jpg', 'util_pic/dog_123.jpg', ...]
base_array = np.array([base_feats[name] for name in base_names])

query_names = list(query_feats.keys())
query_array = np.array([query_feats[name] for name in query_names])

print(f"底库图片数量: {len(base_names)}, 查询图片数量: {len(query_names)}")

# ---------- 标签提取 ----------
def get_label(rel_path, is_query=False):
    """
    从相对路径中提取地标代号。
    - query 图片：全部都是交大相关，直接提取横线前的标签。
    - base 图片：只有路径中包含 'BJTU' 的才提取标签，util_pic 返回 None。
    """
    filename = os.path.basename(rel_path)
    if is_query:
        # query 图片无条件提取标签（前提是文件名包含 '-'）
        if '-' in filename:
            return filename.split('-')[0]
        else:
            return None
    else:
        if 'BJTU' in rel_path and '-' in filename:
            return filename.split('-')[0]
        else:
            return None

# ---------- 搜索函数 ----------
def search_top_k(query_feat, k=60):
    similarities = cosine_similarity_manual(query_feat, base_array)
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return [base_names[i] for i in top_k_indices]

# ---------- 按类别整理 query 图片 ----------
class_query_map = defaultdict(list)
for q_path in query_names:
    label = get_label(q_path, is_query=True)
    if label is not None:
        class_query_map[label].append(q_path)

print(f"共发现 {len(class_query_map)} 个地标类别: {list(class_query_map.keys())}")

# ---------- 评估并画图 ----------
for landmark, q_list in class_query_map.items():
    print(f"正在评估类别: {landmark}，共 {len(q_list)} 张查询图")
    k_values = [20, 40, 60]
    avg_precisions = {k: 0 for k in k_values}
    
    for q_path in q_list:
        q_feat = query_feats[q_path]
        top_k_names = search_top_k(q_feat, k=60)
        
        for k in k_values:
            top_k_current = top_k_names[:k]
            relevant = 0
            query_label = get_label(q_path, is_query=True)
            for name in top_k_current:
                base_label = get_label(name, is_query=False)  # 用 base 的提取方式
                if base_label == query_label:
                    relevant += 1
            avg_precisions[k] += relevant / k
    
    for k in k_values:
        avg_precisions[k] /= len(q_list)
    
    # 画图
    plt.figure()
    precisions = [avg_precisions[20], avg_precisions[40], avg_precisions[60]]
    plt.plot([20, 40, 60], precisions, marker='o')
    plt.xlabel('K')
    plt.ylabel('Precision')
    plt.title(f'P@K for {landmark}')
    plt.xticks([20, 40, 60])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.0)
    plt.savefig(f'P@K_{landmark}.png')
    print(f"  保存图片: P@K_{landmark}.png")
    plt.close()

print("\n全部完成！所有 P@K 图已生成。")