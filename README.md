交大视觉印象 – 图像检索与文字检测

本项目是《计算机视觉基础》课程大作业，实现了一个面向北京交通大学校园地标的图像检索与文字检测联合系统。  
系统接收一张交大校园地标查询图片，从包含 **7,725 张**图像的底库中检索出视觉最相似的图片，并自动检测图像中的文字区域。

---

项目结构
├── base/ # 底库图像（需自行放置）
├── query/ # 查询图像（需自行放置）
├── step1_extract_features.py # 步骤1：提取图像特征
├── step2_search_and_evaluate.py # 步骤2：检索并计算 P@K
├── step3_visualize_detection.py # 步骤3：文字检测与可视化
├── results/ # 运行结果
│ ├── P@K_*.png # 12 张地标 P@K 折线图
│ └── visualization/ # 24 组检索+文字检测可视化图
└── README.md

---

环境要求

- Python 3.8+
- PyTorch ≥ 1.10
- torchvision
- pillow
- numpy
- matplotlib
- tqdm
- scikit-learn（或仅使用 numpy 手动计算余弦相似度）
- EasyOCR

安装依赖

```bash
pip install torch torchvision pillow numpy matplotlib tqdm scikit-learn
pip install easyocr

## EasyOCR 首次运行时会自动下载检测模型（约 100 MB），请保持网络畅通。

运行说明
确保 base/ 和 query/ 文件夹与本项目脚本放在同一目录下。

步骤1：提取图像特征
python step1_extract_features.py

运行后会生成两个文件：
base_features.pkl — 底库所有图像的特征向量
query_features.pkl — 查询图像的特征向量

步骤2：检索与 P@K 评估
python step2_search_and_evaluate.py

运行后会：
对每一张 query 图像在 base 中进行检索
按 12 个地标类别分别计算 P@20、P@40、P@60
生成 12 张 P@K_类别.png 折线图，保存在当前目录

步骤3：文字检测与可视化
python step3_visualize_detection.py

运行后会：
对每个地标随机选取 2 张 query 图像
检索最相似的 4 张 base 图像
使用 EasyOCR 检测所有图片中的文字区域并绘制红框
拼接生成 24 组对比图，保存在 visualization_results/ 文件夹

结果展示
图像检索性能：12 张 P@K 折线图（见 results/）
文字检测可视化：24 组“检索 + 检测”对比图（见 results/visualization/）

说明
本实验全程未使用数据集的类别标签进行训练，检索仅依赖 ImageNet 预训练 ResNet-50 特征，文字检测使用 EasyOCR 通用中文检测模型。
提供的 base/ 和 query/ 数据集仅供本课程作业使用，不包含在本仓库中。
