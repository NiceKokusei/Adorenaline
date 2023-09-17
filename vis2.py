import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
from pycocotools.coco import COCO

# 指定模型输出的 .json 文件路径
result_file = '/home/wjb/lj/atelier/mmdetection2/work_dirs/coco_detection/selected_values.json'

# 指定保存可视化结果的文件夹
save_dir = '/home/wjb/lj/atelier/mmdetection2/work_dirs1/test-merged'

# 加载 COCO 数据集的标注文件（通常在数据集的根目录中）
coco_gt = COCO('/data2/wjb/dataset/coco/annotations/image_info_test-dev2017.json')

# 创建保存文件夹，如果不存在
os.makedirs(save_dir, exist_ok=True)

# 读取模型输出的 .json 文件
with open(result_file, 'r') as f:
    results = json.load(f)

# 初始化存储累积图像的字典
accumulated_image = None



# 存储不同类别的颜色映射
color_map = {}

# 读取模型输出的 .json 文件
with open(result_file, 'r') as f:
    results = json.load(f)


# 循环遍历每个结果
for result in results:
    # 获取图像 ID
    img_id = result['image_id']

    # 获取图像文件名（如果需要）
    img_info = coco_gt.loadImgs(img_id)[0]
    img_filename = img_info['file_name']

    # 获取检测框信息
    bbox = result['bbox']
    score = result['score']
    category_id = result['category_id']

    # 只绘制分数大于0.3的对象
    if score > 0.3:
        # 获取图像路径（根据你的数据集存储方式可能需要自定义）
        img_path = '/data2/wjb/dataset/coco/test2017/' + img_filename

        # 读取图像
        img = cv2.imread(img_path)

        # 获取颜色并将其转换为整数格式
        if category_id not in color_map:
            color_map[category_id] = np.random.randint(0, 255, size=3)

        color = color_map[category_id]
        color = (int(color[0]), int(color[1]), int(color[2]))

        # 绘制检测框
        x, y, w, h = bbox
        label = f"Category {category_id}"


        
        if accumulated_image is None  or img_id != prev_img_id:
            accumulated_image = img
        else:
            accumulated_image = accumulated_image
        prev_img_id = img_id
         


        # 绘制检测框和文本标注
        cv2.rectangle(accumulated_image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.putText(accumulated_image, f"Score: {score:.2f}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(accumulated_image, label, (int(x), int(y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        
        
        # 保存可视化结果到指定路径（每次保存一张图片）
        save_path = os.path.join(save_dir, f"{img_id:012d}_result.png")
        cv2.imwrite(save_path, accumulated_image)



# 所有结果都已保存完毕
print("可视化结果已保存到:", save_dir)




