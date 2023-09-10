#这是2，3步骤，负责生成filtered_dict；以及后续nms（但其nms部分有小问题，使用filtnms.py）


import json

import numpy as np
np.set_printoptions(precision=18) #设置小数位置为18位
# 读取JSON文件
with open('/home/wjb/lj/atelier/mmdetection2/work_dirs/coco_detection/merged1.json', 'r') as file:
    data = json.load(file)
    
# 提取合并JSON数据中的 "detections" 列表
detections = data  
boxes = []
scores = []
idxs = []
labels = []
converted_boxes = []
# 创建一个新的字典来存储同一张图，同一类别的字典
filtered_dict = {}
# 创建一个新的列表 select_values 来存储nms后的字典
select_values = []

def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]#左上
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]#右下

    areas = (x2 - x1) * (y2 - y1)

    # 从大到小对应的的索引
    order = scores.argsort()[::-1]#进行排序，并返回排序后的索引

    # 记录输出的bbox
    keep = []#得分降序排列的索引中具有最大得分的索引，并将其添加到keep列表中
    while order.size > 0:
        i = order[0]
        # 记录本轮最大的score对应的index
        keep.append(i)

        if order.size == 1:
            break

        # 计算当前bbox与剩余的bbox之间的IoU
        # 计算IoU需要两个bbox中最大左上角的坐标点和最小右下角的坐标点
        # 即重合区域的左上角坐标点和右下角坐标点
        xx1 = np.maximum(x1[i], x1[order[1:]])#表示除了当前边界框 i 之外，剩余边界框的左上角坐标
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 如果两个bbox之间没有重合, 那么有可能出现负值
        w = np.maximum(0.0, (xx2 - xx1))
        h = np.maximum(0.0, (yy2 - yy1))
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 删除IoU大于指定阈值的bbox(重合度高), 保留小于指定阈值的bbox
        ids = np.where(iou <= threshold)[0]
        # 因为ids表示剩余的bbox的索引长度
        # +1恢复到order的长度
        order = order[ids + 1]

    return keep


# 遍历 data
for d in data:
    image_id = d.get("image_id")
    category_id = d.get("category_id")
    # 使用元组 (image_id, category_id) 作为键
    key = (image_id, category_id)

    # 如果键不存在于 filtered_dict 中，则创建一个新的键并初始化值为一个空列表
    if key not in filtered_dict:
        filtered_dict[key] = []

    # 将当前字典添加到对应键的值列表中
    filtered_dict[key].append(d)


# # 保存中间文件为.txt
# #将字典转换为字符串格式
# dict_str = str(filtered_dict)
# # 定义要保存的文件名
# file_name = "filtered_dict.txt"
# # 打开文件并将字典字符串写入文件
# with open(file_name, "w") as file:
#     file.write(dict_str)



# 遍历 filtered_dict 中的所有键并取出对应的值
for key in filtered_dict:#同一张图，同一类别
    values = filtered_dict[key]#value形状[{},{},{}],和json相同
    for elem in values:#同一张图，同一类别的所有框的字典
        scores.append(elem["score"])
        scoresnp = np.array(scores)
        boxes.append(elem["bbox"])
        converted_boxes = [[x, y, x + w, y + h] for x, y, w, h in boxes]
        converted_boxesnp = np.array(converted_boxes)
    keep = nms(converted_boxesnp, scoresnp, threshold=0.5)#keep是索引，anchors是[[],[]],scores是[]。
        
    proposals_scoresnp = scoresnp[keep]
        
    # # 遍历 values 中的每个字典
    # for value in values:
    # 获取当前字典的 "score" 属性值
    #proposals_score = values.get("score")

    # 遍历列表 values 中的每个字典
    for value in values:
        # 使用 get() 方法获取 "score" 属性的值，并添加到 proposals_score 列表中
        score = value.get("score")
        #proposals_score.append(score)

        # 检查当前 "score" 是否在 proposal_scores 中
        if score in proposals_scoresnp:
        # 如果相等，将字典添加到 select_values 中
            select_values.append(value)
    
# 定义要保存的文件名
file_name = "selected_values.json"

# 打开文件并将 select_values 内容以 JSON 格式写入文件
with open(file_name, "w") as file:
    json.dump(select_values, file)





#keep = nms(anchors, scores, threshold=0.5)#keep是索引，anchors是[[],[]],scores是[]。
# proposals = anchors[keep]
# proposals_score = scores[keep]
    