#这是第1部分，负责两个模型json融合成merged1.json

import json

# 读取第一个JSON文件
with open('test.bbox.json', 'r') as file1:
    data1 = json.load(file1)

# 读取第二个JSON文件
with open('test2.bbox.json', 'r') as file2:
    data2 = json.load(file2)


# 提取两个JSON数据中的 "detections" 列表
detections1 = data1
detections2 = data2
# 合并两个 "detections" 列表
merged_detections = detections1 + detections2

new_map = {} #id: {}
for elem in detections1:
    new_map[elem["image_id"]] = {
        "bboxes": ...
        
    }
for elem in detections2:
    new_map[elem["image_id"]]["bboxes"] = new_map[elem["image_id"]]["bboxes"] + elem["bboxes"]

merged_data =merged_detections\
# 保存合并后的数据为新的.json文件
with open('merged1.json', 'w') as output_file:
    json.dump(merged_data, output_file)  












# # 执行NMS操作
# def nms1(boxes, scores, iou_threshold):
#     # 创建一个列表来存储保留的框的索引
#     keep_indices = []
    
#     # 将框按分数降序排序
#     sorted_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    
#     while len(sorted_indices) > 0:
#         idx = sorted_indices[0]  # 获取分数最高的框的索引
#         keep_indices.append(idx)  # 将该框的索引加入保留列表
#         current_box = boxes[idx]
        
#         # 计算当前框与其他框的IoU
#         iou_values = [calculate_iou(current_box, boxes[i]) for i in sorted_indices[1:]]
        
#         # 找到IoU小于阈值的框的索引
#         filtered_indices = [i + 1 for i in range(len(iou_values)) if iou_values[i] <= iou_threshold]
        
#         # 更新排序后的索引列表，仅保留通过NMS的框
#         sorted_indices = [sorted_indices[i] for i in filtered_indices]
    
#     return keep_indices



# # 分别提取检测框、置信度得分和类别标签
# detections = merged_data['detections']
# bboxes = [detection['bbox'] for detection in detections]
# scores = [detection['score'] for detection in detections]

# # 执行NMS操作
# keep_indices = nms1(bboxes, scores, iou_threshold)

# # 获取NMS后的检测结果
# nms_detections = [detections[i] for i in keep_indices]

# # 构造包含NMS结果的字典
# nms_data = {'detections': nms_detections}

# # 将NMS后的结果保存为新的JSON文件
# with open('nms_results.json', 'w') as nms_file:
#     json.dump(nms_data, nms_file, indent=4)
