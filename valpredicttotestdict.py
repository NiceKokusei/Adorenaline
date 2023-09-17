# 打开保存原始数据的文件
with open('/home/wjb/lj/predictions-frcnn.txt', 'r') as file:
    original_data = file.readlines()

# 初始化目标字典
target_dict = {}

# 遍历原始数据行
for line in original_data:
    parts = line.split()
    image_id = int(parts[0])
    label = int(parts[1])
    score = float(parts[2])
    x, y, w, h = map(float, parts[3:])
    
    key = (image_id, label)
    if key not in target_dict:
        target_dict[key] = []
    target_dict[key].append({
        'image_id': image_id,
        'bbox': [x, y, w, h],
        'score': score,
        'category_id': label
    })


# # 将结果保存为filter-predictions-dict.txt文件
# with open('filter-predictions-dict1.txt', 'w') as outfile:
#     for key, value in target_dict.items():
#         outfile.write(f"{key}: {value},\n")
        
# 保存中间文件为.txt
#将字典转换为字符串格式
dict_str = str(target_dict)
# 定义要保存的文件名
file_name = "filter-predictions-dict2.txt"
# 打开文件并将字典字符串写入文件
with open(file_name, "w") as file:
    file.write(dict_str)