import pickle

# 读取第一个.pkl文件
with open('/home/wjb/lj/atelier/mmdetection2/configs/atss/atss-val.pkl', 'rb') as f1:
    data1 = pickle.load(f1)

# 读取第二个.pkl文件
with open('/home/wjb/lj/atelier/mmdetection2/configs/faster_rcnn/frcnn-val.pkl', 'rb') as f2:
    data2 = pickle.load(f2)

# 合并两个数据
merged_data = data1 + data2

# 将合并后的数据保存为新的.pkl文件
with open('valmerged.pkl', 'wb') as merged_file:
    pickle.dump(merged_data, merged_file)
