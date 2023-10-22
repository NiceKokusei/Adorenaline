import json
import cv2
import os
import numpy as np

file_name = '/data3/wjb/dataset/isaid_split/train/instancesonly_filtered_train.json'
imgs_path = '/data3/wjb/dataset/isaid_split/train/images/'
save_path = '/home/wjb/lj/atelier/OBBDetection/VIS-ZZ/'


# color_mapping = {  #bgr
#     "Large_Vehicle": (255, 128, 0),
#     "Swimming_Pool": (147, 115, 116),
#     "Helicopter": (0, 0, 255),
#     "Bridge": (0, 255, 0),
#     "plane": (165, 42, 42),
#     "ship": (255, 0, 255),
#     "Soccer_ball_field": (255, 250, 205),
#     "basketball_court": (255, 193, 193),
#     "Ground_Track_Field": (255, 0, 0),
#     "Small_Vehicle": (138, 43, 226),
#     "baseball_diamond": (189, 183, 107),
#     "tennis_court": (0, 255, 255),
#     "Roundabout": (8, 139, 139),
#     "storage_tank": (0, 51, 153),
#     "Harbor": (255, 255, 0),
# }

color_mapping = {
    'Large_Vehicle': (0, 128, 255),
    'Swimming_Pool': (116, 115, 147),
    'Helicopter': (255, 0, 0),
    'Bridge': (0, 255, 0),
    'plane': (42, 42, 165),
    'ship': (255, 0, 255),
    'Soccer_ball_field': (205, 250, 255),
    'basketball_court': (193, 193, 255),
    'Ground_Track_Field': (0, 0, 255),
    'Small_Vehicle': (226, 43, 138),
    'baseball_diamond': (107, 183, 189),
    'tennis_court': (255, 255, 0),
    'Roundabout': (139, 139, 8),
    'storage_tank': (153, 51, 0),
    'Harbor': (0, 255, 255)
}

with open(file_name) as jsf:
    dataset_annotation = json.load(jsf)
imgs = dataset_annotation['images']
anns = dataset_annotation['annotations']

num_imgs = len(imgs)
num_anns = len(anns)

categories = {cat['id']: cat['name'] for cat in dataset_annotation['categories']}
imgs_mask = {}
for i in range(num_anns):
    img_id = anns[i]['image_id']
    img_name = imgs[img_id]['file_name']
    category_name = categories[anns[i]['category_id']]
    segmentation = anns[i]['segmentation']
    if img_name not in imgs_mask:
        imgs_mask[img_name] = [dict(category=category_name, segmentation=segmentation),]
    else:
        imgs_mask[img_name].append(dict(category=category_name, segmentation=segmentation))

for i in range(num_imgs):
    img_name = imgs[i]['file_name']
    if img_name in imgs_mask:
        masks = imgs_mask[img_name]

    img = cv2.imread(os.path.join(imgs_path, img_name))
    img1 = img.copy()
    img2 = img.copy()
    #result = img.copy()

    for mask in masks:
        category = mask['category']
        segmentation = mask['segmentation'][0]
        x = np.array(segmentation[0::2])  
        y = np.array(segmentation[1::2])  
        coordinates = np.stack([x, y], axis=1)  
        color = np.random.randint(0, 255, (3,))
        color1= tuple(color)
        color1 = int(color1[0]), int(color1[1]), int(color1[2])
        
        # roi_original = img[coordinates[:, 0], coordinates[:, 1]]  #x,y颠倒
        # img[coordinates[:, 0], coordinates[:, 1]] = roi_original * 0.3 + color * 0.7  #而且只有外轮廓
        #tmp1 = cv2.fillPoly(img, pts=[coordinates], color=(np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255)))

        

        # 根据 category_name 找到对应的颜色，如果未知则使用默认颜色
        linecolor = color_mapping.get(category, (0, 0, 0))



        rect = cv2.minAreaRect(coordinates)
        box = cv2.boxPoints(rect)
        box = np.int0(box)  
        #result = cv2.drawContours(img1, [box], 0, (0, 255, 0, 255), thickness=2) 
        result = cv2.drawContours(img1, [box], 0, linecolor, thickness=1, lineType=cv2.LINE_AA) 

        tmp1 = cv2.fillPoly(img1, pts=[coordinates], color=color1)  #掩码涂色
        result = cv2.addWeighted(img2, 0.3, tmp1, 0.7, 0)  #原图叠加掩码  img2恒为原图不变，img1是每个掩码不同
        #result = cv2.addWeighted(result, 0.7, tmp1, 0.3, 0)
        


        
    cv2.imwrite(os.path.join(save_path, img_name[:-4]+'_'+'segrec'+'.png'), result)
        
        
        
        
        #cv2.fillPoly(img, pts=[coordinates], color=(255, 0, 0))#BGR

    




print('sucessfully')