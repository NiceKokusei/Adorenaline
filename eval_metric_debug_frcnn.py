# Copyright (c) OpenMMLab. All rights reserved.
# ##eval_metric.py  评估一个计算机视觉模型的性能的工具
import argparse

import mmengine
from mmengine import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope

from mmdet.registry import DATASETS

import json

predictions = []

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                     'results saved in pkl format')
    parser.add_argument('config', help='Config of the model')
                        #default='/home/wjb/lj/atelier/mmdetection2/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py')
    parser.add_argument('pkl_results', help='Results in pickle format')
                        #default='/home/wjb/lj/atelier/mmdetection2/work_dirs/coco_detection/valmerged.pkl')
    parser.add_argument(#eg   python your_script.py --cfg-options model.name="VGG" model.num_layers=16 model.batch_size=32
        '--cfg-options',#允许用户在命令行中覆盖配置文件中的一些设置;用户在命令行中使用 --cfg-options 来指定覆盖配置文件设置的选项
        nargs='+',#示允许接受多个值，这些值会存储在一个列表中
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
#加载配置文件、模型预测结果和数据集，然后使用指定的评估器对模型性能进行评估，并将结果打印出来
    args = parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet'))#根据配置中的信息初始化默认的作用域

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)#如果存在 args.cfg_options，则将其合并到配置对象 cfg 中

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    
    #global predictions  # 声明 predictions 为全局变量
    
    predictions = mmengine.load(args.pkl_results)

    # # 保存预测结果到文件
    # with open('predictions-frcnn.txt', 'w') as prediction_file:
    #     for prediction in predictions:
    #         image_id = prediction['img_id']
    #         labels = prediction['pred_instances']['labels']
    #         scores = prediction['pred_instances']['scores']
    #         bboxes = prediction['pred_instances']['bboxes']
            
    #         for label, score, bbox in zip(labels, scores, bboxes):
    #             bbox_str = ' '.join([str(coord) for coord in bbox.tolist()])
    #             prediction_str = f"{image_id} {label} {score:.4f} {bbox_str}\n"
    #             prediction_file.write(prediction_str)



    # 读取 selected_val_values0.5.json 文件
    with open('selected_val_values0.5.json', 'r') as selected_file:
        selected_data = json.load(selected_file)

    # 初始化一个用于保存匹配的原始 predictions 数据的列表
    predictions1 = []

    # 遍历 selected_data 并提取相关信息
    for item in selected_data:
        image_id = item["image_id"]
        label = item["category_id"]
        score = item["score"]

        # # 在 predictions 中查找匹配的数据
        # for prediction in predictions:
        #     if (
        #         prediction["img_id"] == image_id
        #         and prediction['pred_instances']['scores'] == score
        #         and prediction['pred_instances']['labels'] == label

        #     ):
        #         # 匹配成功，将原始 predictions 数据添加到列表中
        #         predictions1.append(prediction)
        
        for prediction in predictions:
            if (prediction["img_id"] == image_id):
            # 遍历批次中的每个结果
                for result in prediction['pred_instances']:
                    if (
                        #result["img_id"] == image_id
                        score in result['scores']
                        and label in result['labels']
                    ):
                        # 匹配成功，将原始结果添加到临时列表中
                        matched_results.append(result)

    # 将匹配的结果作为一个批次添加到 predictions1 列表中
    if matched_results:
        predictions1.append(matched_results)






    evaluator = Evaluator(cfg.val_evaluator)
    evaluator.dataset_meta = dataset.metainfo
    eval_results = evaluator.offline_evaluate(predictions)#core
    print(eval_results)#最后，将评估结果打印到控制台


if __name__ == '__main__':
    main()
