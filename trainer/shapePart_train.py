'''
Author       : ZHP
Date         : 2022-01-05 16:17:57
LastEditors  : ZHP
LastEditTime : 2022-02-15 14:02:45
FilePath     : /trainer/shapePart_train.py
Description  : 
Copyright 2022 ZHP, All Rights Reserved. 
2022-01-05 16:17:57
'''
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
from functools import reduce
import time


from .trainer_basic import Trainer
from data import ShapeNetPartNormalDataset, ShapeNetPartNormalDataset_Sample
from utils import utils

NUM_PARTS = 50
EARLY_STOPPING_PATIENCE = 30

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


@torch.no_grad()
def calculate_batch_iou(predict, labels):
    '''
    @description: Calculate the average Intersection over Union(mIoU) 
                  between the predicted label and the true label
    @param {
        predict : (B, parts, N)
        labels : (B, N) 
    }
    @return {
        accuracy(ndarray) : shape in (2,) the accuracy of prediction results:index = 0 is the correct points, other is sum of all points
        cat_ious(dictionary) : the keys represent the object category(13),the value is a lsit that means it's parts ious
        output_3(ndarray) : shape in (50,),record every part TP + FN points
        output_4(ndarray) : shape in (50,),record every part TP points
    }
    '''    
    batch_size, npoints = labels.shape
    predict_label = np.zeros((batch_size, npoints)).astype(np.int32)  # 用于保存预测标签
    gt_list = [0 for _ in range(NUM_PARTS)]                 # 统计每个part gt的点
    pre_correct_list = [0 for _ in range(NUM_PARTS)]        # 统计每个part预测正确的点
    cat_ious = {cat_ : [] for cat_ in seg_classes.keys()}    # 统计每个category的ious
    for batch in range(batch_size):
        # 在每个样本上计算
        category = seg_label_to_cat[labels[batch, 0]]  # get Object category
        parts_label = seg_classes[category]  # [16, 17, 18]
        # 在当前类的parts标签通道上找最大值，而不是在所有50个part上找最大
        relative_label = np.argmax(predict[batch, parts_label, :], axis=0)   # (N) 每个值代表在当前类parts列表里的索引
        relative_label = relative_label + parts_label[0] # [B, N]  
        # 因为每类的parts标签都是连续的
        predict_label[batch, :] = relative_label

        seg_gt = labels[batch, :]
        part_ious = [0.0 for _ in range(len(parts_label))]  # 记录当前N个点(当前category)的每个part iou
        for part_l in parts_label:
            relative_idx = part_l - parts_label[0]
            if (np.sum(seg_gt == part_l) == 0) and (
                np.sum(relative_label == part_l) == 0):   # 这个样本中没有part_l
                part_ious[relative_idx] = 1.0
            else:
                part_ious[relative_idx] = np.sum((seg_gt == part_l) & (
                    relative_label == part_l)) / float(np.sum(
                        (seg_gt == part_l) | (relative_label == part_l)
                    ))   # 交/并
            # cat_ious[category].append(np.mean(part_ious))
        cat_ious[category].append(np.mean(part_ious))  # 每个category维护一个自己的平均part iou list，list每个元素是自己的一个样本上的cat iou
    accuracy_list = [np.sum(predict_label == labels), (batch_size * npoints) ]

    for k in range(NUM_PARTS):
        gt_list[k] += np.sum(labels == k)   # 记录该part GT的点，即TP + FN
        pre_correct_list[k] += np.sum((predict_label == k) & (labels == k))  # 该part预测正确的点的个数，即 TP

    return np.array(accuracy_list), cat_ious, np.array(gt_list, dtype=np.float), np.array(pre_correct_list, dtype=np.float64)


class PartSegTrainer(Trainer):

    def run(self):
        nEpochs = self.config["nEpochs"]
        start_epoch, best_metrics = self.prepare()
        indent = " " * 3
        for epoch in range(start_epoch, nEpochs + 1):
            train_dict, train_time = self.train(**{
                "epoch" : epoch,
                "nEpochs" : nEpochs
            })
            (valid_metrics, valid_category_iou_dict), val_time = self.valid()
            remark, stopping, best_metrics = self.save_model(valid_metrics, best_metrics, epoch)

            valid_metrics.update({
                "loss_train" : train_dict["loss_train"],
                "accuracy_train" : train_dict["train_acc"],
                "learningRate" : train_dict["now_lr"]
            })
            self.write_info(epoch=epoch, train_dict=train_dict, indent=indent,\
                valid_metrics=valid_metrics, remark=remark, valid_category_iou_dict=valid_category_iou_dict)
            self.add_tensorboard(epoch, **valid_metrics)
            print("now_lr : {:^6.5f} | train loss: {:^5.3f} | valid loss:{:^5.3f} | cat.mIoU:{:^5.3f} | ins.mIoU:{:^5.3f} | train time:{}s | valid time:{}s\n".format(
                train_dict["now_lr"], train_dict["loss_train"], valid_metrics.get("loss_valid")[0], valid_metrics.get("mIoU_category")[0],\
                valid_metrics.get("mIoU_instance")[0], train_time, val_time))
            self.best_metrics = best_metrics
            self.upload_wandb(valid_metrics)
            if stopping:
                print("train process ready to stop")
                break
        self.board.close()
        self.end_train(best_metrics)

    def write_info(self, **kw):
        epoch, train_dict, valid_category_iou_dict = kw["epoch"], kw["train_dict"], kw["valid_category_iou_dict"],
        indent, valid_metrics, remark = kw["indent"], kw["valid_metrics"], kw["remark"]
        self.log.write_txt("trainLog", f'{epoch : ^5}{indent}{train_dict["now_lr"] : ^13.6f}{indent}' + 
            f'{train_dict.get("loss_train"):^10.5f}{indent}{train_dict["train_acc"]:^9.4f}{indent}' + 
            f'{valid_metrics["loss_valid"][0]:^10.5f}{indent}{valid_metrics["mIoU_category"][0]:^14.3f}{indent}' 
            + f'{valid_metrics["mIoU_instance"][0]:^14.3f}{indent}{valid_metrics["accuracy_valid"][0]:^9.4f}{indent}' +  
            f'{valid_metrics["accuracy_PartAvg"] : ^18.4f}{indent}' + f'{remark:^18}\n')
        
        iou_info = [epoch, indent]
        tuple_list = sorted(valid_category_iou_dict.items(), key=lambda x : x[0])
        valid_category_iou_list = [iou[1] for iou in tuple_list]
        iou_info.extend([valid_category_iou_list[i // 2] if i % 2 == 0 else indent for i in range(32) ])
        iou_info.extend([indent, kw["remark"]])
        iou_info = "{:^5}{}{:^8.3f}{}{:^5.3f}{}{:^5.3f}{}{:^5.3f}{}{:^5.3f}{}{:^8.3f}{}"\
            "{:^6.3f}{}{:^5.3f}{}{:^5.3f}{}{:^6.3f}{}{:^9.3f}{}{:^5.3f}{}{:^6.3f}{}"\
                "{:^6.3f}{}{:^10.3f}{}{:^5.3f}{}{}{:^6}\n".format(*iou_info)
        self.log.write_txt("cat_iou", iou_info)

    def prepare(self, **kw):
        indent = " " * 3
        self.config_gpus()          # resume 时model移动到cuda操作需在加载optimizer断点之前
        if self.resume:
            # 断点续训练
            model_save_dir = self.config["model_save_dir"]
            last_model = [os.path.join(model_save_dir, path) for path in sorted(
                os.listdir(model_save_dir), key=lambda x : int(x.split("_")[-1].split(".")[0]))][-1]
            checkpoint = torch.load(last_model)
            self.model.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']
            best_metrics = checkpoint["best_metrics"]
            resume_prompt = '\n' + "=" * 60 + "resume time : %s" % self.start_time + "=" * 80 + '\n'
            self.log.write_txt("trainLog", resume_prompt)
            self.log.write_txt('cat_iou', resume_prompt)
        else:  
            self.log.write_txt("trainLog", '\n' + "=" * 50 + " " + self.config["task"] + " training process " + "=" * 80 + '\n\n')
            self.log.write_txt("trainLog", "epoch%slearning rate%strain loss%strain acc%svalid loss%svalid cat.mIoU"\
            "%svalid ins.mIoU%svalid acc%svalid part-avg-acc%s%sremark%s\n" % ((indent,) * 11 ))
            cat_keys = sorted(list(seg_classes.keys()))
            info = "epoch" + indent + reduce(lambda x,y : x + indent + y, list(map(lambda x : "{: ^5}".format(x), cat_keys))) + \
                indent * 2 + "remark" + "\n"   # Airplane Bag ..   remark
            self.log.write_txt("cat_iou", info)
            best_metrics = {
                "mIoU_category" : 0,
                "mIoU_instance" : 0,
                "best_mIoU_category_epoch" : -1,
                "best_mIoU_instance_epoch" : -1,
                "loss_valid" : 10000}
            start_epoch = 1
        return start_epoch, best_metrics

    def build_loader(self):
        super().build_loader()
        dataDir = self.config.get("data_dir")
        batch_size = self.config.get("batch_size")

        train_set = ShapeNetPartNormalDataset_Sample(data_dir=dataDir,\
                npoints=self.config.get("npoints"), split="trainval", normalize=self.config.get("normalize"))
        valid_set = ShapeNetPartNormalDataset_Sample(data_dir=dataDir,\
                npoints=self.config.get("npoints"), split="test", normalize=self.config.get("normalize"))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        return train_loader, valid_loader

    @utils.capture_time_cost
    def train(self, **args):
        self.model.train()
        train_iterator = tqdm(enumerate(self.train_loader), total=len(self.train_loader), unit='batch')
        loss_train, train_acc = 0, 0
        self.model.train()
        sum_acc, batch_nums = [], len(self.train_loader)
        train_len = len(self.train_loader.dataset)
        for i, (point_clouds, object_label, part_label) in train_iterator:
            point_clouds[:, :, 0:3] = utils.random_scale_point_cloud(point_clouds[:, :, 0:3])
            point_clouds[:, :, 0:3] = utils.shift_point_cloud(point_clouds[:, :, 0:3])
            point_clouds, part_label = point_clouds.cuda(), part_label.long().cuda()

            predict_part, _ = self.model(point_clouds.permute(0, 2, 1), utils.make_one_hot(object_label.cuda(), 16))
            loss = self.criterion(predict_part.permute(0, 2, 1), part_label)
            loss_train += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                batch_correct = (torch.argmax(predict_part, dim=2) == part_label).float().mean(1)
                batch_correct = batch_correct.mean()
                sum_acc.append(batch_correct.item())

            train_iterator.set_description("train epoch {0}/{1},batch {2}/{3}".format(args.get("epoch"), args.get("nEpochs"), i+1, batch_nums))
        # learning rate decay
        if self.scheduler:
            self.scheduler.step()  # lr_decay
        now_lr = self.optimizer.param_groups[0]['lr']

        loss_train /= train_len
        train_acc = np.mean(sum_acc)
        return {"now_lr" : now_lr, "loss_train" : loss_train, "epoch" : args.get("epoch"), "train_acc" : train_acc}

    @utils.capture_time_cost
    @torch.no_grad()
    def valid(self, **kw):
        self.model.eval()
        valid_len = len(self.valid_loader.dataset)  # valid samples
        cat_ious = {cat_ : [] for cat_ in seg_classes.keys()}  # each category iou
        gt_list, pre_correct_list= np.zeros(NUM_PARTS), np.zeros(NUM_PARTS)  
        category_iou_list = {cat : [] for cat in seg_classes.keys()}
        loss_valid, valid_acc = 0, np.zeros(2)
        valid_metrics = {}
        valid_iter = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), unit="sample", smoothing=0.9)
        for batch_id, (point_clouds, object_label, part_label) in valid_iter:
            valid_pre, valid_context = self.model(point_clouds.cuda().permute(0, 2, 1), utils.make_one_hot(object_label.cuda(), 16))
            valid_pre = valid_pre.permute(0, 2, 1)
            loss_valid += self.criterion(valid_pre, part_label.long().cuda()).item()
            valid_pre = valid_pre.detach().cpu().numpy()
            part_label = part_label.numpy()
            acc_array, cat_ious, gt_part, pre_part = calculate_batch_iou(valid_pre, part_label)
            for cat, iou_list in cat_ious.items():
                category_iou_list.get(cat).extend(iou_list)     # 每个category的列表存储着该类每个样本的iou
            valid_iter.set_description("test phase,valid batch {0}/{1}".format(batch_id + 1, len(self.valid_loader)))
            gt_list = gt_list + gt_part
            pre_correct_list = pre_correct_list + pre_part
            valid_acc += acc_array
        valid_metrics["accuracy_PartAvg"] = np.mean(pre_correct_list / gt_list) # 每个part上的召回率(Recall)
        valid_metrics["accuracy_valid"] = (valid_acc[0] / valid_acc[1], 1)  # 验证集上准确率(accuracy)
        valid_metrics["loss_valid"] = (loss_valid / valid_len, -1)
        all_cats_iou = []
        for cat, iou_list in category_iou_list.items():
            all_cats_iou.extend(iou_list)  # 保存所有样本的iou  用于计算ins.mIoU
            category_iou_list[cat] = np.mean(iou_list)  # 计算每个category的iou  cat.mIoU

        # 数据均值 和 数据分组(不等分) ，对每组求均值，然后对组均值再求均值   二者不一样，结果也不同
        # 如果数组分组的均值  乘以分组权重(改组数据量/总数据量) 得到结果就和全部均值相同
        valid_metrics["mIoU_category"] = (np.mean(list(category_iou_list.values())), 1)   # category mIoU
        valid_metrics["mIoU_instance"] = (np.mean(all_cats_iou), 1)                      # instance mIoU
        
        return valid_metrics, category_iou_list

    def save_model(self, valid_metrics, best_metrics, epoch):
        remark, best_metrics = self.update_sota(valid_metrics, best_metrics, epoch)
        if remark is None:
            self.patience += 1
            if self.patience > EARLY_STOPPING_PATIENCE:
                print("Early stopping with best cat.mIoU:{:<5.3f}, best ins.mIoU:{:<5.3f} and validation cat.mIoU:{} " \
                    "ins.mIoU:{:<5.3f} for this epoch: {:<5d}..\nEarly Stoping..".format(best_metrics["mIoU_category"], \
                             best_metrics["mIoU_instance"], valid_metrics["mIoU_category"][0], valid_metrics["mIoU_instance"][0], epoch))
                return 'no', True, best_metrics
            else:
                return "no", False, best_metrics
        self.patience = 0
        checkpoint = {
            "net" : self.model.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
            "epoch" : epoch,
            "best_metrics" : best_metrics
        }
        if self.scheduler:
            checkpoint.update(lr_scheduler=self.scheduler.state_dict())
        model_save_dir = self.config.get("model_save_dir")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        model_path = os.path.join(model_save_dir, "model_" + str(epoch) + ".pth")
        try:
            torch.save(checkpoint, model_path)
            print("model save in epoch %d" % epoch)
        except:
            print("model save failed in epoch %d!" % epoch)
        return remark, False, best_metrics

    def end_train(self, best_metrics=None):
        if best_metrics is None:
            best_metrics = self.best_metrics
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("\nTrain end at {}, all info saved in {}\n" \
            "Overview:\nbest cat.mIoU : {} at epoch {}\n" \
                "best ins.mIoU : {} at epoch {}".format(end_time, self.config.get("ckpt_folder"), best_metrics.get("mIoU_category"),\
                    best_metrics.get("best_mIoU_category_epoch"), best_metrics["mIoU_instance"], best_metrics["best_mIoU_instance_epoch"]))
        use_file_list = ["model_" + str(best_metrics.get("best_mIoU_category_epoch")) + ".pth",
                         "model_" + str(best_metrics.get("best_mIoU_instance_epoch")) + ".pth"]
        best_metrics.update({"End Time" : end_time})
        self.log.write_json("config", best_metrics)
        utils.delete_useless_files(self.config.get("model_save_dir"), use_file_list, 'pth')
        return super().end_train()
    

    