#-*- coding:utf-8 -*-
'''
Author       : ZHP
Date         : 2021-12-09 19:37:44
LastEditors  : ZHP
LastEditTime : 2022-02-15 14:01:42
FilePath     : /trainer/trainer_basic.py
Description  : 
Copyright 2021 ZHP, All Rights Reserved. 
2021-12-09 19:37:44
'''
import traceback
from sympy import N
import torch
import torch.nn
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models
from tqdm import tqdm
import json
import time
from torch.utils.data import DataLoader, Dataset
from functools import reduce
import numpy as np
from utils import utils
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

EARLY_STOPPING_PATIENCE = 60

class Trainer():
    def __init__(self, config, criterion, model_config) -> None:
        self.start_time, self.config = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), config
        self.config_log(config.setdefault("resume", False), config, model_config)

        self.model = getattr(models, self.config.get("model_name"))(**self.config["model_parameters"])
        self.optimizer, self.scheduler = self.config_optimizer()
        self.train_loader, self.valid_loader = self.build_loader()
        self.criterion, self.resume, self.patience = criterion, config.get("resume"), 0

        self.init_weight_(self.config.get("init"))
        self.print_structure()
        self.config_board()
        self.config_wandb()

    def config_wandb(self):
        # self.data_queue = mp.Queue()
        # self.wandb = mp.Process(target=self.upload_data)
        # self.wandb.start()
        try:
            import wandb

            wandb.init(config=self.config, project=self.config.get("task", "default"), name=self.config.get("name", "defatultName"),)
            self.wandb = True
        except:
            self.wandb = False
        
    def upload_data(self):
        # TODO: 多进程方式,如何实现共享变量wandb
        # while True:
        #     valid_metrics = self.data_queue.get(True)
        #     for k, v in valid_metrics.items():
        #         if isinstance(v, tuple):
        #             valid_metrics[k] = v[0]
        #     wandb.log(valid_metrics)
        pass
    
    def upload_wandb(self, kw):
        if not self.wandb:
            return
        if self.best_metrics:
            for b_key, b_value in self.best_metrics.items():
                wandb.run.summary[b_key] = b_value
        ori_keys = list(kw.keys())
        for k in ori_keys:
            v = kw[k]
            if "_" in k:
                new_k = "/".join(k.split("_"))
                kw[new_k] = v
                kw.pop(k)
                k = new_k
            if isinstance(v, tuple):
                kw[k] = v[0]
        wandb.log(kw)


    def config_log(self, resume, config, model_config=None):
        '''
        Author: ZHP
        func: 配置训练日志
        param {bool} resume ： 是否断点恢复
        param {dict} config ： 训练配置
        '''    
        if resume:
            dir_path = config.get('ckpt_folder')
            self.log = Logs(dir_path=dir_path, \
                logs_path={
                    "config" : os.path.join(dir_path, "config.json"),
                    "trainLog" : os.path.join(dir_path, "trainLog.txt"),
                    "cat_iou" : os.path.join(dir_path, "cat_iou.txt"),
                    "model_structure" : os.path.join(dir_path, "model_structure.txt")
                })
            if os.path.exists(self.log.logs_path.get("config")):
                self.config = self.log.read_json("config")
                self.config["resume"] = True
            else:
                print("the resume log folder [{}]not exist".format(self.log.logs_path.get("config")))
                sys.exit(0)
        else:
            folder_suffix = "train_at_" + self.start_time[5:-3].replace(":", "_").replace(" ", "_").replace("-", "_")
            self.config["name"] = folder_suffix
            project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dir_path = os.path.join(project_folder, config.get("ckpt_folder", "Results/TrainLogs"), self.config.get("task", "noTask"), folder_suffix)
            self.log = Logs(dir_path=dir_path, \
                logs_path={
                    "config" : os.path.join(dir_path, "config.json"),
                    "trainLog" : os.path.join(dir_path, "trainLog.txt"),
                    "cat_iou" : os.path.join(dir_path, "cat_iou.txt"),
                    "model_structure" : os.path.join(dir_path, "model_structure.txt")
                })
            self.config["ckpt_folder"] = dir_path
            self.config["model_save_dir"] = os.path.join(project_folder, config.get("model_save_dir"), self.config.setdefault("task", "noTask"), folder_suffix)
            self.config["tensorboard"]  = os.path.join(project_folder, "Results/tensorboard", self.config["task"], folder_suffix)
            self.config["Train Start Time"], self.config["model_parameters"]= self.start_time, model_config
            self.log.write_json("config", self.config)
    
    def config_gpus(self):
        """config GPUs"""
        self.gpus = list(map(int, self.config["gpus"].split(",")))          # [1,3]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, list(range(self.gpus[0], min(4, self.gpus[-1] + 1))))))    # "1,2,3"  
        # 恢复为相对"CUDA_VISIBLE_DEVICES"的显卡索引，因为多卡并行时，有个相对的主卡:CUDA_VISIBLE_DEVICES中第一个
        self.gpus = (np.array(self.gpus) - self.gpus[0]).tolist()           # [0, 2]
        if torch.cuda.device_count() > 1 and len(self.gpus) > 1:
            print("Let's use", len(self.gpus), "GPUs!")
            self.model = torch.nn.DataParallel(self.model.cuda(), device_ids=self.gpus)
            # device_ids应是CUDA_VISIBLE_DEVICES中的相对索引，而不是真实的显卡编号
        else:
            self.model = self.model
        print("The current program allows the use gpus id : \"{}\" , now use gpus relative index: {}\n".format(os.environ["CUDA_VISIBLE_DEVICES"], self.gpus))
        self.model = self.model.cuda()

    def config_board(self):
        """config tensorboard"""
        tensor_dir = self.config["tensorboard"]
        if not os.path.exists(tensor_dir):
            os.makedirs(tensor_dir)
        self.board = SummaryWriter(log_dir=tensor_dir)
        print('\n可通过    tensorboard --logdir={}    来查看训练实时数据\n'.format(tensor_dir))

    def config_optimizer(self):
        kw = self.config
        lr, decay_ratio = kw.get("learning_rate", 0.05), kw.get("decay_ratio", 0.5)
        max_ratio = kw.get("min_lr", 0.001) / lr
        ''' build optimizer'''
        if kw.get('optimizer', 'Adam') == "SGD":
            optimizer = getattr(torch.optim, "SGD")(self.model.parameters(), \
                lr=lr, momentum=0.9 ,weight_decay=kw.get("weight_decay", 1e-4))
        else:
            optimizer = getattr(torch.optim, kw.get('optimizer', 'Adam'))(self.model.parameters(), \
                lr=lr, weight_decay=kw.get('weight_decay', 1e-4))

        if kw.get('lr_decay', False):
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: max(max_ratio, decay_ratio ** (epoch // kw.get("lr_step", 40))))
            return optimizer, scheduler
        else:
            return optimizer, None

    def build_loader(self):
        print("=====================  Loading datasets  =============================================\n")
        return None, None

    def run(self, **kw):
        start_epoch, nEpochs = self.prepare()
        print('====================  start train  ==================================================\n')
        for epoch in range(start_epoch, nEpochs + 1):
            self.train(epoch=epoch, nEpochs=nEpochs+1)
            self.valid()

    def prepare(self, **kw):
        self.best_metrics = None
        pass

    @utils.capture_time_cost
    def train(self, **args):
        self.model.train()
        """
        train phase
        """
        pass 
    @torch.no_grad()
    def valid(self, *args, **kw):
        self.model.eval()
        """
        validation phase
        """
        self.save_info()
        pass

    def save_info(self):
        pass 
            
    
    def save_model(self, valid_metrics, best_metrics, epoch):
        '''save the model when valid performence improve'''
        """
        if improved:
            update best_metrics
        else:
            self.patience += 1
            if self.patience > EARLY_STOPPING_PATIENCE:
                stop train
            return
        
        save model 
        """
        pass

    def end_train(self):
        pass


    def print_structure(self):
        if self.config["resume"]:
            return
        file_path = self.log.logs_path.setdefault("model_structure", "model_structure.txt")
        with open(file_path, "a") as f:
            print(self.model, file=f)
        print("model structure save in {}.".format(file_path))

    def init_weight_(self, mode="xavier"):
        try:
            if mode == "kaiming":
                for m in self.model.modules():
                    if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear)):
                        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif mode == "xavier":
                for m in self.model.modules():
                    if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear)):
                        torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
            print("init mode = " + mode)
        except:
            print("init error")
            traceback.print_exc()
    
    def add_tensorboard(self, epoch, **kw):
        """update tensorboard"""
        for k, v in kw.items():
            if "_" in k:
                k = "/".join(k.split("_"))
            if isinstance(v, tuple):
                self.board.add_scalar(k, v[0], epoch)
            else:
                self.board.add_scalar(k, v, epoch)

    def update_sota(self, valid_metrics, best_metrics, epoch):
        sota_m = []
        inter_keys = valid_metrics.keys() & best_metrics.keys()
        for key in inter_keys:
            if (valid_metrics[key][0] - best_metrics[key]) * valid_metrics[key][1] > 0:
                sota_m.append(key)
                best_metrics[key], best_metrics["best_" + key + "_epoch"] = valid_metrics[key][0], epoch
        if len(sota_m) == 0:
            return None, best_metrics
        else:
            return " and ".join(sota_m) + " improved", best_metrics

    def trace_end(self):
        # if self.wandb is not None:
        #     time.sleep(60)
        #     self.wandb.terminate()
        #     print("sub process terminate")
        try:
            self.end_train()
        except:
            print("fail to complete train end process!")
        finally:
            traceback.print_exc()
    

        

class Logs():
    def __init__(self, dir_path, logs_path=None):
        '''
        Author: ZHP
        description: Logs配置
        param {str} dir_path : log目录
        param {str} logs_path : 文件路径
        '''    
        self.folder = dir_path
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.logs_path = logs_path

    def write_json(self, name, config):
        file_path = self.logs_path.setdefault(name, str(name) + ".json")
        try:
            with open(file_path, 'a') as f:
                json.dump(config, f, indent=4)
        except:
            print("write json at {} error!".format(name))
            traceback.print_exc()
    
    def read_json(self, name):
        file_path = self.logs_path.get(name)
        content = None
        if file_path:
            with open(file_path, 'r') as f:
                content = json.load(f)
        return content

    def write_txt(self, name, content):
        file_path = self.logs_path.setdefault(name, str(name) + ".txt")
        try:
            with open(file_path, 'a') as f:
                f.write(content)
        except:
            print("write txt file at {} error!".format(name))
            traceback.print_exc()

    def write_csv(self, name, *args):
        pass


    

if __name__ == '__main__':
    pass