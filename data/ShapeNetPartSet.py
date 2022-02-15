'''
Author       : ZHP
Date         : 2021-04-26 15:44:43
LastEditors  : ZHP
LastEditTime : 2021-05-28 19:33:26
FilePath     : \Projects\ptzhs\point_transformer\data\ShapeNetPartSet.py
Description  : 项目说明
Copyright    : ZHP
'''
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json

DATA_DIR = '/root/datasets/ShapeNet'

def pc_normalize(pc):
    '''
    pc : 点云数据 归一化点云
    '''
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc
    
# =========== ShapeNet Part =================
class ShapeNetPartNormalDataset(Dataset):
    def __init__(self, data_dir, npoints=2500, split='train', normalize=False):
        self.npoints = npoints  # 采样数
        self.split = split  # train/valid/test/trainval
        self.root = os.path.join(data_dir, 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {} # {'Airplane': '02691156','Bag': '02773838',...}
        self.normalize = normalize # 是否归一化

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()  # ['Airplane', '02691156'] ['Bag', '02773838']['Cap', '02954340']
                self.cat[ls[0]] = ls[1]  # 'Airplane': '02691156',
        self.cat = {k: v for k, v in self.cat.items()}
        
        self.meta = {}
        # 读取数据
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            #     content = json.load(f)
            #     ['shape_data/03624134/3d2cb9d291ec39dc58a42593b26221da','shape_data/02691156/ed73e946a138f3cfbc0909d98a1ff2b4']
            #     type(content), len(content) # (list, 12137)
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)]) # 3d2cb9d291ec39dc58a42593b26221da
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat: # 遍历键:Airplane Bag
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item]) # shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/
            fns = sorted(os.listdir(dir_point)) # 各个样本名 3d2cb9d291ec39dc58a42593b26221da.txt

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]  # 取名称，末尾.txt不取
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) # '1021a0914a7207aff927ed529ad90a11'
                self.meta[item].append(os.path.join(dir_point, token + '.txt')) # shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1021a0914a7207aff927ed529ad90a11.txt

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn)) # ('Airplane','/disk/dataset/ShapeNet/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1021a0914a7207aff927ed529ad90a11.txt'),
        self.classes = dict(zip(self.cat, range(len(self.cat))))  # cat名称和编号,'Airplane': 0, 'Bag': 1,
        
        # seg_classes代表某类物体的part label如'Airplane': [0, 1, 2, 3]表示Airplane有四部分，分别标记为0,1,2,3
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Tableshape': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}  # from index to (point_set, normal, cls, seg) tuple
        self.cache_size = 20000

    def __len__(self):
      return len(self.datapath)

    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index] 
        else:
            fn = self.datapath[index] # ('Airplane', '/disk/dataset/ShapeNet/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/106dfe858cb8fbc2afc6b80d80a265ab.txt')
            cat = self.datapath[index][0] # 'Airplane'
            object_cls = self.classes[cat] # 0 标签
            object_cls = np.array([object_cls]).astype(np.int32)  # [0]
            data = np.loadtxt(fn[1]).astype(np.float32) # (N, 7)
            point_set = data[:, 0:3] # 坐标XYZ
            normal = data[:, 3:6] # normal后的数据
            seg = data[:, -1].astype(np.int32) # part label  (2488, )
            if len(self.cache) < self.cache_size:  # 写入缓存区
                self.cache[index] = (point_set, normal, seg, object_cls)

        if self.normalize:
            point_set = pc_normalize(point_set)

        choice = np.random.choice(len(seg), self.npoints, replace=True) # 从0-len(seg)中抽样，取npoints个索引出来 可取相同数字
        if self.split == 'train' or self.split == 'trainval':
            # 抽样
            # resample
            # note that the number of points in some points clouds is less than 2048, thus use random.choice
            # remember to use the same seed during train and test for a getting stable result
            point_set = point_set[choice, :]
            seg = seg[choice]
            normal = normal[choice, :]
        # 由于每个样本点个数不一样，所以不能取点
        
        pc = np.concatenate([point_set, normal], axis=1)  
        # 返回点云数据(X,Y,Z,normal)--(4096,6)  类别(1,) 物体大类   part labels(4096,)
        
        return torch.from_numpy(pc).float(), torch.from_numpy(object_cls).float(), torch.from_numpy(seg).float()


class ShapeNetPartNormalDataset_Sample(Dataset):
    def __init__(self, data_dir, npoints=2500, split='train', normalize=False):
        self.npoints = npoints  # 采样数
        self.split = split  # train/valid/test/trainval
        self.root = os.path.join(data_dir, 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {} # {'Airplane': '02691156','Bag': '02773838',...}
        self.normalize = normalize # 是否归一化

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()  # ['Airplane', '02691156'] ['Bag', '02773838']['Cap', '02954340']
                self.cat[ls[0]] = ls[1]  # 'Airplane': '02691156',
        self.cat = {k: v for k, v in self.cat.items()}
        
        self.meta = {}
        # 读取数据
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            #     content = json.load(f)
            #     ['shape_data/03624134/3d2cb9d291ec39dc58a42593b26221da','shape_data/02691156/ed73e946a138f3cfbc0909d98a1ff2b4']
            #     type(content), len(content) # (list, 12137)
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)]) # 3d2cb9d291ec39dc58a42593b26221da
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat: # 遍历键:Airplane Bag
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item]) # shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/
            fns = sorted(os.listdir(dir_point)) # 各个样本名 3d2cb9d291ec39dc58a42593b26221da.txt

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]  # 取名称，末尾.txt不取
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) # '1021a0914a7207aff927ed529ad90a11'
                self.meta[item].append(os.path.join(dir_point, token + '.txt')) # shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1021a0914a7207aff927ed529ad90a11.txt

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn)) # ('Airplane','/disk/dataset/ShapeNet/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1021a0914a7207aff927ed529ad90a11.txt'),
        self.classes = dict(zip(self.cat, range(len(self.cat))))  # cat名称和编号,'Airplane': 0, 'Bag': 1,
        
        # seg_classes代表某类物体的part label如'Airplane': [0, 1, 2, 3]表示Airplane有四部分，分别标记为0,1,2,3
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Tableshape': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}  # from index to (point_set, normal, cls, seg) tuple
        self.cache_size = 20000

    def __len__(self):
      return len(self.datapath)

    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index] 
        else:
            fn = self.datapath[index] # ('Airplane', '/disk/dataset/ShapeNet/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/106dfe858cb8fbc2afc6b80d80a265ab.txt')
            cat = self.datapath[index][0] # 'Airplane'
            object_cls = self.classes[cat] # 0 标签
            object_cls = np.array([object_cls]).astype(np.int32)  # [0]
            data = np.loadtxt(fn[1]).astype(np.float32) # (N, 7)
            point_set = data[:, 0:3] # 坐标XYZ
            normal = data[:, 3:6] # normal后的数据
            seg = data[:, -1].astype(np.int32) # part label  (2488, )
            if len(self.cache) < self.cache_size:  # 写入缓存区
                self.cache[index] = (point_set, normal, seg, object_cls)

        if self.normalize:
            point_set = pc_normalize(point_set)

        choice = np.random.choice(len(seg), self.npoints, replace=True) # 从0-len(seg)中抽样，取npoints个索引出来 可取相同数字
        # 抽样
        # resample
        # note that the number of points in some points clouds is less than 2048, thus use random.choice
        # remember to use the same seed during train and test for a getting stable result
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]
        # 由于每个样本点个数不一样，所以不能取点
        
        pc = np.concatenate([point_set, normal], axis=1)  
        # 返回点云数据(X,Y,Z,normal)--(4096,6)  类别(1,) 物体大类   part labels(4096,)
        
        return torch.from_numpy(pc).float(), torch.from_numpy(object_cls).float(), torch.from_numpy(seg).float()






if __name__ == "__main__":
    folder = '/root/datasets/ShapeNet'
    dataset = ShapeNetPartNormalDataset(folder, 4096, 'train')
    valid_set = ShapeNetPartNormalDataset(folder, 4096, 'test')
    count = 0
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=10)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4)
    for pcs, object_cls, part_label in train_loader:
        print(f"pcs : {pcs.shape} | cls {object_cls.shape} | part {part_label.shape}")
        if count > 5:
            break
        else:
            count += 1
    print("len :", len(train_loader))
    max_size = 0
    count = 0
    for pcs, object_cls, part_label in valid_loader:
        max_size = max(max_size, pcs.shape[1])
        # print(f"pcs : {pcs.shape} | cls {object_cls.shape} | part {part_label.shape}")
        # if count > 5:
        #     break
        # else:
        #     count += 1
        if count == 10:
            print("\n")
            count = 0
        print("|{}".format(pcs.shape[1]), end=" ")
        count += 1
    print(f'max size-{max_size}')
    
    # print(len(dataset))
    # print(len(train_loader.dataset))
