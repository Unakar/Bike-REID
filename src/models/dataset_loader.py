# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings
from enum import Enum

from fastreid.data.datasets.bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY

from models.configs.PersonalConfig import PersonalConfig

class DatasetType(Enum):
    Train = 0
    Query = 1
    Gallery = 2

@DATASET_REGISTRY.register()
class BikePerson(ImageDataset):
    """BikePerson.

    Reference:
        Yuan, Yuan & Zhang, Jianran & Wang, Qi. (2018). Bike-Person Re-identification: A Benchmark and A Comprehensive Evaluation. IEEE Access. PP. 1-1. 10.1109/ACCESS.2018.2872804. 

    URL: `<https://crabwq.github.io/pdf/2018%20Bike-Person%20Re-identification%20A%20Benchmark%20and%20A%20Comprehensive%20Evaluation.pdf>`_


    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = ''
    dataset_name = "BikePerson"

    def __init__(self, root='models/datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        personal_config=PersonalConfig()
        data_dir = personal_config.bike_person_dataset_dir
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(f'Directory {data_dir} not found.')

        train = lambda: self.extract_datasets(self.data_dir, DatasetType.Train)
        query = lambda: self.extract_datasets(self.data_dir, DatasetType.Query)
        gallery = lambda: self.extract_datasets(self.data_dir, DatasetType.Gallery)

        super(BikePerson, self).__init__(train, query, gallery, **kwargs)

    def extract_datasets(self, dir_path, dataset_type: DatasetType = DatasetType.Train):
        img_paths = glob.glob(osp.join(dir_path, 'cam*/Bike/Person_*/*.jpg'))
        folder_cam_pattern = re.compile(r'cam_(\d)_(\d)')
        pid_pattern = re.compile(r'Person_(\d+)')
        camid_pattern = re.compile(r'cam(\d+)_bike') 
        # 注：括号，指对正则表达式分组并记住匹配的文本

        data = []
        for img_path in img_paths:
            folder_camid_1, folder_camid_2 = folder_cam_pattern.search(img_path).groups() 
            # pid: person id, camid: camera id
            pid = pid_pattern.search(img_path).groups()[0]
            camid = camid_pattern.search(img_path).groups()[0]
            
            # 注：search(path).groups()会找到第一处匹配的文本，并将各括号内字符串返回为一个元组
            pid = int(folder_camid_1 + folder_camid_2 + pid)
            camid = int(camid)

            if pid == -1:
                continue  # junk images are just ignored
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0

            # 根据pid模10的余数，将数据分为训练集、查询参考集；查询、参考集按照camid来分
            if pid % 10 < 5:
                if dataset_type != DatasetType.Train:
                    continue
            elif camid % 2 == 0:
                if dataset_type != DatasetType.Query:
                    continue
            else:
                if dataset_type != DatasetType.Gallery:
                    continue
            
            '''
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)'''
            data.append((img_path, pid, camid))

        return data