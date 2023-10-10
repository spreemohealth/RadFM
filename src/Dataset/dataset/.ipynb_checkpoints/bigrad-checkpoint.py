import csv
import json
import logging
import os
import re
import difflib
import sys
import torch
import random
from abc import abstractmethod
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from PIL import Image

class BigRadDataset(Dataset):
    """_summary_
    Args:
        Dataset (_type_): caption task formulated as vqa task for Chestxray classification dataset
        csv_path (_type_): path to csv file
        img_root_dir (_type_): path to image root directory 
        prompt_json_file (_type_): path to json file containing caption prompts
    Output:
        Dict: {
            "image_dict": {"image": image, "position": {"question": 0}}, # image is a tensor of shape [c,w,h,d] [3,512,512,1], position is a dict, random choice of 0 or len(question)
            "question": question, # random choice of caption prompts
            "answer":answer, # caption
            }
    """
    def __init__(self,json_path):
        data_info = pd.read_json(json_path)[:10]
        data_info['image'] = "/mnt/team_s3_synced/msandora/"+data_info['image']
        self.img_path_list = data_info['image'].tolist()
        self.answer_list = data_info['answer'].tolist()
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512,512],scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])   

        self.question = data_info['question'].tolist()

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        try:
            image = Image.open(img_path).convert('RGB')   
            image = self.transform(image)
            image = image.unsqueeze(-1) # c,w,h,d
        except Exception as e:
            print('Exception: ', e)
            image = np.random.randn(3,512,512,4)
        
        answer = self.answer_list[index]
        question = self.question[index]
        image_dict = [{
                "image": image,
                "position": {
                    "question": len(question)
                }
            }]
        return {
            "image_dict": image_dict,
            "question": question,
            "answer":answer,
            }
