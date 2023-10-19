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
import logging
from cvtoolsaugmentations.transforms import PercentileIntensityCutOff, PadToSquare, NormalizeIntensityVolume, FixSlices, ReduceSliceResolution
from MhdHelpers.mhd_handler import MhdHandler
from albumentations import Compose

def transform_volume(volume, composed_transforms, **kwargs):
        """ perform transformations for classification
         Args:
         volume (np.ndarray): The shape should be (Row, Cols, Slice)
         composed_transforms (compose of list of transforms)
         kwargs: optional arguments. 
                For example, metadata={"TransformMatrix": np.eye(3),
                        "Offset": np.zeros(3),
                        "ElementSpacing": np.ones(3),
                        "Dimsize": np.zeros(3)}
           """
        # change to (Row, Col, Slice)
        # print(type(volume), volume.shape, volume.dtype)
        volume = np.transpose(volume,(1,2,0))
        # Augment/transform and change to torch type (Slice, Row, Col)
        # volume = composed_transforms(image=volume)['image'].type(torch.float) 
        volume = composed_transforms(image=volume, **kwargs)['image']
        # print(type(volume), volume.shape, volume.dtype)
        volume = volume.astype(float)
        
        return volume
    
def metadata_as_dict(mhd):
    """ Extract metadata information in the form of a dict.
        This dict is useful for metadata transformation.

    Args:
        mhd (MhdHandler): If None, then the metadata is filled with empty metadata.
                          This is useful when a sequence is missing.
    Returns:
        Dict: dictionary that contains the metadata information.
    """
    if mhd is None:
        metadata = {"TransformMatrix": np.eye(3),
                    "Offset": np.zeros(3),
                    "ElementSpacing": np.ones(3),
                    "Dimsize": np.zeros(3)}
    else:
        metadata = {"TransformMatrix": mhd.meta_data["TransformMatrix"].reshape(3,3),
                                "Offset": mhd.meta_data["Offset"],
                                "ElementSpacing": mhd.meta_data["ElementSpacing"],
                                "Dimsize": mhd.meta_data["DimSize"]}
    return metadata
    
class Internal3DDataset(Dataset):
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
    def __init__(self,pickle_path):
        self.df = pd.read_pickle(pickle_path)
        
        self.df = self.df[self.df.body_part == 'KNEE']
        self.df = self.df[~self.df.findings.isna()]
        self.df['file_paths'] = self.df.file_paths.apply(lambda x: ["/mnt/imaging/KneeNoCrops/" + "/".join(x_i.split("/")[-3:]) for x_i in eval(x)])

        logging.info("loaded dataset")
    
        self.study_ids = list(self.df.study_id.unique())
        
        self.transforms = [
                            ReduceSliceResolution(target_mm_spacing=4.0),
                            FixSlices(nTargetSlice=24),
                            PercentileIntensityCutOff(1, 99),
                            NormalizeIntensityVolume(),
                            PadToSquare(value="minimum", size=256, keep_offset=True),
                          ]
        
    def __len__(self):
        return len(self.study_ids)
    
    def load_mhd(self, img_path):
        
        mhd = MhdHandler(img_path)
        img = mhd.raw_data.astype('float32')
        metadata = metadata_as_dict(mhd)
        metadata['DimSize'] = metadata['Dimsize']

        img = transform_volume(img, composed_transforms=Compose(self.transforms), metadata=metadata) 
        
        img = (img - np.min(img))/ (np.max(img) - np.min(img))
        
        return img
        

    def __getitem__(self, index):
        
        rows = self.df[self.df.study_id == self.study_ids[index]]
        
        image_paths = rows['file_paths'].tolist()
        images = []
        image_dict = []
        
        question = "Describe the findings from the medical images you are provided with."
        answer = "".join(rows.iloc[0]['findings'])
        
        
        for image_path in image_paths:
            print(image_path, ("SagT2FS" in image_path[0] or "CorPDFS" in image_path[0]))
            if "SagT2FS" in image_path[0] or "CorPDFS" in image_path[0]:
                
                print('inside if',image_path)
                mhd_path = image_path[[i for i in range(len(image_path)) if image_path[i][-4:]=='.mhd'][0]]
                print(mhd_path)
                
                
                image = self.load_mhd(mhd_path)
                
                print(image.shape)

                image_dict.append({
                    "image": torch.from_numpy(image).unsqueeze(0).repeat(3,1,1,1),
                    "position": {
                        "question": len(question)
                    }
                })
    
        # print(image_dict)
                                
        return {
            "image_dict": image_dict,
            "question": question,
            "answer":answer,
            }
    
