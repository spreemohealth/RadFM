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
    print(type(volume), volume.shape, volume.dtype)
    volume = np.transpose(volume, (1, 2, 0))
    # Augment/transform and change to torch type (Slice, Row, Col)
    # volume = composed_transforms(image=volume)['image'].type(torch.float)
    volume = composed_transforms(image=volume, **kwargs)['image']
    print(type(volume), volume.shape, volume.dtype)
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
        metadata = {
            "TransformMatrix": np.eye(3),
            "Offset": np.zeros(3),
            "ElementSpacing": np.ones(3),
            "Dimsize": np.zeros(3)
        }
    else:
        metadata = {
            "TransformMatrix": mhd.meta_data["TransformMatrix"].reshape(3, 3),
            "Offset": mhd.meta_data["Offset"],
            "ElementSpacing": mhd.meta_data["ElementSpacing"],
            "Dimsize": mhd.meta_data["DimSize"]
        }
    return metadata

class All_Combi_Dataset(Dataset):
    
    def __init__(self, all_combi_df_path, split=None, qtype=None, sample_num=None):
        
        print('split: ',split)
        
        self.df = pd.read_pickle(all_combi_df_path)
        self.df = self.df[self.df.split.str.contains(split)]
        if qtype is not None: 
            self.df = self.df[self.df.qtype == qtype]
        print('split: ', split, self.df.shape)
        
        if sample_num is not None:
            sampled_rows = []
            for pathology in self.df.pathology.unique():
                sample = self.df[self.df.pathology==pathology].groupby('answer').apply(lambda x: x.sample(100, random_state=8) if len(x) > 100 else x).reset_index(drop = True)
                sampled_rows.append(sample)
                
            self.df = pd.concat(sampled_rows)
        
        self.df = self.df.sample(frac=1, random_state=8).reset_index(drop=True)
        
        self.image_columns = [col for col in self.df.columns if 'x_' in col]
        
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
        return len(self.df)
    
    def load_mhd(self, img_path):

        mhd = MhdHandler(img_path)
        img = mhd.raw_data.astype('float32')
        metadata = metadata_as_dict(mhd)
        metadata['DimSize'] = metadata['Dimsize']

        img = transform_volume(img,
                               composed_transforms=Compose(self.transforms),
                               metadata=metadata)

        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        return img
    
    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_paths = []
        for col in self.image_columns:
            if type(row[col]) == str:
                image_paths.append(row[col])

        image_dict = []

        question = row['question']
        answer = row['answer']
        pathology = row['pathology']
        qtype = row['qtype']

        mhds_to_use = []
        corpd = False
        sagpd = False

        for image_path in image_paths:
            if "CorPDFS" in image_path:
                mhds_to_use.append("CorPDFS")
                corpd = True
            if "SagPDFS" in image_path:
                mhds_to_use.append("SagPDFS")
                sagpd = True

        if corpd is False:
            mhds_to_use.append("CorT2FS")
        if sagpd is False:
            mhds_to_use.append("SagT2FS")

        for image_path in image_paths:
            for mhd in mhds_to_use:
                if mhd in image_path:

                    image = self.load_mhd(image_path)
                    print(image.shape)

                    image_dict.append({
                        "image":
                        torch.from_numpy(image).unsqueeze(0).repeat(
                            3, 1, 1, 1),
                        "position": {
                            "question": len(question)
                        }
                    })

        return {
            "study_id": row['study_id'],
            "image_dict": image_dict,
            "question": question,
            "answer": answer,
            "qtype": qtype,
            "pathology": pathology
        }
    
    

class DfForDlDataset(Dataset):

    def __init__(self, nocrop_path, sep_qa_path, fred_daphne_path, split, pathology_choice=None):

        self.df = pd.read_pickle(nocrop_path)
        self.df = self.df[self.df.split.str.contains(split)]
        print('split: ', split, self.df.shape)

        self.sep_qa_df = pd.read_pickle(sep_qa_path)

        self.fred_daphne_df = pd.read_pickle(fred_daphne_path)
        self.fred_daphne_df['findings'] = self.fred_daphne_df[
            'findings'].apply(lambda x: "".join(x) if type(x) == list else x)

        self.image_columns = [col for col in self.df.columns if 'x_' in col]

        # self.df_crop = pd.read_csv(crop_path)

        logging.info("loaded dataset")

        self.study_ids = list(self.df.study_id.unique())
        
        self.pathology_choice = pathology_choice

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

        img = transform_volume(img,
                               composed_transforms=Compose(self.transforms),
                               metadata=metadata)

        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        return img

    def get_qa(self, row, choice=4):

        if choice is None:
            choice = np.random.choice([1, 2, 3, 4])

        fred_daphne_row = self.fred_daphne_df[self.fred_daphne_df.study_id ==
                                              row['study_id']]

        sep_qa_row = self.sep_qa_df[self.sep_qa_df.ID == row['study_id']]

        if choice == 1:
            question = "Describe the findings from the medical images you are provided with."

            findings = fred_daphne_row['findings']

            if type(findings) != str:
                answer = row['findings']
            else:
                answer = fred_daphne_row['findings']

        elif choice == 2:
            if len(fred_daphne_row) > 0:
                fred_daphne_row = fred_daphne_row.iloc[0]
            else:
                return self.get_qa(row, choice=1)
            if len(fred_daphne_row["findings_segments"]) > 0:
                topic = np.random.choice(
                    list(fred_daphne_row["findings_segments"].keys()))
                question = f"What are the findings for {topic.lower().replace(':', '')} from the given study?"
                answer = fred_daphne_row["findings_segments"][topic]
            elif len(fred_daphne_row["rawtext_segments"]) > 0:

                topics = fred_daphne_row["findings_segments"].keys()
                useful_topics = []
                for topic in topics:

                    f = 0
                    for word in [
                            "history", "systems", "findings", "mri", "-2",
                            "normal"
                    ]:
                        if word in topic.lower():
                            f = 1
                            break

                    if "is " == topic[:3].lower():
                        f = 1

                    if f == 0:
                        useful_topics.append(topic)

                if len(useful_topics) == 0:
                    return self.get_qa(row, choice=1)

                topic = np.random.choice(useful_topics)

                question = f"What are the findings for {topic.lower().replace(':', '')} from the given study?"
                answer = fred_daphne_row["rawtext_segments"][topic]
            else:
                return self.get_qa(row, choice=1)

        elif choice == 3 or choice == 4:
            if len(sep_qa_row) == 0:
                return self.get_qa(row, choice=2)
            else:
                sep_qa_row = sep_qa_row.iloc[0]

                if self.pathology_choice is None:
                    pathology = np.random.choice(list(sep_qa_row['sep_qa'].keys()))
                else:
                    pathology = self.pathology_choice

                if choice == 3:
                    question = sep_qa_row['sep_qa'][pathology]['findings'][
                        'question']
                    answer = sep_qa_row['sep_qa'][pathology]['findings'][
                        'answer']
                elif choice == 4:
                    question = sep_qa_row['sep_qa'][pathology]['severity'][
                        'question']
                    answer = sep_qa_row['sep_qa'][pathology]['severity'][
                        'answer']

        return question, answer

    def __getitem__(self, index):

        row = self.df.iloc[index]
        # print(row)

        image_paths = []
        for col in self.image_columns:
            # print(row[col], type(row[col]))
            if type(row[col]) == str:
                image_paths.append(row[col])
        # print(image_paths)

        image_dict = []

        question, answer = self.get_qa(row)

        mhds_to_use = []
        corpd = False
        sagpd = False

        for image_path in image_paths:
            # if type(image_path) == str:
            if "CorPDFS" in image_path:
                mhds_to_use.append("CorPDFS")
                corpd = True
            if "SagPDFS" in image_path:
                mhds_to_use.append("SagPDFS")
                sagpd = True

        if corpd is False:
            mhds_to_use.append("CorT2FS")
        if sagpd is False:
            mhds_to_use.append("SagT2FS")

        for image_path in image_paths:
            # print(image_path,
            #       ("SagT2FS" in image_path or "CorPDFS" in image_path))
            for mhd in mhds_to_use:
                if mhd in image_path:

                    # print('inside if', image_path)

                    image = self.load_mhd(image_path)

                    print(image.shape)

                    image_dict.append({
                        "image":
                        torch.from_numpy(image).unsqueeze(0).repeat(
                            3, 1, 1, 1),
                        "position": {
                            "question": len(question)
                        }
                    })

        # print(image_dict)

        return {
            "study_id": row['study_id'],
            "image_dict": image_dict,
            "question": question,
            "answer": answer,
        }


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

    def __init__(self, pickle_path):
        self.df = pd.read_pickle(pickle_path)

        self.df = self.df[self.df.body_part == 'KNEE']
        self.df = self.df[~self.df.findings.isna()]
        self.df['file_paths'] = self.df.file_paths.apply(lambda x: [
            "/mnt/imaging/KneeNoCrops/" + "/".join(x_i.split("/")[-3:])
            for x_i in eval(x)
        ])

        logging.info("loaded dataset")

        # self.cache = {}

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

        img = transform_volume(img,
                               composed_transforms=Compose(self.transforms),
                               metadata=metadata)

        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        return img

    def __getitem__(self, index):

        # if index in self.cache:
        #     return self.cache[index]
        # else:
        rows = self.df[self.df.study_id == self.study_ids[index]]

        image_paths = rows['file_paths'].tolist()
        images = []
        image_dict = []

        question = "Describe the findings from the medical images you are provided with."
        answer = "".join(rows.iloc[0]['findings'])

        for image_path in image_paths:
            # print(image_path,
            #       ("SagT2FS" in image_path[0] or "CorPDFS" in image_path[0]))
            if "SagT2FS" in image_path[0] or "CorPDFS" in image_path[0]:

                # print('inside if', image_path)
                mhd_path = image_path[[
                    i for i in range(len(image_path))
                    if image_path[i][-4:] == '.mhd'
                ][0]]
                # print(mhd_path)

                image = self.load_mhd(mhd_path)

                print(image.shape)
            

                image_dict.append({
                    "image":
                    torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1, 1),
                    "position": {
                        "question": len(question)
                    }
                })

        # print(image_dict)

        # self.cache[index] = {
        #     "image_dict": image_dict,
        #     "question": question,
        #     "answer": answer,
        # }

        
        return {
            "study_id": row['study_id'],
            "image_dict": image_dict,
            "question": question,
            "answer": answer,
        }
