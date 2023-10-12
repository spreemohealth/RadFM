import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
import numpy as np
from dataclasses import dataclass, field
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torchvision import transforms
from PIL import Image   
from pydicom_series import read_files
import os 
from PIL import Image
import time
import pandas as pd
from MhdHelpers.mhd_handler import MhdHandler
from cvtoolsaugmentations.transforms import PercentileIntensityCutOff, PadToSquare, NormalizeIntensityVolume, FixSlices, ReduceSliceResolution
from albumentations import Compose

def get_tokenizer(tokenizer_path, max_img_size = 100, image_num = 32):
    '''
    Initialize the image special tokens
    max_img_size denotes the max image put length and image_num denotes how many patch embeddings the image will be encoded to 
    '''
    if isinstance(tokenizer_path,str):
        image_padding_tokens = []
        text_tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path,
        )
        special_token = {"additional_special_tokens": ["<image>","</image>"]}
        for i in range(max_img_size):
            image_padding_token = ""
            
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append("<image"+str(i*image_num+j)+">")
            image_padding_tokens.append(image_padding_token)
            text_tokenizer.add_special_tokens(
                special_token
            )
            ## make sure the bos eos pad tokens are correct for LLaMA-like models
            text_tokenizer.pad_token_id = 0
            text_tokenizer.bos_token_id = 1
            text_tokenizer.eos_token_id = 2    
    
    return  text_tokenizer,image_padding_tokens    

def combine_and_preprocess(question,image_list,image_padding_tokens):
        
        #add code for 3D

    transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512,512],scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
    images  = []
    new_qestions = [_ for _ in question]
    padding_index = 0
    for img in image_list:
        img_path = img['img_path']
        position = img['position']
        
        
        image = Image.open(img_path).convert('RGB')   
        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1) # c,w,h,d
        
        ## pre-process the img first
        target_H = 512 
        target_W = 512 
        target_D = 4 
        # This can be different for 3D and 2D images. For demonstration we here set this as the default sizes for 2D images. 
        images.append(torch.nn.functional.interpolate(image, size = (target_H,target_W,target_D)))
        
        ## add img placeholder to text
        new_qestions[position] = "<image>"+ image_padding_tokens[padding_index] +"</image>" + new_qestions[position]
        padding_index +=1
    
    vision_x = torch.cat(images,dim = 1).unsqueeze(0) #cat tensors and expand the batch_size dim
    text = ''.join(new_qestions) 
    return text, vision_x, 

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

def combine_and_preprocess_3D(question,image_list,image_padding_tokens):
    new_qestions = [_ for _ in question]
    padding_index = 0
    
    test_trnsfrms = [
        ReduceSliceResolution(target_mm_spacing=4.0),
        FixSlices(nTargetSlice=24),
        PercentileIntensityCutOff(1, 99),
        NormalizeIntensityVolume(),
        PadToSquare(value="minimum", size=256, keep_offset=True),
    ]

    images = []
    for ii in image_list:
        img_path = ii['img_path']
        position = ii['position']

        # print(img_path)
        if "mhd" in img_path:
            mhd = MhdHandler(img_path)
            img = mhd.raw_data.astype('float32')
            # print(img.shape)
            # print(np.min(img), np.max(img))
            
            metadata = metadata_as_dict(mhd)
            metadata['DimSize'] = metadata['Dimsize']
            
            img = transform_volume(img, composed_transforms=Compose(test_trnsfrms), metadata=metadata) 
            # print(img.shape)
            # print(np.min(img), np.max(img))
            

            
        elif "dicom" or "dcm" in img_path:
            all_series = read_files(img_path, False, False)
            for i in all_series:
                img = i.get_pixel_array()
                
        img = (img - np.min(img))/ (np.max(img) - np.min(img))
        # print('*'*100)
        # print('after transformation')
        # print(img.shape)
        # print(np.min(img), np.max(img))

        
        img = torch.from_numpy(img).unsqueeze(0).repeat(3,1,1,1).unsqueeze(0)
#         D = img.shape[0]
#         H, W = img.shape[1], img.shape[2]
#         n_D = 4-(D%4)
#         img = np.concatenate([img, np.zeros((n_D,H,W))])

#         img1 = torch.transpose(torch.tensor(img), 0, 1)
#         img2 = torch.transpose(torch.tensor(img1), 1, 2)
#         img3 = torch.nn.functional.interpolate(torch.tensor(img2).unsqueeze(0).unsqueeze(0), size = (512, 512, 12))
#         img3 = img3.repeat(1,3,1,1,1)
        
        # print('*'*100)
        # print('finally')
        # print(img.shape)
        # print(torch.min(img), torch.max(img))
        
        
        images.append(img.type(torch.float32))

            ## add img placeholder to text
        new_qestions[position] = new_qestions[position] + "<image>" + image_padding_tokens[padding_index] + "</image>"
        padding_index +=1
    
    vision_x = torch.cat(images,dim = 0).unsqueeze(0) #cat tensors and expand the batch_size dim
    text = ''.join(new_qestions)
    print(text)
    
    return text, vision_x, 


    # pad the dept to multiple of 4
    # interpolate to 512
       
    
    
def main():
    
    print("Setup tokenizer")
    text_tokenizer,image_padding_tokens = get_tokenizer('./Language_files')
    
    # print('image_padding_tokens: ',image_padding_tokens)
    print("Finish loading tokenizer")
    
    df_for_dl = pd.read_pickle("~/kawshik/multimodal/dataframes/df_for_dl_no_crop.pkl")
    df_for_dl['ID'] = df_for_dl['study_id']

    columns = [x for x in df_for_dl.columns if "x_" in x]
    for col in columns:

        df_for_dl[col] = df_for_dl.apply(lambda x: '/'.join(['/mnt','imaging','KneeNoCrops',x[col].split("/")[-1].split("_")[0],x[col].split("/")[-1].split(".")[0], x[col].split("/")[-1]]) if not pd.isna(x[col]) else x[col], axis=1)

    df_for_dl = df_for_dl.dropna(subset=['raw_text'])


    
    remove_cases = pd.concat([pd.read_csv("~/kawshik/multimodal/dataframes/negative_spacing_cases.csv"),pd.read_csv("~/kawshik/multimodal/dataframes/too_big_spacing_cases.csv")])

    df_for_dl = df_for_dl[~df_for_dl.study_id.isin(remove_cases.study_id.tolist())]
    
    print("Setup Model")
    model = MultiLLaMAForCausalLM(
        lang_model_path='./Language_files', ### Build up model based on LLaMa-13B config
    )
    ckpt = torch.load('/mnt/team_s3_synced/msandora/RadFM//pytorch_model.bin',map_location ='cpu') # Please dowloud our checkpoint from huggingface and Decompress the original zip file first
    
    
    model.load_state_dict(ckpt)
    
    for n, p in model.named_parameters():
        print(n,p.dtype,p.requires_grad)
        break
    
    model = model.to('cuda')
    model.eval() 
    
    print("Finish loading model")

    while True:
        sample = df_for_dl.sample().iloc[0]
        while type(sample['x_SagFS']) == float or sample['x_CorFS'] == float:
            sample = df_for_dl.sample().iloc[0]
        
    # print(sample)

        ### Initialize a simple case for demo ###
        print("Setup demo case")

        # for question in ["What disease can be diagnosed from these radiological images and what specific features are typically observed on the images?", "Please generate a radiology report from these scans", "What can you find from these scans", "Describe the findings and impressions from the following scans", "Please caption these scans with finding and impression", "Please make diagnosis based on these images", "What is the modality of these images"]:
        while True:


            print('raw_text: ',sample["raw_text"])

            # print("question: ", question)

            question = input("enter question (press 0 to exit)") #"What can you find from the scans ?"
            # question = "What disease can be diagnosed from these radiological images and what specific features are typically observed on the images?"

            if question == "0":
                break

            image =[
                    {
                        'img_path': sample['x_SagFS'],
                        'position': len(question)-1, #indicate where to put the images in the text string, range from [0,len(question)-1]
                    }, # can add abitrary number of imgs
                    {
                        'img_path': sample['x_CorFS'],
                        'position': len(question)-1, #indicate where to put the images in the text string, range from [0,len(question)-1]
                    }, # can add abitrary number of imgs
                ] 

            text,vision_x = combine_and_preprocess_3D(question,image,image_padding_tokens) 
            # print(text)
            # print(vision_x.shape)

            print("Finish loading demo case")

            with torch.no_grad():
                lang_x = text_tokenizer(
                        text, max_length=2048, truncation=True, return_tensors="pt"
                )['input_ids'].to('cuda')

                vision_x = vision_x.to('cuda')
                generation = model.generate(lang_x,vision_x)
                generated_texts = text_tokenizer.batch_decode(generation, skip_special_tokens=True) 

                print('Input: ', question)
                print('Output: ', generated_texts[0])
            
        

if __name__ == "__main__":
    main()
       
