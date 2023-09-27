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

class RadModel():
    def __init__(self, model_path: str, device="cuda"):

        print("Setup tokenizer")
        self.text_tokenizer,self.image_padding_tokens = get_tokenizer('./Language_files')
        print("Finish loading tokenizer")

        print("Setup Model")
        model = MultiLLaMAForCausalLM(
            lang_model_path='./Language_files', ### Build up model based on LLaMa-13B config
        )
        ckpt = torch.load(model_path, map_location ='cpu') # Please dowloud our checkpoint from huggingface and Decompress the original zip file first
        model.load_state_dict(ckpt)
        print("Finish loading model")
        
        self.model = model.to('cuda')
        self.model.eval() 
    

    def combine_and_preprocess(self, question,image_list,image_padding_tokens):
        
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

    def combine_and_preprocess_3D(self, question,image_list,image_padding_tokens):
        new_qestions = [_ for _ in question]
        padding_index = 0

        images = []
        for ii in image_list:
            img_path = ii['img_path']
            position = ii['position']

            all_series = read_files(img_path, False, False)
            for i in all_series:
                img = i.get_pixel_array()
            img = (img - np.min(img))/ (np.max(img) - np.min(img))

            D = img.shape[0]
            H, W = img.shape[1], img.shape[2]
            n_D = 4-(D%4)
            img = np.concatenate([img, np.zeros((n_D,H,W))])

            img1 = torch.transpose(torch.tensor(img), 0, 1)
            img2 = torch.transpose(torch.tensor(img1), 1, 2)
            img3 = torch.nn.functional.interpolate(torch.tensor(img2).unsqueeze(0).unsqueeze(0), size = (512, 512, 12))
            img3 = img3.repeat(1,3,1,1,1)
            images.append(img3.type(torch.float32))

                ## add img placeholder to text
            new_qestions[position] = "<image>"+ image_padding_tokens[padding_index] +"</image>" + new_qestions[position]
            padding_index +=1
        
        vision_x = torch.cat(images,dim = 1).unsqueeze(0) #cat tensors and expand the batch_size dim
        text = ''.join(new_qestions) 
        return text, vision_x, 

    def __call__(self, question: str, ip_image):

        if os.path.isdir(image):
            text,vision_x = self.combine_and_preprocess_3D(question,image,self.image_padding_tokens) 
        else:   
            text,vision_x = self.combine_and_preprocess(question,image,self.image_padding_tokens) 
        print("Finish loading demo case")
        

        with torch.no_grad():
            lang_x = self.text_tokenizer(
                    text, max_length=2048, truncation=True, return_tensors="pt"
            )['input_ids'].to('cuda')
            
            vision_x = vision_x.to('cuda')
            generation = self.model.generate(lang_x,vision_x)
            generated_texts = self.text_tokenizer.batch_decode(generation, skip_special_tokens=True) 
            return {
                'prompt': question,
                'answer': generated_texts[0]}