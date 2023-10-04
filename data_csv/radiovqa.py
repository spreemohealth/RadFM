import os 
import csv 
import json
import pandas as pd 
from tqdm import tqdm 

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def json_csv(json_file,csv_file):
    with open(csv_file,'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path','question','answer'])
        json_data = read_json_file(json_file)
        for data in tqdm(json_data):
            image_path = data['npy_path']
            qa_list = data['qa_list']
            for qa_idx in qa_list:
                writer.writerow([image_path,qa_idx['question'],qa_idx['answer']])
    csvfile.close()
    
# json_file = './radiology_article_npy_test.json'
# csv_file = './radiology_vqa_test.csv'
# json_csv(json_file,csv_file)

# json_file = './radiology_article_npy_train.json'
# csv_file = './radiology_vqa_train.csv'
# json_csv(json_file,csv_file)

# json_file = '/Users/zxm/Desktop/ustc/Radio_VQA/processed_file/processed_jsons/processed_json_wm_all.json'
# save_csv_file = './radio_modality_all.csv'
# json_data = read_json_file(json_file)

# with open(save_csv_file,'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['image_path','answer'])
#     for data in tqdm(json_data):
#         image_list = data['image']
#         modality_list = data['image_modality']
#         aux_modality_list = data['aux_modality']
#         try:
#             for idx in range(len(image_list)):
#                 image = image_list[idx]
#                 modality = modality_list[idx]
#                 aux_modality = aux_modality_list[idx]
#                 if modality != None and modality in ['CT', 'MRI', 'X-ray', 'Annotated image', 'Ultrasound', 'Fluoroscopy', 'DSA (angiography)', 'Nuclear medicine', 'Mammography', 'Barium']:
#                     if  aux_modality != None and str(aux_modality) != 'None':
#                         answer = str(modality) + ',' + str(aux_modality)
#                     else:
#                         answer = str(modality)
#                     writer.writerow([image,answer])
#                 else:
#                     print(modality)
#         except:
#             pass 
        
# # 69368
# df = pd.read_csv('./radio_modality_all.csv')
# df[:69368].to_csv('./radio_modality_test.csv',index=False)
# df[69368:].to_csv('./radio_modality_train.csv',index=False)

# # 重新找出有article的例子，增加一列radiology_features
# json_file = './radiology_test.json'
# article_file = './articles.json'
# json_data = read_json_file(json_file)
# article_data = read_json_file(article_file)
# new_article_data = {}

# # Iterate over each item in the data list
# for item in article_data:
#     # Extract the article name as the key
#     article_name = item['article']
#     # Remove the 'article' key from the item dictionary
#     del item['article']
#     # Add the modified item to the new_data dictionary
#     new_article_data[article_name] = item

# # Save the modified data to a new JSON file
# with open('articles_resave.json', 'w') as f:
#     json.dump(new_article_data, f, indent=4)
    
# save_json_file = './radiology_article_test.json'
# save_data_dict = []
# for data_idx in json_data:
#     article_idx = data_idx['articles']
#     radiographic_features = []
#     for article in article_idx:
#         if article in new_article_data.keys():
#             radiographic_features.append(new_article_data[article]['chatgpt_radiographic_features'])
#     if len(radiographic_features) != 0:
#         data_idx['radiographic_features'] = radiographic_features
#         save_data_dict.append(data_idx)


# with open(save_json_file, 'w') as f:
#     json.dump(save_data_dict, f,indent=4)



json_file = './radiology_article_train.json'
save_json_file = './radiology_article_npy_train.json'
json_data = read_json_file(json_file)

df = pd.read_csv('npy_to_case.csv')
npy_path_list = df['npy_path']
nii_case_id_list = df['nii_case_id']
nii_case_num_list = df['nii_case_num']

save_data_dict = []
for data_idx in tqdm(json_data):
    save_data_idx = data_idx
    image_path = data_idx['image_path'][0]
    # nii_case_id,nii_case_num
    nii_case_id = image_path.split('/')[-3]
    nii_case_num = image_path.split('/')[-2]
    for idx in range(len(df)):
        if str(nii_case_id_list[idx]) == str(nii_case_id) and str(nii_case_num_list[idx]) == str(nii_case_num):
            npy_path = npy_path_list[idx]
            # npy_path = get_npy_path(nii_case_id,nii_case_num)
            save_data_idx['npy_path'] = npy_path
            save_data_dict.append(save_data_idx)
            break

with open(save_json_file, 'w') as f:
    json.dump(save_data_dict, f,indent=4)
    
    
json_file = './radiology_article_npy_test.json'
csv_file = './radiology_vqa_test.csv'
json_csv(json_file,csv_file)

json_file = './radiology_article_npy_train.json'
csv_file = './radiology_vqa_train.csv'
json_csv(json_file,csv_file)