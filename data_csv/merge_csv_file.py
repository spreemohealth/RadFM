# PMC_ID,Figure_path,Question,Answer
 
import pandas as pd

# # read in the two csv files
# df1 = pd.read_csv('pmcvqa_compound_test.csv')
# img_root_dir1 = '/home/cs/leijiayu/data/PMC_OA_papers/figures'
# #增加一个column为img_root_dir
# df1['img_root_dir'] = img_root_dir1
# df2 = pd.read_csv('pmcvqa_noncompound_test.csv')
# img_root_dir2 = img_root_dir = img_root_dir = '/home/cs/leijiayu/data/PMCVQA/caption_T060_filtered_top4_sep_v0_subfigures'
# df2['img_root_dir'] = img_root_dir2

# # merge the two dataframes on the common column
# merged_df = pd.concat([df1, df2])

# # write the merged dataframe to a new csv file
# merged_df.to_csv('pmcvqa_test.csv', index=False)


# # read in the two csv files
# df1 = pd.read_csv('pmcvqa_compound_train.csv')
# df1['img_root_dir'] = img_root_dir1
# df2 = pd.read_csv('pmcvqa_noncompound_train.csv')
# df2['img_root_dir'] = img_root_dir2

# # merge the two dataframes on the common column
# merged_df = pd.concat([df1, df2])

# # write the merged dataframe to a new csv file
# merged_df.to_csv('pmcvqa_train.csv', index=False)



# 读取csv文件
df = pd.read_csv('radio_train.csv')

# 删除caption列为空的行
df.dropna(subset=['image_caption'], inplace=True)

# 保存修改后的csv文件
df.to_csv('radio_train.csv', index=False)


df = pd.read_csv('radio_test.csv')

# 删除caption列为空的行
df.dropna(subset=['image_caption'], inplace=True)

# 保存修改后的csv文件
df.to_csv('radio_test.csv', index=False)

df = pd.read_csv('pmcvqa_test.csv')

# 删除caption列为空的行
df.dropna(subset=['Answer'], inplace=True)

# 保存修改后的csv文件
df.to_csv('pmcvqa_test.csv', index=False)

df = pd.read_csv('pmcvqa_train.csv')

# 删除caption列为空的行
df.dropna(subset=['Answer'], inplace=True)

# 保存修改后的csv文件
df.to_csv('pmcvqa_train.csv', index=False)