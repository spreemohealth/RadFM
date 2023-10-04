import os 
import csv
import pandas as pd 
from tqdm import tqdm
#check the 'image_path' column in '../process/merge.csv', if the path is not exist, delete the row
# csv_file = './radiochat_train.csv'
# save_csv_file = './cd.csv'

# with open(csv_file, 'r') as file:
#     reader = csv.reader(file)
#     rows = [row for row in reader if all(row)]
    
# with open(csv_file, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(rows)
    
# with open(save_csv_file, 'r') as file:
#     reader = csv.reader(file)
#     rows = [row for row in reader if all(row)]
    
# with open(save_csv_file, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(rows)

 
# Read the CSV file and extract the required columns
import pandas as pd

df = pd.read_csv('radiochat_train.csv')
df = df[['image_caption', 'question', 'answer']]

# Strip the question and answer columns
df['question'] = df['question'].str.strip()
df['answer'] = df['answer'].str.strip()

# Save the cleaned data to a new CSV file
df.to_csv('radiochat_train.csv', index=False)



df = pd.read_csv('radiochat_test.csv')
df = df[['image_caption', 'question', 'answer']]

# Strip the question and answer columns
df['question'] = df['question'].str.strip()
df['answer'] = df['answer'].str.strip()

# Save the cleaned data to a new CSV file
df.to_csv('radiochat_test.csv', index=False)