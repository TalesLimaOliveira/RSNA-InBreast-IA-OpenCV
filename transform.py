import pandas as pd
import os
import shutil
import random

df = pd.read_excel('~/Downloads/archive/inbreast/INbreast.xls')
database_path = '/media/alunotgn/0e7a0a9e-ad6d-4cef-a692-d7003b444387/datasets/processed/classification/rsna-breast-cancer-detection/ROI/'
final_database_path = '/media/alunotgn/0e7a0a9e-ad6d-4cef-a692-d7003b444387/datasets/final/'

files = os.listdir(database_path)
for row in df.iterrows():
    try:
        bi_rads = str(row[1]['Bi-Rads'])[0]
        file_name = str(int(row[1]['File Name']))
    except ValueError:
        continue
    os.makedirs(final_database_path+bi_rads, exist_ok=True)
    for file in files:
        if file.startswith(file_name):
            shutil.copy2(os.path.join(database_path, file), os.path.join(final_database_path+bi_rads, file))
            break

# create a train and test folder with 80% and 20% of the data
random.seed(42)
os.makedirs(final_database_path + 'train', exist_ok=True)
os.makedirs(final_database_path + 'test', exist_ok=True)
for folder in os.listdir(final_database_path):
    if folder == 'train' or folder == 'test':
        continue
    os.makedirs(final_database_path + 'train/' + folder, exist_ok=True)
    os.makedirs(final_database_path + 'test/' + folder, exist_ok=True)
    files = os.listdir(final_database_path+ folder)
    random.shuffle(files)
    for file in files[:int(0.8*len(files))]:
        shutil.copy2(os.path.join(final_database_path+folder, file), os.path.join(final_database_path + 'train/'+folder, file))
    for file in files[int(0.8*len(files)):]:
        shutil.copy2(os.path.join(final_database_path+folder, file), os.path.join(final_database_path+ 'test/'+folder, file))
    
