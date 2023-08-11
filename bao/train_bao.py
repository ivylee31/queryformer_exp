import numpy as np
import os
import torch
import torch.nn as nn
import pandas as pd
from scipy.stats import pearsonr
import json


import bao_model 
from bao_util import Normalizer
import os
import shutil

print('========================= 加载数据库数据集 ==========================')

imdb_path = './imdb/'
full_train_df = pd.DataFrame()

df_json=[]
for i in range(18):
    file = imdb_path + 'plan_and_cost/train_plan_part{}.csv'.format(i)
    df = pd.read_csv(file)
    full_train_df=full_train_df._append(df)
    full_train_df_json=json.loads(full_train_df.to_json(orient='records'))
    for record in full_train_df_json:
      if 'BitmapAnd' not in record['json']:
        df_json.append(record['json'])
    
       
#print(df_json)

def train_and_save_model(fn, verbose=True):
 
    x = df_json
     #['{"Plan": {...}, "Buffers": {...}}']
    labels= [json.loads(plan)['Execution Time'] for plan in full_train_df['json'] if 'BitmapAnd' not in plan]
    cost_norm= Normalizer(-3.61192, 12.290855)
    y= torch.from_numpy(cost_norm.normalize_labels(labels))
    print("x,y ready")
    reg = bao_model.BaoRegression(have_cache_data=False, verbose=verbose)
    reg.fit(x, y)
    reg.save(fn)
    
    return reg




print('========================= 训练模型 ==========================')
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: train.py MODEL_FILE")
        exit(-1)
    train_and_save_model(sys.argv[1])

    print("Model saved, attempting load...")
    reg = bao_model.BaoRegression(have_cache_data=True)
    reg.load(sys.argv[1])


