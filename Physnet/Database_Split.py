# type: ignore
import sys
import numpy as np
import os
import cv2
import pandas as pd
import csv
import time

train_file='./train_file/'
test_file='./test_file/'
if not os.path.exists(train_file):
    os.makedirs(train_file)
if not os.path.exists(test_file):
    os.makedirs(test_file)

np.random.seed(42)

csv_path='./label.csv'
files=pd.read_csv(csv_path)

indices=np.zeros(len(files))
for i in range(int(len(files)*0.2)):
    indices[i]=1
indices=np.random.permutation(indices)

f_train=open('./train_file/target/target.csv','w')
f_test=open('./test_file/target/target.csv','w')
train_writer=csv.writer(f_train)
test_writer=csv.writer(f_test)
train_writer.writerow(('Name','Phy'))
test_writer.writerow(('Name','Phy'))
print('start!')
start_time=time.time()
for i in range(len(files)):
    if indices[i]==0:
        images=cv2.imread('./Database/'+files.iloc[i,0],0)
        phy=files.iloc[i,1]
        cv2.imwrite(train_file+'img/'+files.iloc[i,0],images)
        train_writer.writerow((files.iloc[i,0],files.iloc[i,1]))
    else:
        images=cv2.imread('./Database/'+files.iloc[i,0],0)
        phy=files.iloc[i,1]
        cv2.imwrite(test_file+'img/'+files.iloc[i,0],images)
        test_writer.writerow((files.iloc[i,0],files.iloc[i,1]))
    if i%int(len(files)/10)==0:
        print(f'Files:{i}/{len(files)} have been completed, time elapsed:{time.time()-start_time}')
f_train.close()
f_test.close()
print('Transformation Completed!')
