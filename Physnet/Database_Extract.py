# type: ignore
import sys
from matplotlib.pyplot import winter
import os
import cv2
import time
import csv
from math import floor


def extract(database_path,extract_path,csv_writer):
    simulations=30
    renders=6
    start_time=time.time()
    data=4
    frames=60
    for i in range(data):
        #for n in range (renders):
            #image_dir=database_path+'/rendering_'+str(n+1)+'_'+str(i+1).zfill(2)+'/'
            image_dir=database_path+'data_'+str(i).zfill(2)+'/'
            for t in range (frames):
                #print ('image:',image_dir+str(t+1).zfill(4)+'.png')
                image=cv2.imread(image_dir+str(t+1).zfill(4)+'.png',0)
                cv2.imwrite(extract_path+'data_'+str(i)+'_'+str(t+1).zfill(4)+'.png',image)
                csv_writer.writerow(('data_'+str(i)+'_'+str(t+1).zfill(4)+'.png',i+31))
                #cv2.imwrite(extract_path+'rendering_'+str(n+1)+'_'+str(i+1).zfill(2)+'_'+str(t+1).zfill(4)+'.png',image)
                #csv_writer.writerow(('rendering_'+str(n+1)+'_'+str(i+1).zfill(2)+'_'+str(t+1).zfill(4)+'.png',i+1))
            print(f'Batch {i+1}/{data}: Time Elapsed: {time.time()-start_time}')
    print ('extract completed!')

database_path='./BayOptim_session/'
extract_path='./Database/'
if not os.path.exists(extract_path):
    os.makedirs(extract_path)
csv_writer=csv.writer(open('./explore.csv','a'))
#csv_writer.writerow(('Name','Simulating'))
extract(database_path,extract_path,csv_writer)