#pylint:skip-file
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2
import time

dim=(640,480)
def extract(database_path,extract_path):
    Phy=30
    Render=6
    start_time=time.time()
    for i in range(Phy):
        for n in range (Render):
            image_dir=database_path+'rendering_'+str(n+1)+'_'+str(i+1).zfill(2)+'/'
            for t in range(60):
                #print ('image:',image_dir+str(t+1).zfill(4)+'.png')
                image=cv2.imread(image_dir+str(t+1).zfill(4)+'.png',cv2.IMREAD_UNCHANGED)
                image=cv2.resize(image,dim,cv2.INTER_AREA)
                if not os.path.exists(extract_path+'rendering_'+str(n+1)+'_'+str(i+1).zfill(2)+'/'):
                    os.makedirs(extract_path+'rendering_'+str(n+1)+'_'+str(i+1).zfill(2)+'/')
                cv2.imwrite(extract_path+'rendering_'+str(n+1)+'_'+str(i+1).zfill(2)+'/'+str(t+1).zfill(4)+'.png',image)
        print(f'Batch {i+1}/{Phy}: Time Elapsed: {time.time()-start_time}')
    print ('extract completed!')

database_path='./Sources/'
extract_path='./Resized/'
if not os.path.exists(extract_path):
    os.makedirs(extract_path)
extract(database_path,extract_path)