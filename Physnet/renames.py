#type:ignore
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2
import time
'''
simulations=30
renders=6
start_time=time.time()
for i in range (simulations):
    for n in range (renders):
        file_name='./Sources_Original/colored_rendering_'+str(n+1)+'_'+str(i+1).zfill(2)+'/'
        target_name='./Target/rendering_'+str(n+1).zfill(2)+'_'+str(i+1).zfill(2)+'/'
        if not os.path.exists(target_name):
            os.makedirs(target_name)
        for t in range (60):
            #print (file_name+str(t+1).zfill(4)+'.png')
            image=cv2.imread(file_name+str(t+1).zfill(4)+'.png',0)
            cv2.imwrite(target_name+str(t+1).zfill(4)+'.png',image)
    print (f'{i}/{simulations} is finised! time:{time.time()-start_time}')
print ('finished!') 
'''
data=1
target_name='./data_00_target/'
if not os.path.exists(target_name):
    os.makedirs(target_name)

for i in range (data):
    file_name='./data_'+str(i).zfill(2)+'/'
    for t in range (60):
        #print (file_name+str(t+1).zfill(4)+'.png')
        image=cv2.imread(file_name+str(t+1).zfill(4)+'.png',0)
        cv2.imwrite(target_name+str(t+1).zfill(4)+'.png',image)
print ('finished!')