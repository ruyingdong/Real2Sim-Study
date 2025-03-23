#type:ignore
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2

category=['black_denim','gray_interlock','white_tablecloth']

def database_reconst(source_file,target_file):
    for name_index in range (len(category)):
        for k in range (10):
            for n in range (6):
                dir=source_file+category[name_index]+'_ldls/'+'rendering_'+str(n+1)+'_'+str((k+1)*3).zfill(2)+'/'
                for i in range (60):
                    #print ('image:',dir+str(i+1).zfill(4)+'.png')
                    image=cv2.imread(dir+str(i+1).zfill(4)+'.png',0)
                    #print ('target:',str(n+1),name_index*10+k,(k+1)*3,i+1)
                    if not os.path.exists(target_file+'rendering_'+str(n+1)+'_'+str(name_index*10+(k+1)).zfill(2)+'/'):
                        os.makedirs(target_file+'rendering_'+str(n+1)+'_'+str(name_index*10+(k+1)).zfill(2)+'/')
                    cv2.imwrite(target_file+'rendering_'+str(n+1)+'_'+str(name_index*10+(k+1)).zfill(2)+'/'+category[name_index]
                    +'_ldls_'+str((k+1)*3).zfill(2)+'_'+str(i+1).zfill(4)+'.png',image)
        print (f'{category[name_index]}_has been finished!')
    print ('const has been finished!')

source_file='./sources_'
target_file='./Reconst_Database/'
if not os.path.exists(target_file):
    os.makedirs(target_file)
database_reconst(source_file,target_file)