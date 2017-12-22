# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 21:37:26 2017

@author: Shirelle
"""

from PIL import Image
import feature
import numpy as np
import pickle

face_dir="./datasets/original/face/face_"
nonface_dir="./datasets/original/nonface/nonface_"

for i in range(500):
        dir=face_dir+str("%.3d"%i)+".jpg"
        im=Image.open(dir).convert("L").resize((24,24))#L：灰色图像，数字表示不同的灰度
        NPD = feature.NPDFeature(np.array(im))
        if(i==0):
            X=NPD.extract() 
            y=np.array([1])
        else:   
            X=np.vstack((X,NPD.extract())) 
            y=np.vstack((y,np.array([1])))  
            
for i in range(500):
        dir=nonface_dir+str("%.3d"%i)+".jpg"
        im=Image.open(dir).convert("L").resize((24,24))
        NPD = feature.NPDFeature(np.array(im))
        X=np.vstack((X,NPD.extract()))
        y=np.vstack((y,np.array([-1])))


cache_X='X.pkl'
X_out=open(cache_X,'wb')
pickle.dump(X,X_out)
X_out.close()

cache_y='y.pkl'
y_out=open(cache_y,'wb')
pickle.dump(y,y_out)
y_out.close()
