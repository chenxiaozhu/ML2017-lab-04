# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:29:41 2017

@author: Shirelle
"""

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from ensemble import AdaBoostClassifier
import sklearn.tree

from sklearn.metrics import classification_report

if __name__ == "__main__":
    # write your code here
   X_file='X.pkl'
   X_in=open(X_file,'rb')
   X=pickle.load(X_in)
   X_in.close()

   y_file='y.pkl'
   y_in=open(y_file,'rb')
   y=pickle.load(y_in)
   y_in.close()


   X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, 
                                                        test_size=0.2,
                                                        random_state=0)
   classifier = AdaBoostClassifier(sklearn.tree.DecisionTreeClassifier,4)
   classifier.fit(X_train, y_train)

   y_pred=classifier.predict(X_test)
   y_test=np.transpose(y_test>=0)
   report=np.mean(y_pred==y_test)
   target_names = ['nonface', 'face'] 
   report = classification_report(y_test[0], y_pred, target_names=target_names)
   print(report)
   with open("./report.txt","w") as f:
        f.write(report)

