import pickle
import numpy as np
import math
import random
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier=weak_classifier 
        self.n_weakers_limit=n_weakers_limit
        self.classifier=[]   #基层分类器list
        self.weight_classifier=[] #基层分类器权重list
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        n_samples = X.shape[0]
        n_features = X.shape[1]
        w=np.ones(n_samples)
        for i in range(self.n_weakers_limit):#迭代次数为基分类器的个数
            classifier=self.weak_classifier(max_depth=3) 
            classifier.fit(X,y,w)
            score=1-classifier.score(X,y,w)   #score是正确率，1-score是错误率

            if(score>0.5):
                break
            classifier_weight=0.5*math.log((1-score)/(score+1e-6))  #基层分类器的权重,1e-6是为了防止分母为0
			#把训练出来的基层分类器和它的权重放入list
            self.weight_classifier.append(classifier_weight)
            self.classifier.append(classifier)

            pred=classifier.predict(X)
            #tmp=classifier_weight*y*pred
            tmp=np.multiply(pred,np.transpose(y)*classifier_weight) 
            tmp=np.exp(-tmp)
            tmp=tmp/np.sum(tmp)
			
            w=np.multiply(tmp,w).reshape([n_samples])
			
        pass
	

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        y=0
		
        for i in range(self.n_weakers_limit):
            if(i==0):
                y=self.weight_classifier[i]*self.classifier[i].predict(X)
            else:
                y=np.vstack((y,self.weight_classifier[i]*self.classifier[i].predict(X)))
        
		#y是各个基层分类器的预测值的加权平均值
        y=np.mean(y,0)  
        return y

        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        y_=self.predict_scores(X)
        y=(y_>=threshold)
        return y


        pass

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
