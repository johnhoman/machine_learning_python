'''
Created on Jul 16, 2017

@author: homan
'''

from logistic_regression import LogisticRegression
from hinge_loss import HingeLoss
from utils import lmap

class GradientDescentFactory():
    
    def __init__(self,data_path,labels_path):
        
        self._data   = self.read_data(data_path)
        self._labels = self.read_labels(labels_path)
    
    def __call__(self,algorithm = 'logistic'):
        
        self.get(algorithm=algorithm)
    
    def get(self,algorithm='logistic'):
    
        if 'logistic' in algorithm.lower():
            
            return LogisticRegression(self.data,self.labels)
        
        elif 'hinge' in algorithm.lower():
            
            return HingeLoss(self.data,self.labels)
    
    @property
    def data(self):
        
        return self._data
    
    @property
    def labels(self):
        
        return self._labels
    
    def read_data(self,data_path):
        
        with open(data_path,"rt") as file_:
    
            func  = lambda line:[1] + lmap(float,line.split())
        
            data  = lmap(func,file_.readlines())
                        
        return data    
            
    def read_labels(self,labels_path):
        
        with open(labels_path,"rt") as file_:
        
            func = lambda line:lmap(int,reversed(line.split()))
            
            labels = dict(map(func,file_.readlines()))
        
        return labels
