'''
Created on Jul 9, 2017

@author: homan
'''
import random
import sys
import os
import math
from abc import ABCMeta,abstractmethod,abstractproperty

lmap = lambda func,*iterables:list(map(func,*iterables))
        
lzip = lambda *iterables:list(zip(*iterables))

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

class GradientDescentAbstractBC(metaclass=ABCMeta):
    
    @abstractmethod
    def __call__(self):pass
    @abstractmethod
    def __len__ (self):pass
    @abstractmethod
    def __iter__(self):pass
    @abstractmethod
    def prediction(self,row):pass
    @abstractmethod
    def gradient_wrapper(self,i,row):pass
    @abstractmethod
    def loss_function_wrapper(self):pass
    @abstractmethod
    def dot_product  (self):pass
    @abstractproperty
    def klass  (self):pass
    @abstractproperty
    def data   (self):pass
    @abstractproperty
    def labels (self):pass
    @abstractproperty
    def size   (self):pass
    @abstractproperty
    def rows   (self):pass
    @abstractproperty
    def columns(self):pass
    @abstractproperty
    def T(self):pass
    @abstractproperty
    def eta(self):pass
    @abstractproperty
    def eta(self,value):pass
    @abstractproperty
    def w(self):pass
    @abstractproperty
    def w(self,value):pass
    @abstractmethod
    def update_w(self,gradient):pass
        
class GradientDescent(GradientDescentAbstractBC):
    
    def __init__(self,data,labels):
        
        super(GradientDescent,self).__init__()
        
        self._data = data
        
        self._labels = labels
        
        self._logging = list()
        
        self._w = [0.02 * random.random() - 0.01 for i in self.columns]
        
    def __call__(self,eta,stop,objective = 100000000000):
        
        self.train(eta,stop,objective,_print=True)
                
        normw = sum([i**2 for i in self.w[1:]])**2
        
        self.logging = "w = %s" % ",".join(map(str,self.w[1:]))
                        
        self.logging = "||w|| = %f" % normw
        
        self.logging = "w0 = %f" % self.w[0]
        
        self.logging = "distance to origin = %f" % abs(self.w[0]/normw)
        
        print("\n".join(self.logging))
        
        return "\n".join(self.logging)
      
    def train(self,eta,stop,objective = 100000000000,_print=False):
        
        self._eta = eta
        
        while True:

            gradient = self.gradient_wrapper()

            self.w = self.update_w(gradient)

            current = self.loss_function_wrapper()
            
            if _print:
                
                print("objective = {}".format(current))
            
            if abs(current - objective) > stop:
                
                objective = current
                
            else: 
                
                return
            
    def classify(self):
        
        str_ = []
    
        for i,row in enumerate(data):
            
            if i not in self.klass:
                
                dp = self.dot_product(row)
                
                str_ += ["{} {}".format(i,1 if dp > 0 else 0)]
    
    def __len__(self):
        
        return len(self.data)
    
    def __iter__(self):
        
        for row in self.data:
            
            yield row
    
    @property
    def logging(self):
        
        return self._logging
    
    @logging.setter
    def logging(self,value):
        
        self._logging.append(value)
    
    @property
    def klass(self):
        
        return self.labels
           
    @property
    def data(self):
        
        return self._data
    
    @property
    def labels(self):
        
        return self._labels
    
    @property
    def size(self):
        
        return len(self.data),len(self.data[0])
            
    @property
    def rows(self):
        
        return (i for i in range(self.size[0]))
    
    @property
    def columns(self):
        
        return (i for i in range(self.size[1]))
        
    @property
    def T(self):
        
        return self.lmap(list,self.lzip(*self.data))
    
    @property
    def eta(self):
        
        return self._eta
    
    @eta.setter
    def eta(self,value):
        
        self._eta = value
     
    @property
    def w(self):
        
        return self._w
    
    @w.setter
    def w(self,value):
        
        self._w = value
    
    def update_w(self,gradient):
        
        raise NotImplementedError() 
    
    def prediction(self,row,i):
            
        klass = 1 if self.dot_product(row) > 0 else 0
        
        return "%d %d" % (klass,i)
        
    def dot_product(self,row):    

        return sum([self.w[j]*row[j] for j in self.columns])
           
    def gradient_wrapper(self):
        
        raise NotImplementedError()
    
    def loss_function_wrapper(self): 
        
        raise NotImplementedError()

class LogisticRegression(GradientDescent):
     
    def gradient_wrapper(self):
        
        def gradient():
            
            delf = [0]*self.size[1]
            
            for i,row in enumerate(self.data):
            
                for j,value in enumerate(row):
                                    
                    y = self.klass[i]
                    
                    dot_product = self.dot_product(row)
                    
                    logistic = self.logistic_function(dot_product)
                    
                    delf[j] += (y - logistic)*logistic*(1 - logistic)*value
            
            return delf
            
        return gradient()
      
    def update_w(self,gradient):
        
        return [wt + self.eta*d for wt,d in zip(self._w,gradient)]
    
    def logistic_function(self,dot_product):
        
        return (1 + math.exp(-dot_product))**-1
    
    def loss_function_wrapper(self):        
        
        def likelyhood():
            
            negative_log_l = 0
            
            exponential = math.exp
            
            natural_log = math.log
             
            for i,row in enumerate(self.data):
                
                y = self.klass[i]
                
                dot_product = self.dot_product(row)
                
                logistic = self.logistic_function(dot_product)
                

                _1st_term = -y * natural_log(logistic)
                
                _2nd_term = (1 - y) * natural_log(exponential(-dot_product)*logistic)
                
                negative_log_l += (_1st_term - _2nd_term)
                
            return negative_log_l
        
        return likelyhood()

class HingeLoss(GradientDescent): 
    
    def __init__(self,data,labels):
        
        super(HingeLoss,self).__init__(data,labels)
        
        self._labels = {k:-1 if v is 0 else 1 for k,v in labels.items()}

    def gradient_wrapper(self):
        
        def hinge_loss_gradient():
            
            delf = [0]*self.size[1]
        
            for i,row in enumerate(self.data):
        
                dot_product = self.dot_product(row)
                
                for j,value in enumerate(row):
                    
                    if (1 - self.klass[i]*dot_product > 0):
                        
                        delf[j] -= self.klass[i]*value 
                        
            return delf
        
        return hinge_loss_gradient()
    
    def update_w(self,gradient):

        return [wt - self.eta*d for wt,d in zip(self.w,gradient)]
        
    def loss_function_wrapper(self):

        def hinge_loss():
            
            """i = (index,row)"""
            
            loss_fcn = lambda i:max(0,1 - self.klass[i[0]]*self.dot_product(i[1]))
            
            return sum(map(loss_fcn,enumerate(self.data)))
        
        return hinge_loss()
        


if __name__=="__main__":
    
    #path = r"C:\Users\homan\OneDrive\Documents\CS675-Machine Learning\data"
    
    data = sys.argv[1]#os.path.join(path,'input_data_assignment4')
    labels = sys.argv[2]#os.path.join(path,'label_data_assignment4')    
    
    factory = GradientDescentFactory(data,labels)
    
    logistic = factory.get('logistic')
    
    logistic(eta=0.01,stop=0.00001)
    
    
    
    
    