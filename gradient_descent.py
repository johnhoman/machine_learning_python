'''
Created on Jul 16, 2017

@author: homan
'''

from gradient_descent_abstract import GradientDescentAbstractBC
import random

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
    
        for i,row in enumerate(self.data):
            
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
