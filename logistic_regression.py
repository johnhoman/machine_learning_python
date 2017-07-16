'''
Created on Jul 16, 2017

@author: homan
'''
from gradient_descent import GradientDescent
import math

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
