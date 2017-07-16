'''
Created on Jul 16, 2017

@author: homan
'''
from gradient_descent import GradientDescent

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