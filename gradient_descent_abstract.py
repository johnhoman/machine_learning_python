'''
Created on Jul 16, 2017

@author: homan
'''
from abc import ABCMeta,abstractmethod,abstractproperty

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
        