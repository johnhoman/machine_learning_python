'''
Created on Jul 16, 2017

@author: homan
'''
lmap = lambda func,*iterables:list(map(func,*iterables))
        
lzip = lambda *iterables:list(zip(*iterables))
