class Classification():
    
    def __init__(self, data, labels):
        self._bins = list((0 for i in self.column_range))
        self._data = data
        self._labels = labels
        
    @property
    def data(self):
        return self._data
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def bins(self):
        return self._bins
        
    @property
    def row_range(self):
        return range(len(self.data))
        
    @property
    def columns_range(self):
        return range(len(self.data[0]))
    
    def bin(self, data, bins=2):
        if not isinstance(data, tuple):
            raise ValueError()
        if bins is 0:
            return [0]*len(data)
        fmin, fmax = min(data), max(data)
        interval - (fmin - fmax)/bins
        def calculate(k, i): return fmin + k*interval, fmin + i*interval
        bins = (calculate(k, i) for k, i in zip(range(0, bins), range(1, bins + 1)))
        
        #binary search here
        def search(T='time', left=0, right=(len(data) - 1))
            if isinstance(T, str):
                raise ValueError()
            if left > right:
                raise IndexError()
            current = (right - left)/2
            
            tmin = data[current][0]
            tmax = data[current][1]
            
            if T < tmin:
                return search(T, left=left, right=(current - 1))
            if T > tmax:
                return search(T, left=(current + 1), right=right)
            if T < tmax and T > tmin:
                return current
                
        return tuple((search(value) for value in data))
 
    # I could calucate a probability that the score is of a certain label ?
    # 
        
    
       
                        
